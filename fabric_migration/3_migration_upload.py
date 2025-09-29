# COMMAND ----------
# MAGIC %md


# COMMAND ----------
# MAGIC %md
### Step 1: Setup and Configuration

# COMMAND ----------
import logging
import json
import time
import sys
from typing import Iterator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from openai import AzureOpenAI
from pyspark.sql import DataFrame, functions as F, Window
from pyspark.sql.types import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COMMAND ----------
# MAGIC %md
### Step 2: Main Execution Block

# COMMAND ----------

# --- Batch Processing Configuration ---
PROCESSING_BATCH_SIZE = 2

# --- Rate Limiting Configuration (within each batch) ---
MICRO_BATCH_SIZE = 100
DELAY_BETWEEN_BATCHES = 5  # seconds

print(f"Batch Config: Processing Batch Size={PROCESSING_BATCH_SIZE}, Micro-Batch Size={MICRO_BATCH_SIZE}, Delay={DELAY_BETWEEN_BATCHES}s")
sys.stdout.flush()

try:
    # === Step 2.1: Load Configurations ===
    print("\n--- Step 2.1: Loading Configurations ---")
    sys.stdout.flush()
    config_path = "/lakehouse/default/Files/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    cosmos_config = {
        "spark.cosmos.accountEndpoint": config.get("COSMOS_DB_ENDPOINT"),
        "spark.cosmos.accountKey": config.get("COSMOS_DB_KEY"),
        "spark.cosmos.database": config.get("COSMOS_DB_DATABASE_NAME"),
        "spark.cosmos.container": config.get("COSMOS_DB_SILVER_CONTAINER_NAME"),
        "spark.cosmos.write.strategy": "ItemOverwrite"
    }
    AZURE_OPENAI_ENDPOINT = config.get("AZURE_OPENAI_ENDPOINT2")
    AZURE_OPENAI_KEY = config.get("AZURE_OPENAI_KEY2")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = config.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT2")

    if not all(cosmos_config.values()) or not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_EMBEDDING_DEPLOYMENT]):
        raise ValueError("One or more required settings for Cosmos DB or Azure OpenAI are missing in config.json")
    print("Successfully loaded configurations.")
    sys.stdout.flush()

    # === Step 2.2: Read Data and Add Row Numbers ===
    print("\n--- Step 2.2: Reading data and preparing for batching ---")
    sys.stdout.flush()
    source_df = spark.read.table("silver_gdelt_events_detailed")
    
    window_spec = Window.orderBy("global_event_id")
    df_with_row_num = source_df.withColumn("row_num", F.row_number().over(window_spec))
    
    df_with_row_num.cache()
    total_records = df_with_row_num.count()
    print(f"Found {total_records} total records to process.")
    sys.stdout.flush()

    if total_records == 0:
        print("No data found. Exiting.")
        sys.stdout.flush()
    else:
        # === Step 2.3: Define Rate-Limited Embedding UDFs ===
        print("\n--- Step 2.3: Defining Rate-Limited Embedding UDFs ---")
        sys.stdout.flush()
        
        @F.pandas_udf(ArrayType(FloatType()))
        def generate_content_embeddings_udf(texts: pd.Series) -> pd.Series:
            client = AzureOpenAI(api_key=AZURE_OPENAI_KEY, api_version="2023-05-15", azure_endpoint=AZURE_OPENAI_ENDPOINT)
            deployment_name = AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            all_embeddings = {}
            non_empty_texts = texts[texts.notna() & (texts != "")]
            if not non_empty_texts.empty:
                text_list = non_empty_texts.tolist()
                for i in range(0, len(text_list), MICRO_BATCH_SIZE):
                    batch = text_list[i:i + MICRO_BATCH_SIZE]
                    try:
                        res = client.embeddings.create(input=batch, model=deployment_name)
                        embeddings = [item.embedding for item in res.data]
                        if embeddings:
                            sample_vector = [round(v, 4) for v in embeddings[0][:5]]
                            print(f"      -> Content vector generated. Sample: {sample_vector}...")
                            sys.stdout.flush()
                        all_embeddings.update(dict(zip(batch, embeddings)))
                    except Exception as e:
                        logger.warning(f"Content micro-batch failed. Error: {e}")
                    time.sleep(DELAY_BETWEEN_BATCHES)
            default_embedding = [0.0] * 1536
            return texts.apply(lambda x: all_embeddings.get(x, default_embedding))

        @F.pandas_udf(ArrayType(FloatType()))
        def generate_location_embeddings_udf(df_iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
            scaler = MinMaxScaler()
            client = AzureOpenAI(api_key=AZURE_OPENAI_KEY, api_version="2023-05-15", azure_endpoint=AZURE_OPENAI_ENDPOINT)
            deployment_name = AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            default_embedding = [0.0] * 1536
            for pdf in df_iterator:
                if pdf.empty: yield pd.Series([], dtype='object'); continue
                pdf_numeric = pdf.fillna(0)
                scaled_data = scaler.fit_transform(pdf_numeric)
                string_series = pd.Series(scaled_data.tolist(), index=pdf.index).apply(lambda x: json.dumps({"numerical_features": x}))
                if string_series.empty: yield pd.Series([], dtype='object'); continue
                all_embeddings = {}
                texts_to_embed = string_series.tolist()
                for i in range(0, len(texts_to_embed), MICRO_BATCH_SIZE):
                    batch = texts_to_embed[i:i + MICRO_BATCH_SIZE]
                    try:
                        res = client.embeddings.create(input=batch, model=deployment_name)
                        embeddings = [item.embedding for item in res.data]
                        if embeddings:
                            sample_vector = [round(v, 4) for v in embeddings[0][:5]]
                            print(f"      -> Location vector generated. Sample: {sample_vector}...")
                            sys.stdout.flush()
                        all_embeddings.update(dict(zip(batch, embeddings)))
                    except Exception as e:
                        logger.warning(f"Location micro-batch failed. Error: {e}")
                    time.sleep(DELAY_BETWEEN_BATCHES)
                yield string_series.apply(lambda x: all_embeddings.get(x, default_embedding))
        
        print("UDFs defined successfully.")
        sys.stdout.flush()

        # === Step 2.4: Main Processing Loop ===
        print(f"\n\n>>> 시작: 전체 {total_records}개 레코드를 {PROCESSING_BATCH_SIZE}개씩 배치 처리합니다. <<<")
        sys.stdout.flush()
        
        for i in range(0, total_records, PROCESSING_BATCH_SIZE):
            start_row = i + 1
            end_row = i + PROCESSING_BATCH_SIZE
            
            print(f"\n--- [진행 중] 배치 처리: {start_row} ~ {end_row} (전체 {total_records} 중) ---")
            sys.stdout.flush()

            print(f"    (1/4) 데이터 필터링..."); sys.stdout.flush()
            batch_df = df_with_row_num.where(F.col("row_num").between(start_row, end_row))

            print(f"    (2/4) 데이터 준비..."); sys.stdout.flush()
            df_for_rag = batch_df.fillna({'actor1_name': 'Unknown', 'actor2_name': 'Unknown', 'action_geo_fullname': 'Unknown', 'event_code': 'Unknown', 'mention_source_name': 'Unknown', 'event_date': '1970-01-01', 'goldstein_scale': 0.0, 'quad_class': 0})
            df_for_rag = df_for_rag.withColumn("event_date", F.col("event_date").cast(StringType()))
            df_for_rag = df_for_rag.withColumn("date_added", F.col("date_added").cast(StringType()))
            df_for_rag = df_for_rag.withColumn("processed_time", F.col("processed_time").cast(StringType()))
            df_for_rag = df_for_rag.withColumn("quad_class_str", F.when(F.col("quad_class") == 1, "Verbal Cooperation").when(F.col("quad_class") == 2, "Material Cooperation").when(F.col("quad_class") == 3, "Verbal Conflict").when(F.col("quad_class") == 4, "Material Conflict").otherwise("Unknown"))
            df_for_rag = df_for_rag.withColumn("content", F.format_string("On %s, an event categorized as '%s' involving '%s' and '%s' occurred in '%s'. The event has a Goldstein scale of %s, indicating its intensity. Event type: %s. Source: %s.", F.col("event_date"), F.col("quad_class_str"), F.col("actor1_name"), F.col("actor2_name"), F.col("action_geo_fullname"), F.col("goldstein_scale"), F.col("event_code"), F.col("mention_source_name")))
            numerical_cols = ["goldstein_scale", "avg_tone", "num_mentions", "num_sources", "num_articles", "action_geo_lat", "action_geo_long"]
            for col_name in numerical_cols: df_for_rag = df_for_rag.fillna(0, subset=[col_name])
            df_for_rag = df_for_rag.withColumnRenamed("global_event_id", "id").withColumn("id", F.col("id").cast(StringType()))

            print(f"    (3/4) 벡터 생성 중 (API 호출)..."); sys.stdout.flush()
            df_with_vectors = df_for_rag.withColumn("contentVector", generate_content_embeddings_udf(F.col("content")))\
                                        .withColumn("locationVector", generate_location_embeddings_udf(F.struct(*numerical_cols)))
            
            final_columns = ["id"] + [c for c in source_df.columns if c != "global_event_id"] + ["contentVector", "locationVector", "content"]
            final_batch_df = df_with_vectors.select(*final_columns)

            print(f"    (4/4) Cosmos DB에 저장 중..."); sys.stdout.flush()
            (final_batch_df.write
                .format("cosmos.oltp")
                .options(**cosmos_config)
                .mode("append")
                .save())
            
            processed_count = min(end_row, total_records)
            print(f"--- [완료] 배치 저장 성공. (누적 {processed_count} / {total_records} 처리) ---")
            sys.stdout.flush()

except Exception as e:
    logger.error(f"An error occurred during the historical vectorization job: {e}", exc_info=True)
    raise
finally:
    if 'df_with_row_num' in locals():
        df_with_row_num.unpersist()
    logger.info("Historical vectorization job finished.")

print("\n\n>>> 완료: 모든 배치 처리가 성공적으로 끝났습니다. <<<")
sys.stdout.flush()
# COMMAND ----------