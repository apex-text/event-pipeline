# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. GDELT 3-Way Bronze to Silver Pipeline (Definitive Final Version)
# MAGIC 
# MAGIC This notebook transforms and joins the three bronze tables into two comprehensive silver tables, ensuring ALL original columns are preserved.
# MAGIC 
# MAGIC **Functionality:**
# MAGIC 1.  Reads new data from the three Bronze tables.
# MAGIC 2.  Transforms each dataset into a rich Silver format with all columns.
# MAGIC 3.  Saves the transformed Events data to `silver_gdelt_events`.
# MAGIC 4.  Performs a 3-way join.
# MAGIC 5.  Saves the final joined data to `silver_gdelt_events_detailed`.

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 1: Setup and Configuration

# COMMAND ----------
# Assumes delta-spark is installed via Workspace settings
import logging
import os
import requests
import json
from typing import Iterator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from openai import AzureOpenAI
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import *
from delta.tables import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 2: Define Transformation and Notification Functions

# COMMAND ----------
def clean_string_fields(df: DataFrame) -> DataFrame:
    string_columns = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    for col_name in string_columns:
        df = df.withColumn(col_name, F.when(F.trim(F.col(col_name)) == "", None).otherwise(F.col(col_name)))
    return df

def transform_events_to_silver(bronze_df: DataFrame) -> DataFrame:
    logger.info("Transforming Events data to a rich Silver format with ALL columns...")
    min_expected_columns = 61
    valid_df = bronze_df.filter(F.size("bronze_data") >= min_expected_columns)
    
    silver_df = valid_df.select(
        # All columns from the original events_transformer.py
        F.col("bronze_data")[0].cast(LongType()).alias("global_event_id"),
        F.col("bronze_data")[1].alias("event_date_str"),
        F.col("bronze_data")[5].alias("actor1_code"),
        F.col("bronze_data")[6].alias("actor1_name"),
        F.col("bronze_data")[7].alias("actor1_country_code"),
        F.col("bronze_data")[8].alias("actor1_known_group_code"),
        F.col("bronze_data")[9].alias("actor1_ethnic_code"),
        F.col("bronze_data")[10].alias("actor1_religion1_code"),
        F.col("bronze_data")[11].alias("actor1_religion2_code"),
        F.col("bronze_data")[12].alias("actor1_type1_code"),
        F.col("bronze_data")[13].alias("actor1_type2_code"),
        F.col("bronze_data")[14].alias("actor1_type3_code"),
        F.col("bronze_data")[15].alias("actor2_code"),
        F.col("bronze_data")[16].alias("actor2_name"),
        F.col("bronze_data")[17].alias("actor2_country_code"),
        F.col("bronze_data")[18].alias("actor2_known_group_code"),
        F.col("bronze_data")[19].alias("actor2_ethnic_code"),
        F.col("bronze_data")[20].alias("actor2_religion1_code"),
        F.col("bronze_data")[21].alias("actor2_religion2_code"),
        F.col("bronze_data")[22].alias("actor2_type1_code"),
        F.col("bronze_data")[23].alias("actor2_type2_code"),
        F.col("bronze_data")[24].alias("actor2_type3_code"),
        F.col("bronze_data")[25].cast(IntegerType()).alias("is_root_event"),
        F.col("bronze_data")[26].alias("event_code"),
        F.col("bronze_data")[27].alias("event_base_code"),
        F.col("bronze_data")[28].alias("event_root_code"),
        F.col("bronze_data")[29].cast(IntegerType()).alias("quad_class"),
        F.col("bronze_data")[30].cast(DoubleType()).alias("goldstein_scale"),
        F.col("bronze_data")[31].cast(IntegerType()).alias("num_mentions"),
        F.col("bronze_data")[32].cast(IntegerType()).alias("num_sources"),
        F.col("bronze_data")[33].cast(IntegerType()).alias("num_articles"),
        F.col("bronze_data")[34].cast(DoubleType()).alias("avg_tone"),
        F.col("bronze_data")[35].cast(IntegerType()).alias("actor1_geo_type"),
        F.col("bronze_data")[36].alias("actor1_geo_fullname"),
        F.col("bronze_data")[37].alias("actor1_geo_country_code"),
        F.col("bronze_data")[38].alias("actor1_geo_adm1_code"),
        F.col("bronze_data")[39].alias("actor1_geo_adm2_code"),
        F.col("bronze_data")[40].cast(DoubleType()).alias("actor1_geo_lat"),
        F.col("bronze_data")[41].cast(DoubleType()).alias("actor1_geo_long"),
        F.col("bronze_data")[42].alias("actor1_geo_feature_id"),
        F.col("bronze_data")[43].cast(IntegerType()).alias("actor2_geo_type"),
        F.col("bronze_data")[44].alias("actor2_geo_fullname"),
        F.col("bronze_data")[45].alias("actor2_geo_country_code"),
        F.col("bronze_data")[46].alias("actor2_geo_adm1_code"),
        F.col("bronze_data")[47].alias("actor2_geo_adm2_code"),
        F.col("bronze_data")[48].cast(DoubleType()).alias("actor2_geo_lat"),
        F.col("bronze_data")[49].cast(DoubleType()).alias("actor2_geo_long"),
        F.col("bronze_data")[50].alias("actor2_geo_feature_id"),
        F.col("bronze_data")[51].cast(IntegerType()).alias("action_geo_type"),
        F.col("bronze_data")[52].alias("action_geo_fullname"),
        F.col("bronze_data")[53].alias("action_geo_country_code"),
        F.col("bronze_data")[54].alias("action_geo_adm1_code"),
        F.col("bronze_data")[55].alias("action_geo_adm2_code"),
        F.col("bronze_data")[56].cast(DoubleType()).alias("action_geo_lat"),
        F.col("bronze_data")[57].cast(DoubleType()).alias("action_geo_long"),
        F.col("bronze_data")[58].alias("action_geo_feature_id"),
        F.col("bronze_data")[59].alias("date_added_str"),
        F.col("bronze_data")[60].alias("source_url"),
        F.col("ingestion_time").alias("processed_time")
    ).filter(F.col("global_event_id").isNotNull())
    
    silver_df = silver_df.withColumn("event_date", F.to_date(F.col("event_date_str"), "yyyyMMdd")) \
                         .withColumn("date_added", F.to_timestamp(F.col("date_added_str"), "yyyyMMddHHmmss")) \
                         .drop("event_date_str", "date_added_str")
    
    silver_df = silver_df.fillna(0, subset=["num_mentions", "num_sources", "num_articles"])
    return clean_string_fields(silver_df)

def transform_mentions_to_silver(bronze_df: DataFrame) -> DataFrame:
    logger.info("Transforming Mentions data to a rich Silver format...")
    min_expected_columns = 16
    valid_df = bronze_df.filter(F.size("bronze_data") >= min_expected_columns)
    silver_df = valid_df.select(
        F.col("bronze_data")[0].cast(LongType()).alias("global_event_id"),
        F.col("bronze_data")[3].cast(IntegerType()).alias("mention_type"),
        F.col("bronze_data")[4].alias("mention_source_name"),
        F.col("bronze_data")[5].alias("mention_identifier"),
        F.col("bronze_data")[11].cast(IntegerType()).alias("confidence"),
        F.col("bronze_data")[13].cast(DoubleType()).alias("mention_doc_tone")
    ).filter(F.col("global_event_id").isNotNull())
    return clean_string_fields(silver_df)

def transform_gkg_to_silver(bronze_df: DataFrame) -> DataFrame:
    logger.info("Transforming GKG data to a rich Silver format...")
    min_expected_columns = 27
    valid_df = bronze_df.filter(F.size("bronze_data") >= min_expected_columns)
    silver_df = valid_df.select(
        F.col("bronze_data")[0].alias("gkg_record_id"),
        F.col("bronze_data")[4].alias("document_identifier"),
        F.col("bronze_data")[7].alias("themes"),
        F.col("bronze_data")[9].alias("locations"),
        F.col("bronze_data")[11].alias("persons"),
        F.col("bronze_data")[13].alias("organizations")
    ).filter(F.col("gkg_record_id").isNotNull())
    return clean_string_fields(silver_df)

def notify_gdelt_anomalies(silver_df: DataFrame, DISCORD_WEBHOOK_URL: str, MS_TEAMS_WEBHOOK_URL: str):
    """
    GDELT DataFrame에서 goldstein_scale 이상치를 감지하고 구성된 웹훅(Discord, MS Teams)으로 알림을 보냅니다.
    URL이 중복일 경우 하나로 합쳐서 메시지를 보냅니다.
    """
    if not DISCORD_WEBHOOK_URL and not MS_TEAMS_WEBHOOK_URL:
        logger.warning("알림을 위한 웹훅 URL이 설정되지 않았습니다. 건너뜁니다.")
        print("⚠️ 알림을 위한 웹훅 URL이 설정되지 않았습니다. 건너뜁니다.")
        return

    try:
        outliers_df = silver_df.filter(F.col("goldstein_scale") >= 8.5).select(
            "global_event_id", "source_url", "goldstein_scale"
        )
        outliers_count = outliers_df.count()

        if outliers_count > 0:
            logger.info(f"📢 {outliers_count}개의 이상치를 발견했습니다. 그룹화하여 알림을 보냅니다...")
            print(f"📢 {outliers_count}개의 이상치를 발견했습니다. 그룹화하여 알림을 보냅니다...")
            grouped_outliers_df = outliers_df.groupBy("source_url").agg(
                F.collect_list("global_event_id").alias("event_ids"),
                F.min("goldstein_scale").alias("min_goldstein_scale"),
            )
            total_urls = grouped_outliers_df.count()
            title = f"🚨 GDELT 이벤트 이상치 탐지 ({outliers_count}건 / {total_urls}개 URL) 🚨"
            message_lines = [title]
            outliers_to_show = grouped_outliers_df.limit(5).collect()
            for row in outliers_to_show:
                ids_str = ", ".join(map(str, row["event_ids"]))
                message_lines.append(
                    f"  - IDs: {ids_str}, 영향력: {row['min_goldstein_scale']:.2f}, URL: {row['source_url']}"
                )
            if total_urls > 5:
                message_lines.append(f"  ... 외 {total_urls - 5}개 URL 더 있습니다.")
            message = "\n".join(message_lines)

            if DISCORD_WEBHOOK_URL:
                try:
                    discord_message = message.replace("URL: ", "URL: <") + ">"
                    payload = {"content": discord_message}
                    response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
                    response.raise_for_status()
                    logger.info("🚀 Discord 알림을 성공적으로 보냈습니다.")
                except Exception as e:
                    logger.error(f"❌ Discord 알림 전송 중 오류 발생: {e}", exc_info=True)

            if MS_TEAMS_WEBHOOK_URL:
                try:
                    teams_message = message.replace("\n", "<br>")
                    payload = {"text": teams_message} # MS Teams uses "text" key
                    response = requests.post(MS_TEAMS_WEBHOOK_URL, json=payload)
                    response.raise_for_status()
                    logger.info("🚀 Microsoft Teams 알림을 성공적으로 보냈습니다.")
                except Exception as e:
                    logger.error(f"❌ Microsoft Teams 알림 전송 중 오류 발생: {e}", exc_info=True)
        else:
            logger.info("✅ 이상치를 발견하지 않았습니다.")
            print("✅ 이상치를 발견하지 않았습니다.")
    except Exception as e:
        logger.error(f"❌ 알림 처리 중 오류 발생: {e}", exc_info=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 3: Main Execution Block

# COMMAND ----------
try:
    print("\n--- Step 3.1: Reading New Data from Bronze Tables ---")
    twenty_minutes_ago = F.expr("current_timestamp() - interval 20 minutes")
    df_events_bronze = spark.read.table("bronze_gdelt_events").filter(F.col("ingestion_time") >= twenty_minutes_ago)
    df_mentions_bronze = spark.read.table("bronze_gdelt_mentions").filter(F.col("ingestion_time") >= twenty_minutes_ago)
    df_gkg_bronze = spark.read.table("bronze_gdelt_gkg").filter(F.col("ingestion_time") >= twenty_minutes_ago)
    
    events_count = df_events_bronze.count()
    print(f"Found {events_count} new records in bronze_gdelt_events.")
    print(f"Found {df_mentions_bronze.count()} new records in bronze_gdelt_mentions.")
    print(f"Found {df_gkg_bronze.count()} new records in bronze_gdelt_gkg.")

    if events_count == 0:
        print("\nNo new events data found. Exiting gracefully.")
    else:
        print("\n--- Step 3.2: Transforming Data to Rich Silver ---")
        events_silver = transform_events_to_silver(df_events_bronze)
        mentions_silver = transform_mentions_to_silver(df_mentions_bronze)
        gkg_silver = transform_gkg_to_silver(df_gkg_bronze)
        events_silver.cache()

        print("\n--- Step 3.3: Saving to silver_gdelt_events ---")
        if not spark.catalog.tableExists("silver_gdelt_events"):
            spark.createDataFrame([], events_silver.schema).write.format("delta").mode("overwrite").saveAsTable("silver_gdelt_events")
        DeltaTable.forName(spark, "silver_gdelt_events").alias("t").merge(
            events_silver.alias("s"), "t.global_event_id = s.global_event_id"
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        print(f"Successfully merged {events_silver.count()} records into silver_gdelt_events.")
        
        print("\n--- Step 3.4: Performing 3-Way Join ---")
        joined_df = events_silver.join(mentions_silver, "global_event_id", "left") \
                                 .join(gkg_silver, events_silver.source_url == gkg_silver.document_identifier, "left")
        joined_df.cache()
        joined_count = joined_df.count()
        print(f"Joined data contains {joined_count} records.")

        print("\n--- Step 3.5: Saving to silver_gdelt_events_detailed ---")
        if not spark.catalog.tableExists("silver_gdelt_events_detailed"):
             spark.createDataFrame([], joined_df.schema).write.format("delta").saveAsTable("silver_gdelt_events_detailed")
        DeltaTable.forName(spark, "silver_gdelt_events_detailed").alias("t").merge(
            joined_df.alias("s"), "t.global_event_id = s.global_event_id"
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        print(f"Successfully merged {joined_count} records into silver_gdelt_events_detailed.")

        # COMMAND ----------
        # MAGIC %md
        # MAGIC ### Step 4: Generate Hybrid Vectors, Save to Cosmos DB, and Notify

        # COMMAND ----------
        # --- Main Logic for Vectorization and Notification ---
        if 'joined_df' in locals() and joined_count > 0:
            try:
                # === Step 4.1: Load Configurations ===
                print("\n--- Step 4.1: Loading Configurations ---")
                config_path = "/lakehouse/default/Files/config.json"
                with open(config_path, "r") as f: config = json.load(f)
                
                cosmos_config = {
                    "spark.cosmos.accountEndpoint": config.get("COSMOS_DB_ENDPOINT"),
                    "spark.cosmos.accountKey": config.get("COSMOS_DB_KEY"),
                    "spark.cosmos.database": config.get("COSMOS_DB_DATABASE_NAME"),
                    "spark.cosmos.container": config.get("COSMOS_DB_SILVER_CONTAINER_NAME"),
                    "spark.cosmos.write.strategy": "ItemOverwrite"
                }
                AZURE_OPENAI_ENDPOINT = config.get("AZURE_OPENAI_ENDPOINT")
                AZURE_OPENAI_KEY = config.get("AZURE_OPENAI_KEY")
                AZURE_OPENAI_EMBEDDING_DEPLOYMENT = config.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
                DISCORD_WEBHOOK_URL = config.get("DISCORD_WEBHOOK_URL")
                MS_TEAMS_WEBHOOK_URL = config.get("MS_TEAMS_WEBHOOK_URL")
                
                if not all(cosmos_config.values()) or not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_EMBEDDING_DEPLOYMENT]):
                    raise ValueError("One or more required settings for Cosmos DB or Azure OpenAI are missing in config.json")
                print("Successfully loaded configurations.")

                # === Step 4.2: Define Batch-Optimized Embedding UDFs ===
                print("\n--- Step 4.2: Defining Batch-Optimized Embedding UDFs ---")
                
                @F.pandas_udf(ArrayType(FloatType()))
                def generate_content_embeddings_udf(texts: pd.Series) -> pd.Series:
                    client = AzureOpenAI(api_key=AZURE_OPENAI_KEY, api_version="2023-05-15", azure_endpoint=AZURE_OPENAI_ENDPOINT)
                    deployment_name = AZURE_OPENAI_EMBEDDING_DEPLOYMENT
                    non_empty_texts = texts[texts.notna() & (texts != "")]
                    embeddings_dict = {}
                    if not non_empty_texts.empty:
                        try:
                            res = client.embeddings.create(input=non_empty_texts.tolist(), model=deployment_name)
                            embeddings = [item.embedding for item in res.data]
                            embeddings_dict = dict(zip(non_empty_texts, embeddings))
                        except Exception as e:
                            print(f"Warning: Batch content embedding failed. {e}")
                    default_embedding = [0.0] * 1536
                    return texts.apply(lambda x: embeddings_dict.get(x, default_embedding))

                @F.pandas_udf(ArrayType(FloatType()))
                def generate_location_embeddings_udf(df_iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
                    scaler = MinMaxScaler()
                    client = AzureOpenAI(api_key=AZURE_OPENAI_KEY, api_version="2023-05-15", azure_endpoint=AZURE_OPENAI_ENDPOINT)
                    deployment_name = AZURE_OPENAI_EMBEDDING_DEPLOYMENT
                    default_embedding = [0.0] * 1536
                    for pdf in df_iterator:
                        if pdf.empty:
                            yield pd.Series([], dtype='object')
                            continue
                        pdf = pdf.fillna(0)
                        scaled_data = scaler.fit_transform(pdf)
                        string_series = pd.Series(scaled_data.tolist()).apply(lambda x: json.dumps({"numerical_features": x}))
                        if string_series.empty:
                            yield pd.Series([], dtype='object')
                            continue
                        texts_to_embed = string_series.tolist()
                        embeddings_dict = {}
                        try:
                            res = client.embeddings.create(input=texts_to_embed, model=deployment_name)
                            embeddings = [item.embedding for item in res.data]
                            embeddings_dict = dict(zip(texts_to_embed, embeddings))
                        except Exception as e:
                            print(f"Warning: Batch location embedding failed. {e}")
                        yield string_series.apply(lambda x: embeddings_dict.get(x, default_embedding))
                
                print("UDFs defined successfully.")

                # === Step 4.3: Prepare DataFrame for RAG ===
                print("\n--- Step 4.3: Preparing Data for Vectorization ---")
                df_for_rag = joined_df.fillna({'actor1_name': 'Unknown', 'actor2_name': 'Unknown', 'action_geo_fullname': 'Unknown', 'event_code': 'Unknown', 'mention_source_name': 'Unknown', 'event_date': '1970-01-01', 'goldstein_scale': 0.0, 'quad_class': 0})
                
                # Explicitly cast event_date to a string to preserve its format in Cosmos DB
                df_for_rag = df_for_rag.withColumn("event_date", F.col("event_date").cast(StringType()))

                df_for_rag = df_for_rag.withColumn("quad_class_str", F.when(F.col("quad_class") == 1, "Verbal Cooperation").when(F.col("quad_class") == 2, "Material Cooperation").when(F.col("quad_class") == 3, "Verbal Conflict").when(F.col("quad_class") == 4, "Material Conflict").otherwise("Unknown"))
                df_for_rag = df_for_rag.withColumn("content", F.format_string("On %s, an event categorized as '%s' involving '%s' and '%s' occurred in '%s'. The event has a Goldstein scale of %s, indicating its intensity. Event type: %s. Source: %s.", F.col("event_date").cast(StringType()), F.col("quad_class_str"), F.col("actor1_name"), F.col("actor2_name"), F.col("action_geo_fullname"), F.col("goldstein_scale"), F.col("event_code"), F.col("mention_source_name")))
                
                numerical_cols = ["goldstein_scale", "avg_tone", "num_mentions", "num_sources", "num_articles", "action_geo_lat", "action_geo_long"]
                for col_name in numerical_cols:
                    df_for_rag = df_for_rag.fillna(0, subset=[col_name])
                
                df_for_rag = df_for_rag.withColumnRenamed("global_event_id", "id").withColumn("id", F.col("id").cast(StringType()))
                print("Data preparation complete.")

                # === Step 4.4: Generate Hybrid Vectors ===
                print("\n--- Step 4.4: Generating Hybrid Vectors... ---")
                df_with_vectors = df_for_rag.withColumn("contentVector", generate_content_embeddings_udf(F.col("content"))) \
                                            .withColumn("locationVector", generate_location_embeddings_udf(F.struct(*numerical_cols)))
                
                final_columns = ["id"] + [c for c in joined_df.columns if c != "global_event_id"] + ["contentVector", "locationVector", "content"]
                df_final = df_with_vectors.select(*final_columns)
                df_final.cache()
                final_count = df_final.count()
                print(f"Vector generation complete. {final_count} records are ready.")

                # === Step 4.5: Write to Cosmos DB ===
                print(f"\n--- Step 4.5: Writing {final_count} documents to Cosmos DB ---")
                (df_final.write.format("cosmos.oltp").options(**cosmos_config).mode("append").save())
                print(">>> Successfully wrote vectorized data to Cosmos DB! <<<")
                df_final.unpersist()

                # === Step 4.6: Anomaly Detection and Notification ===
                print("\n--- Running Anomaly Detection and Notification ---")
                notify_gdelt_anomalies(events_silver, DISCORD_WEBHOOK_URL, MS_TEAMS_WEBHOOK_URL)
                print("--- Anomaly Detection Finished ---")

            except Exception as e_rag:
                logger.error(f"An error occurred during the RAG/Notification pipeline step: {e_rag}", exc_info=True)
                raise e_rag

except Exception as e:
    logger.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)
    raise
finally:
    if 'events_silver' in locals(): events_silver.unpersist()
    if 'joined_df' in locals(): joined_df.unpersist()
    logger.info("Pipeline run finished.")

print("\n--- Bronze to Silver Pipeline Finished ---")
# COMMAND ----------
