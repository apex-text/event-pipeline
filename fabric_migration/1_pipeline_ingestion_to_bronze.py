# COMMAND ----------
# MAGIC %md
## 1. GDELT 3-Way Ingestion to Bronze Tables

# COMMAND ----------
import requests
import zipfile
import io
import logging
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from datetime import datetime, timedelta, timezone
from delta.tables import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
CHECKPOINT_PATH = "Files/checkpoints/producer/last_success_timestamp.txt"
DATA_TYPES_CONFIG = {
    "events": {"extension": ".export.CSV.zip", "table_name": "bronze_gdelt_events", "pk_index": 0},
    "mentions": {"extension": ".mentions.CSV.zip", "table_name": "bronze_gdelt_mentions", "pk_index": 0},
    "gkg": {"extension": ".gkg.csv.zip", "table_name": "bronze_gdelt_gkg", "pk_index": 0}
}

# COMMAND ----------
# MAGIC %md
### Step 1: Checkpoint Management

# COMMAND ----------
def get_start_time(default_start_time_str: str) -> str:
    try:
        return mssparkutils.fs.head(CHECKPOINT_PATH)
    except Exception as e:
        if "Path does not exist" in str(e) or "FileNotFoundException" in str(e):
            logger.warning("Checkpoint file not found. Using default start time.")
            return default_start_time_str
        else:
            raise

def save_end_time(timestamp_str: str):
    mssparkutils.fs.put(CHECKPOINT_PATH, timestamp_str, overwrite=True)
    logger.info(f"Checkpoint updated successfully with timestamp: {timestamp_str}")

print("--- Step 1: Determining Time Window ---")
now = datetime.now(timezone.utc)
minute_rounded = (now.minute // 15) * 15
end_time = now.replace(minute=minute_rounded, second=0, microsecond=0)
default_start_dt = end_time - timedelta(hours=1)
start_time_str = get_start_time(default_start_dt.isoformat())
end_time_str = end_time.isoformat()
print(f"Processing data from {start_time_str} to {end_time_str}")

# COMMAND ----------
# MAGIC %md
### Step 2: Generate GDELT URLs

# COMMAND ----------
def get_gdelt_urls_for_period(start_str, end_str) -> dict:
    logger.info(f"Generating GDELT URLs from {start_str} to {end_str}")
    start_dt = datetime.fromisoformat(start_str)
    end_dt = datetime.fromisoformat(end_str)
    start_minute_rounded = (start_dt.minute // 15) * 15
    current_dt = start_dt.replace(minute=start_minute_rounded, second=0, microsecond=0)
    gdelt_urls = {dt: [] for dt in DATA_TYPES_CONFIG.keys()}
    base_url = "http://data.gdeltproject.org/gdeltv2/"
    while current_dt < end_dt:
        timestamp_str = current_dt.strftime("%Y%m%d%H%M%S")
        for data_type, config in DATA_TYPES_CONFIG.items():
            file_name = f"{timestamp_str}{config['extension']}"
            gdelt_urls[data_type].append(f"{base_url}{file_name}")
        current_dt += timedelta(minutes=15)
    return gdelt_urls

print("\n--- Step 2: Generating URLs ---")
all_urls = get_gdelt_urls_for_period(start_time_str, end_time_str)
for data_type, urls in all_urls.items():
    print(f"Found {len(urls)} URLs for {data_type.upper()}")

# COMMAND ----------
# MAGIC %md
### Step 3: Process and Ingest Each Data Type

# COMMAND ----------
def process_and_ingest(data_type: str, urls: list):
    config = DATA_TYPES_CONFIG[data_type]
    table_name = config["table_name"]
    pk_index = config["pk_index"]
    
    print(f"\n--- Step 3: Starting ingestion for {data_type.upper()} ---")
    if not urls:
        print(f"No URLs to process for {data_type.upper()}. Skipping.")
        return

    all_data = []
    for i, url in enumerate(urls, 1):
        print(f"Downloading {data_type.upper()} file {i}/{len(urls)}: {url.split('/')[-1]}")
        try:
            response = requests.get(url, timeout=120)
            if response.status_code == 404:
                logger.warning(f"URL not found (404): {url}")
                continue
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                filename = z.namelist()[0]
                # ** FIX: Use a lenient decoding strategy to handle character errors **
                content = z.open(filename).read().decode('utf-8', errors='ignore')
                all_data.extend(content.strip().split('\n'))
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            continue
    
    if not all_data:
        print(f"No data was downloaded for {data_type.upper()}. Skipping.")
        return
    
    print(f"Downloaded a total of {len(all_data)} rows for {data_type.upper()}. Processing with Spark...")
    rdd = spark.sparkContext.parallelize(all_data)
    df_lines = rdd.map(lambda line: (line.split('\t'),)).toDF(["bronze_data"])

    df_with_meta = df_lines.withColumn("ingestion_time", F.current_timestamp()) \
                           .withColumn("primary_key", F.col("bronze_data").getItem(pk_index))

    df_deduplicated = df_with_meta.dropDuplicates(["primary_key"])
    
    record_count = df_deduplicated.count()
    print(f"Writing {record_count} unique records to Bronze table: {table_name}...")
    
    spark.sql(f"CREATE TABLE IF NOT EXISTS {table_name} (bronze_data ARRAY<STRING>, ingestion_time TIMESTAMP, primary_key STRING) USING delta")
    delta_table = DeltaTable.forName(spark, table_name)
    
    (delta_table.alias("target")
     .merge(df_deduplicated.alias("source"), "target.primary_key = source.primary_key")
     .whenMatchedUpdateAll()
     .whenNotMatchedInsertAll()
     .execute())
    
    print(f"--- Finished ingestion for {data_type.upper()} ---")

all_success = True
for data_type, urls in all_urls.items():
    try:
        process_and_ingest(data_type, urls)
    except Exception as e:
        logger.error(f"CRITICAL FAILURE during ingestion of {data_type.upper()}: {e}", exc_info=True)
        all_success = False
        
# COMMAND ----------
# MAGIC %md
### Step 4: Update Checkpoint

# COMMAND ----------
print("\n--- Step 4: Updating Checkpoint ---")
if all_success:
    save_end_time(end_time_str)
    print(f"Successfully updated checkpoint to {end_time_str}")
else:
    logger.error("One or more data types failed to ingest. Checkpoint will NOT be updated.")
    raise RuntimeError("Ingestion pipeline failed for one or more data types.")

print("\n--- Ingestion Notebook Finished ---")
# COMMAND ----------