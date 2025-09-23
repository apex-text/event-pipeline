# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. GDELT Bronze to Silver Pipeline (Full Version)
# 
# This notebook replaces the Spark job `gdelt_15min_to_silver.py` and the notification logic.
# 
# **Functionality:**
# 1.  Reads new data from the `bronze_gdelt_events` table.
# 2.  Transforms the raw data array into a structured silver format.
# 3.  Detects anomalies based on `avg_tone`.
# 4.  Sends a notification to a webhook if anomalies are found.
# 5.  Appends the processed data to the `silver_gdelt_events` table.
# 
# **Setup:**
# - This notebook should be scheduled to run every 15 minutes.
# - Webhook URLs are configured in `Files/config/config.json` in the Lakehouse.

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 1: Setup and Configuration

# COMMAND ----------
# %pip install requests

# COMMAND ----------
import os
import requests
import logging
import json
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import *

# --- Load configurations from Lakehouse file ---
try:
    # The path /lakehouse/default/ points to the root of the default Lakehouse attached to the notebook
    with open("/lakehouse/default/Files/config.json", "r") as f:
        config = json.load(f)
    
    DISCORD_WEBHOOK_URL = config.get("DISCORD_WEBHOOK_URL", "")
    MS_TEAMS_WEBHOOK_URL = config.get("MS_TEAMS_WEBHOOK_URL", "")
    
    logging.info("Successfully loaded configurations from Lakehouse.")

except FileNotFoundError:
    logging.warning("config.json not found in Files/config/. Webhook notifications will be disabled.")
    DISCORD_WEBHOOK_URL = ""
    MS_TEAMS_WEBHOOK_URL = ""
# ------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 2: Define Transformation and Notification Functions
# | 
# These functions are adapted directly from the original project scripts.

# COMMAND ----------
def transform_raw_to_silver(raw_df: DataFrame) -> DataFrame:
    """Transforms raw bronze data to a structured silver format based on GDELT 2.0 event schema."""
    logger.info("Starting transformation from bronze to silver...")
    
    min_expected_columns = 61 # The GDELT event format has at least 61 columns
    valid_df = raw_df.filter(F.size("raw_data") >= min_expected_columns)
    
    # This transformation logic is based on the original gdelt_15min_to_silver.py script
    silver_df = valid_df.select(
        F.col("raw_data")[0].cast(LongType()).alias("global_event_id"),
        F.col("raw_data")[1].alias("event_date_str"),
        F.col("raw_data")[5].alias("actor1_code"),
        F.col("raw_data")[6].alias("actor1_name"),
        F.col("raw_data")[7].alias("actor1_country_code"),
        F.col("raw_data")[15].alias("actor2_code"),
        F.col("raw_data")[16].alias("actor2_name"),
        F.col("raw_data")[17].alias("actor2_country_code"),
        F.col("raw_data")[25].cast(IntegerType()).alias("is_root_event"),
        F.col("raw_data")[26].alias("event_code"),
        F.col("raw_data")[27].alias("event_base_code"),
        F.col("raw_data")[28].alias("event_root_code"),
        F.col("raw_data")[29].cast(IntegerType()).alias("quad_class"),
        F.col("raw_data")[30].cast(DoubleType()).alias("goldstein_scale"),
        F.col("raw_data")[31].cast(IntegerType()).alias("num_mentions"),
        F.col("raw_data")[32].cast(IntegerType()).alias("num_sources"),
        F.col("raw_data")[33].cast(IntegerType()).alias("num_articles"),
        F.col("raw_data")[34].cast(DoubleType()).alias("avg_tone"),
        F.col("raw_data")[59].alias("date_added_str"),
        F.col("raw_data")[60].alias("source_url"), # Use the URL from the data row itself
        F.col("source_file"), # Metadata from bronze table
        F.col("ingestion_time").alias("processed_time") # Metadata from bronze table
    ).filter(F.col("global_event_id").isNotNull())

    silver_df = (
        silver_df.withColumn("event_date", F.to_date(F.col("event_date_str"), "yyyyMMdd"))
        .withColumn("date_added", F.to_timestamp(F.col("date_added_str"), "yyyyMMddHHmmss"))
        .drop("event_date_str", "date_added_str")
    )

    string_columns = [f.name for f in silver_df.schema.fields if isinstance(f.dataType, StringType)]
    for col_name in string_columns:
        silver_df = silver_df.withColumn(col_name, F.when(F.trim(F.col(col_name)) == "", None).otherwise(F.col(col_name)))

    silver_df = silver_df.fillna({"num_mentions": 0, "num_sources": 0, "num_articles": 0})
    
    logger.info("Transformation complete.")
    return silver_df

def notify_anomalies(silver_df: DataFrame):
    """Detects anomalies in the DataFrame and sends notifications."""
    if not DISCORD_WEBHOOK_URL and not MS_TEAMS_WEBHOOK_URL:
        logger.warning("No webhook URL is configured. Skipping notification.")
        return

    logger.info("Checking for anomalies (avg_tone <= -10)...")
    outliers_df = silver_df.filter(F.col("avg_tone") <= -10).select("global_event_id", "source_url", "avg_tone")
    outliers_count = outliers_df.count()

    if outliers_count > 0:
        logger.info(f"Found {outliers_count} anomalies. Preparing notification.")
        grouped_outliers_df = outliers_df.groupBy("source_url").agg(
            F.collect_list("global_event_id").alias("event_ids"),
            F.min("avg_tone").alias("min_avg_tone")
        )
        total_urls = grouped_outliers_df.count()
        title = f"🚨 GDELT 이벤트 이상치 탐지 ({outliers_count} 건 / {total_urls}개 URL) 🚨"
        message_lines = [title]
        outliers_to_show = grouped_outliers_df.limit(5).collect()
        for row in outliers_to_show:
            ids_str = ", ".join(map(str, row['event_ids']))
            message_lines.append(f"  - IDs: {ids_str}, Tone: {row['min_avg_tone']:.2f}, URL: {row['source_url']}")
        if total_urls > 5:
            message_lines.append(f"  ... and {total_urls - 5} more URLs.")
        message = "\n".join(message_lines)

        if DISCORD_WEBHOOK_URL:
            try:
                payload = {"content": message}
                response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
                response.raise_for_status()
                logger.info("Successfully sent notification to Discord.")
            except Exception as e:
                logger.error(f"Failed to send Discord notification: {e}", exc_info=True)
        if MS_TEAMS_WEBHOOK_URL:
            try:
                # ** FIX: Replace newline characters with <br> for Teams **
                teams_message = message.replace("\n", "<br>")
                payload = {"content": teams_message}
                response = requests.post(MS_TEAMS_WEBHOOK_URL, json=payload, timeout=10)
                response.raise_for_status()
                logger.info("Successfully sent notification to Microsoft Teams.")
            except Exception as e:
                logger.error(f"Failed to send Microsoft Teams notification: {e}", exc_info=True)
    else:
        logger.info("No anomalies found.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 3: Main Execution Block

# COMMAND ----------
try:
    logger.info("Reading new data from bronze_gdelt_events table...")
    bronze_df = spark.read.table("bronze_gdelt_events")
    
    twenty_minutes_ago = F.expr("current_timestamp() - interval 20 minutes")
    df_new_bronze = bronze_df.filter(F.col("ingestion_time") >= twenty_minutes_ago)
    df_new_bronze.cache()
    
    record_count = df_new_bronze.count()
    if record_count == 0:
        logger.info("No new records found in the bronze table. Exiting gracefully.")
    else:
        logger.info(f"Found {record_count} new records to process.")
        silver_df = transform_raw_to_silver(df_new_bronze)
        notify_anomalies(silver_df)
        
        logger.info("Writing transformed data to silver_gdelt_events table...")
        silver_table_name = "silver_gdelt_events"
        (silver_df.write
                  .mode("append")
                  .format("delta")
                  .saveAsTable(silver_table_name))
        
        final_count = silver_df.count()
        logger.info(f"Successfully appended {final_count} records to the '{silver_table_name}' table.")

except Exception as e:
    logger.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)
    raise
finally:
    if 'df_new_bronze' in locals():
        df_new_bronze.unpersist()
    logger.info("Pipeline run finished.")

# COMMAND ----------
