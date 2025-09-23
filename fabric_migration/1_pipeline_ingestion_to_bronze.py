# COMMAND ----------
# MAGIC %md
## 1. GDELT Ingestion to Bronze Table


# COMMAND ----------
# %pip install requests

# COMMAND ----------
import requests
import zipfile
import io
import logging
from pyspark.sql.functions import col, lit, current_timestamp
from pyspark.sql.types import StructType, StructField, ArrayType, StringType
import pyspark.sql.functions as F

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COMMAND ----------
# MAGIC %md
### Step 1: Get Latest GDELT Data URL

# COMMAND ----------
def get_latest_gdelt_data_url():
    """Fetches the URL of the latest 15-minute GDELT export data."""
    try:
        latest_gdelt_url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
        response = requests.get(latest_gdelt_url, timeout=30)
        response.raise_for_status()
        for line in response.text.split("\n"):
            if "export.CSV.zip" in line:
                url = line.split(" ")[2]
                logger.info(f"Found latest GDELT data URL: {url}")
                return url
        logger.warning("export.CSV.zip not found in lastupdate.txt")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch latest GDELT URL: {e}")
        return None

latest_url = get_latest_gdelt_data_url()

# COMMAND ----------
# MAGIC %md
### Step 2: Download, Process, and Save to Bronze Table (Corrected Schema)

# COMMAND ----------
if latest_url:
    try:
        logger.info(f"Downloading data from: {latest_url}")
        response = requests.get(latest_url, stream=True, timeout=120)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]
            logger.info(f"Processing file: {csv_filename}")
            
            csv_content = z.open(csv_filename).read().decode('utf-8')
            
            lines_rdd = spark.sparkContext.parallelize(csv_content.strip().split('\n'))
            
            # Split each line by tab, but wrap the result in a Row object for schema application
            data_rdd = lines_rdd.map(lambda line: (line.split('\t'),))
            
            # ** FIX: Define the schema explicitly **
            # This tells Spark that 'raw_data' is an array of strings.
            schema = StructType([
                StructField("raw_data", ArrayType(StringType()), True)
            ])
            
            # Create the DataFrame with the correct schema
            df_bronze_raw = spark.createDataFrame(data_rdd, schema)

            # Add metadata columns
            df_bronze = df_bronze_raw.withColumn("source_url", lit(latest_url)) \
                                     .withColumn("source_file", lit(csv_filename)) \
                                     .withColumn("ingestion_time", current_timestamp())

            # Filter out any empty or malformed lines
            df_bronze = df_bronze.filter(F.size(F.col("raw_data")) > 1)

            bronze_table_name = "bronze_gdelt_events"
            record_count = df_bronze.count()
            logger.info(f"Writing {record_count} records to {bronze_table_name}...")
            
            (df_bronze.write
                      .mode("append")
                      .format("delta")
                      .option("mergeSchema", "true")
                      .saveAsTable(bronze_table_name))
            
            logger.info(f"Successfully appended {record_count} records to the '{bronze_table_name}' table.")

    except Exception as e:
        logger.error(f"An error occurred during data ingestion: {e}", exc_info=True)
        raise
else:
    logger.warning("No new URL found. Exiting gracefully.")

# COMMAND ----------
