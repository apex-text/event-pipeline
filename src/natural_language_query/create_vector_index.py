# src/natural_language_query/create_vector_index.py
import os
import json
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

class PostgresVectorIndexer:
    def __init__(self):
        """
        Initializes the PostgreSQL connection and the Azure OpenAI Embedding model.
        """
        # Initialize PostgreSQL connection
        try:
            self.conn = psycopg2.connect(
                host=os.getenv("PG_HOST"),
                port=os.getenv("PG_PORT"),
                dbname=os.getenv("PG_DATABASE"),
                user=os.getenv("PG_USER"),
                password=os.getenv("PG_PASSWORD")
            )
            register_vector(self.conn)
            print("Successfully connected to PostgreSQL.")
        except psycopg2.OperationalError as e:
            print(f"Error connecting to PostgreSQL: {e}")
            self.conn = None
        
        # Initialize Azure OpenAI Embedding model
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_TEXTEMBEDDING_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        print("PostgresVectorIndexer initialized successfully.")

    def create_tables_and_indexes(self):
        """
        Reads and executes the SQL script to create tables and indexes.
        """
        if not self.conn:
            print("Cannot create tables, no database connection.")
            return
            
        try:
            with self.conn.cursor() as cur:
                print("Reading create_table.sql script...")
                with open("src/natural_language_query/create_table.sql", "r") as f:
                    sql_script = f.read()
                
                cur.execute(sql_script)
                self.conn.commit()
                print("Successfully created tables and indexes from create_table.sql.")
        except Exception as e:
            print(f"Error creating tables: {e}")
            self.conn.rollback()

    def load_data_from_json(self, file_path: str):
        """
        Loads data from a JSON file.
        """
        print(f"Loading data from {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} documents.")
            return data
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return []

    def generate_and_insert_embeddings(self, documents: list):
        """
        Generates embeddings for documents and inserts them into PostgreSQL.
        """
        if not self.conn or not documents:
            print("Cannot insert embeddings, no database connection or no documents.")
            return

        print(f"Starting to generate and insert embeddings for {len(documents)} documents...")
        
        records_to_insert = []
        for i, doc in enumerate(documents):
            try:
                # Create a meaningful text chunk from the document to embed.
                text_to_embed = (
                    f"Event Actor 1: {doc.get('actor1_name', 'N/A')}. "
                    f"Event Actor 2: {doc.get('actor2_name', 'N/A')}. "
                    f"Location: {doc.get('action_geo_fullname', 'N/A')}. "
                    f"Source: {doc.get('source_url', 'N/A')}"
                )

                # Generate embedding for the text chunk
                embedding_vector = self.embedding_model.embed_query(text_to_embed)
                
                records_to_insert.append((json.dumps(doc), embedding_vector))
                print(f"  ({i + 1}/{len(documents)}) Generated embedding for document id: {doc.get('global_event_id', 'N/A')}")

            except Exception as e:
                print(f"  ({i + 1}/{len(documents)}) Error processing document id {doc.get('globaleventid', 'N/A')}: {e}")

        # Use execute_values for efficient batch insertion
        if records_to_insert:
            print("Inserting records into the database...")
            try:
                with self.conn.cursor() as cur:
                    execute_values(
                        cur,
                        "INSERT INTO gdelt_embeddings (content, embedding) VALUES %s",
                        records_to_insert
                    )
                    self.conn.commit()
                    print(f"Successfully inserted {len(records_to_insert)} records.")
            except Exception as e:
                print(f"Error during batch insert: {e}")
                self.conn.rollback()

    def __del__(self):
        if self.conn:
            self.conn.close()
            print("PostgreSQL connection closed.")


if __name__ == "__main__":
    print("--- PostgreSQL Vector Indexing Script ---")
    print("This script will create tables, load data from a JSON file, generate vector embeddings, and save them to PostgreSQL.")
    print("WARNING: This may incur costs on your Azure OpenAI account.")
    
    run_script = input("Do you want to proceed? (yes/no): ").lower()
    
    if run_script == 'yes':
        try:
            indexer = PostgresVectorIndexer()
            
            # 1. Create database tables and indexes
            indexer.create_tables_and_indexes()
            
            # 2. Load data from the sample JSON file
            documents = indexer.load_data_from_json("src/natural_language_query/gdelt_sample_for_upload.json")
            
            # 3. Generate embeddings and insert into the database
            if documents:
                indexer.generate_and_insert_embeddings(documents)
            
            print("\nIndexing process completed.")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("\nPlease ensure your .env file is correctly configured.")
    else:
        print("Script cancelled by user.")
