# src/natural_language_query/create_vector_index.py
import os
import json
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, PartitionKey
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

"""
IMPORTANT SETUP INSTRUCTIONS FOR COSMOS DB:

This script creates vector embeddings for your data and adds them to your Cosmos DB container.
Before running this script, you MUST configure a vector embedding policy on your container.

How to configure the vector policy in Azure Portal:
1. Go to your Azure Cosmos DB account.
2. Navigate to 'Data Explorer'.
3. Select your database and container.
4. Go to 'Settings' -> 'Vector Indexes'.
5. Click 'Add Index'.
6. For the 'Vector Embedding Policy', provide the following JSON.
   This policy specifies that the vector field is "/embedding" and uses a cosine distance metric.
   The dimension `1536` is for the `text-embedding-ada-002` model. If you use a different
   model, you might need to change this value.

[
    {
        "path": "/embedding",
        "dataType": "float32",
        "dimensions": 1536,
        "distanceFunction": "cosine"
    }
]

7. Save the policy. Now you are ready to run this script.
"""

class VectorIndexer:
    def __init__(self):
        """
        Initializes the Cosmos DB client and the Azure OpenAI Embedding model.
        """
        # Initialize Cosmos DB client
        endpoint = os.getenv("COSMOS_ENDPOINT")
        key = os.getenv("COSMOS_KEY")
        database_name = os.getenv("COSMOS_DATABASE_NAME")
        container_name = os.getenv("COSMOS_CONTAINER_NAME")

        if not all([endpoint, key, database_name, container_name]):
            raise ValueError("Cosmos DB environment variables are not set.")

        self.client = CosmosClient(endpoint, credential=key)
        self.database = self.client.get_database_client(database_name)
        self.container = self.database.get_container_client(container_name)
        
        # Initialize Azure OpenAI Embedding model
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_TEXTEMBEDDING_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        print("VectorIndexer initialized successfully.")

    def get_all_documents(self):
        """
        Retrieves all documents from the container that do not yet have an embedding.
        """
        print("Fetching documents without embeddings from Cosmos DB...")
        # Query for all items. In a larger dataset, you might want to paginate this.
        # We add a check to only fetch documents where the 'embedding' field is not set.
        query = "SELECT * FROM c WHERE NOT IS_DEFINED(c.embedding)"
        items = list(self.container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        print(f"Found {len(items)} documents to process.")
        return items

    def create_and_upsert_embeddings(self):
        """
        Generates embeddings for documents and upserts them back into Cosmos DB.
        """
        documents = self.get_all_documents()
        if not documents:
            print("All documents already have embeddings. No new documents to process.")
            return

        print(f"Starting to generate and upsert embeddings for {len(documents)} documents...")
        
        for i, doc in enumerate(documents):
            try:
                # Create a meaningful text chunk from the document to embed.
                # Here, we are combining a few key fields. You can customize this.
                text_to_embed = (
                    f"Event Actor 1: {doc.get('actor1_name', 'N/A')}. "
                    f"Event Actor 2: {doc.get('actor2_name', 'N/A')}. "
                    f"Location: {doc.get('action_geo_fullname', 'N/A')}. "
                    f"Source: {doc.get('source_url', 'N/A')}"
                )

                # Generate embedding for the text chunk
                embedding_vector = self.embedding_model.embed_query(text_to_embed)

                # Add the new 'embedding' field to the document
                doc['embedding'] = embedding_vector

                # Upsert the document back into Cosmos DB
                self.container.upsert_item(body=doc)
                
                print(f"  ({i + 1}/{len(documents)}) Successfully processed and upserted document with id: {doc['id']}")

            except Exception as e:
                print(f"  ({i + 1}/{len(documents)}) Error processing document id {doc.get('id', 'N/A')}: {e}")

        print("\nEmbedding generation and upsert process completed.")


if __name__ == "__main__":
    print("--- Cosmos DB Vector Indexing Script ---")
    print("This script will fetch documents from your Cosmos DB, generate vector embeddings, and save them back.")
    print("WARNING: This may incur costs on your Azure OpenAI and Cosmos DB accounts.")
    
    run_script = input("Do you want to proceed? (yes/no): ").lower()
    
    if run_script == 'yes':
        try:
            indexer = VectorIndexer()
            indexer.create_and_upsert_embeddings()
        except Exception as e:
            print(f"An error occurred: {e}")
            print("\nPlease ensure your .env file is correctly configured and you have set up the vector policy in your Cosmos DB container (see instructions in the script).")
    else:
        print("Script cancelled by user.")
