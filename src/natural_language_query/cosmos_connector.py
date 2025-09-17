# cosmos_connector.py
import os
from azure.cosmos import CosmosClient
from dotenv import load_dotenv

load_dotenv()

class CosmosDBConnector:
    """
    A connector class to interact with Azure Cosmos DB.
    Includes methods for standard SQL queries and vector similarity searches.
    """
    def __init__(self):
        """
        Initializes the Cosmos DB client.
        """
        endpoint = os.getenv("COSMOS_ENDPOINT")
        key = os.getenv("COSMOS_KEY")
        database_name = os.getenv("COSMOS_DATABASE_NAME")
        container_name = os.getenv("COSMOS_CONTAINER_NAME")

        if not all([endpoint, key, database_name, container_name]):
            raise ValueError("Cosmos DB environment variables are not set.")

        self.client = CosmosClient(endpoint, credential=key)
        self.database = self.client.get_database_client(database_name)
        self.container = self.database.get_container_client(container_name)
        print("Cosmos DB client initialized successfully.")

    def vector_search(self, query_vector: list[float], top_k: int = 5) -> list:
        """
        Performs a vector similarity search on the Cosmos DB container.

        Args:
            query_vector (list[float]): The embedding vector of the user's query.
            top_k (int): The number of similar documents to return.

        Returns:
            list: A list of the top_k most similar documents.
        """
        print(f"Performing vector search for top {top_k} results.")
        try:
            # The field `c.embedding` must match the vector field path in your Cosmos DB container's vector policy.
            # For example, if your vector is stored in a field named "embedding", the path is "/embedding".
            query = """
            SELECT TOP @top_k c.id, c.actor1_name, c.actor2_name, c.source_url, VectorDistance(c.embedding, @query_vector) AS similarityScore
            FROM c
            ORDER BY VectorDistance(c.embedding, @query_vector)
            """
            
            params = [
                {"name": "@top_k", "value": top_k},
                {"name": "@query_vector", "value": query_vector}
            ]

            results = self.container.query_items(
                query=query,
                parameters=params,
                enable_cross_partition_query=True
            )
            
            items = list(results)
            print(f"Vector search returned {len(items)} items.")
            return items
        except Exception as e:
            print(f"An error occurred during vector search: {e}")
            # It's common to get an error if the vector index is not configured.
            # Provide a helpful message.
            if "Vector search is not supported" in str(e):
                 raise RuntimeError(
                    "Vector search is not enabled or configured for this Cosmos DB container. "
                    "Please ensure you have created a container with a vector embedding policy. "
                    "You can run the `create_vector_index.py` script to populate the data with embeddings."
                ) from e
            return {"error": str(e)}

    def execute_query(self, query: str):
        """
        Executes a given SQL query against the Cosmos DB container.

        Args:
            query (str): The SQL query to execute.

        Returns:
            list: A list of items returned by the query.
        """
        try:
            print(f"Executing query: {query}")
            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            print(f"Query returned {len(items)} items.")
            return items
        except Exception as e:
            print(f"An error occurred while executing the query: {e}")
            return {"error": str(e)}

if __name__ == '__main__':
    # Example usage (for testing purposes)
    try:
        connector = CosmosDBConnector()
        
        # This is a dummy vector for testing. A real vector would have many more dimensions.
        # The dimension must match the one used in your embedding model (e.g., 1536 for text-embedding-ada-002).
        # NOTE: This will fail if your data is not indexed. Run `create_vector_index.py` first.
        dummy_vector = [0.0] * 1536 # Replace with the actual dimension of your embedding model
        
        print("\nPerforming a test vector search...")
        # results = connector.vector_search(dummy_vector, top_k=3)
        # print("\nVector Search Results:")
        # for item in results:
        #     print(item)

    except ValueError as ve:
        print(ve)
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
