# llm_service.py
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from prompts import ANSWER_GENERATION_PROMPT

load_dotenv()

class PostgresConnector:
    """
    Handles the connection to and vector search operations in PostgreSQL.
    """
    def __init__(self):
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

    def vector_search(self, query_vector: list[float], top_k: int = 5) -> list:
        """
        Performs a vector similarity search in the PostgreSQL database.
        """
        if not self.conn:
            return {"error": "Database connection is not available."}
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # The <=> operator calculates the cosine distance.
                # We must explicitly cast the list of floats to the `vector` type using `::vector`.
                cur.execute(
                    "SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity FROM gdelt_embeddings ORDER BY embedding <=> %s::vector LIMIT %s",
                    (query_vector, query_vector, top_k)
                )
                results = cur.fetchall()
                return results
        except Exception as e:
            print(f"An error occurred during vector search: {e}")
            # If an error occurs, rollback the transaction to keep the connection healthy.
            if self.conn:
                self.conn.rollback()
            return {"error": str(e)}

    def __del__(self):
        if self.conn:
            self.conn.close()
            print("PostgreSQL connection closed.")


class LLMQueryService:
    """
    Service to handle the RAG pipeline using PostgreSQL for vector search.
    """
    def __init__(self):
        """
        Initializes the LLMs, embedding model, and the Postgres connector.
        """
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_TEXTEMBEDDING_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        self.db_connector = PostgresConnector()
        print("LLM service initialized with PostgreSQL connector.")

    def embed_question(self, question: str) -> list[float]:
        """
        Generates an embedding vector for the user's question.
        """
        print("Embedding user question...")
        query_vector = self.embedding_model.embed_query(question)
        print("Question embedded successfully.")
        return query_vector

    def generate_final_answer(self, question: str, context: list) -> str:
        """
        Generates a final natural language answer based on the query results.
        """
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template=ANSWER_GENERATION_PROMPT
        )
        chain = LLMChain(llm=self.llm, prompt=prompt_template)

        print("Generating final answer from context...")
        context_str = json.dumps(context, indent=2, ensure_ascii=False, default=str)
        
        response = chain.invoke({"question": question, "context": context_str})
        answer = response.get('text', '').strip()
        print(f"Generated answer: {answer}")
        return answer

    def process_user_question(self, question: str) -> str:
        """
        Full RAG pipeline: question -> embedding -> vector search -> get context -> generate answer.
        """
        try:
            # 1. Embed the user's question
            query_vector = self.embed_question(question)
            if not query_vector:
                return "죄송합니다, 질문을 임베딩으로 변환할 수 없습니다."

            # 2. Perform vector search in PostgreSQL
            search_results = self.db_connector.vector_search(query_vector, top_k=5)
            if "error" in search_results or not search_results:
                return f"데이터를 검색하는 중 오류가 발생했거나 관련 데이터를 찾지 못했습니다. (오류: {search_results.get('error', '결과 없음')})"

            # 3. Generate final answer based on the retrieved context
            final_answer = self.generate_final_answer(question, search_results)
            return final_answer

        except Exception as e:
            print(f"An error occurred in the process: {e}")
            return f"죄송합니다, 요청을 처리하는 중에 예상치 못한 오류가 발생했습니다: {e}"
