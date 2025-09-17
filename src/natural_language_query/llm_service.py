# llm_service.py
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json

from cosmos_connector import CosmosDBConnector
from prompts import ANSWER_GENERATION_PROMPT

load_dotenv()

class LLMQueryService:
    """
    Service to handle the logic of converting natural language to a vector,
    performing a similarity search in Cosmos DB, and generating a natural language response.
    """
    def __init__(self):
        """
        Initializes the LLMs, embedding model, and the Cosmos DB connector.
        """
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_TEXTEMBEDDING_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        self.cosmos_connector = CosmosDBConnector()
        print("LLM service initialized with embedding model.")

    def embed_question(self, question: str) -> list[float]:
        """
        Generates an embedding vector for the user's question.

        Args:
            question (str): The user's natural language question.

        Returns:
            list[float]: The embedding vector for the question.
        """
        print("Embedding user question...")
        query_vector = self.embedding_model.embed_query(question)
        print("Question embedded successfully.")
        return query_vector

    def generate_final_answer(self, question: str, context: list) -> str:
        """
        Generates a final natural language answer based on the query results.

        Args:
            question (str): The original user question.
            context (list): The data retrieved from Cosmos DB.

        Returns:
            str: The final natural language answer.
        """
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template=ANSWER_GENERATION_PROMPT
        )
        chain = LLMChain(llm=self.llm, prompt=prompt_template)

        print("Generating final answer from context...")
        # Convert context to a JSON string for the prompt
        context_str = json.dumps(context, indent=2, ensure_ascii=False)
        
        response = chain.invoke({"question": question, "context": context_str})
        answer = response.get('text', '').strip()
        print(f"Generated answer: {answer}")
        return answer

    def process_user_question(self, question: str) -> str:
        """
        Full RAG pipeline: question -> embedding -> vector search -> get context -> generate answer.

        Args:
            question (str): The user's natural language question.

        Returns:
            str: The final natural language answer.
        """
        try:
            # 1. Embed the user's question
            query_vector = self.embed_question(question)
            if not query_vector:
                return "죄송합니다, 질문을 임베딩으로 변환할 수 없습니다."

            # 2. Perform vector search in Cosmos DB
            search_results = self.cosmos_connector.vector_search(query_vector, top_k=5)
            if "error" in search_results or not search_results:
                return f"데이터를 검색하는 중 오류가 발생했거나 관련 데이터를 찾지 못했습니다. (오류: {search_results.get('error', '결과 없음')})"

            # 3. Generate final answer based on the retrieved context
            final_answer = self.generate_final_answer(question, search_results)
            return final_answer

        except Exception as e:
            print(f"An error occurred in the process: {e}")
            return f"죄송합니다, 요청을 처리하는 중에 예상치 못한 오류가 발생했습니다: {e}"
