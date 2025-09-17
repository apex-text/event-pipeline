# app.py
import streamlit as st
import os
from dotenv import load_dotenv

# It's important to load environment variables before importing other modules
# that depend on them.
load_dotenv()

# Check for environment variables before initializing the service
required_env_vars = [
    "PG_HOST", "PG_PORT", "PG_DATABASE", "PG_USER", "PG_PASSWORD",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_TEXTEMBEDDING_DEPLOYMENT_NAME"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    st.error(f"The following environment variables are missing in your .env file: {', '.join(missing_vars)}")
    st.info("Please create a .env file in the 'src/natural_language_query' directory and fill in the required values.")
else:
    # Import the service only if the environment variables are present
    from llm_service import LLMQueryService

    def main():
        st.title("PostgreSQL 기반 자연어 쿼리 (벡터 검색 RAG)")
        st.write("PostgreSQL 데이터베이스에 자연어로 질문하세요. (의미 기반 검색)")

        # Initialize the service and cache it
        @st.cache_resource
        def get_llm_service():
            return LLMQueryService()

        llm_service = get_llm_service()

        # User input
        user_question = st.text_input("질문을 입력하세요:", "")

        if st.button("질문하기"):
            if user_question:
                with st.spinner("답변을 생성하는 중입니다..."):
                    try:
                        answer = llm_service.process_user_question(user_question)
                        st.success("답변:")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"오류가 발생했습니다: {e}")
            else:
                st.warning("질문을 입력해주세요.")

    if __name__ == "__main__":
        main()
