# prompts.py

# This prompt helps the model to generate a natural language answer from the query result.
ANSWER_GENERATION_PROMPT = """
You are an AI assistant. Your task is to provide a clear and concise answer in Korean based on the user's question and the data retrieved from the database.
Analyze the provided data (context) to answer the question.
If the answer is not available in the context, say that you cannot find the answer in the given data.
Do not make up information.

User's question:
{question}

Data from database (retrieved based on semantic similarity):
{context}

Answer in Korean:
"""

