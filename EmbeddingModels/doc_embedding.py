from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", dimensions=32)

documents = [
    "Who is prime minister of India",
    "Capital of France is Paris",
    "What is fixed deposit"
]
result = embedding.embed_documents(documents)
print(str(result))