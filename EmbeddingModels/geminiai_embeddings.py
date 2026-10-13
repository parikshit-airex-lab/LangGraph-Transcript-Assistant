from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

#embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", dimensions=32)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
result= embedding.embed_query("Details of Fixed Deposit in India")
print(str(result))