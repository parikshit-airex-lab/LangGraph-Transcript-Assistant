from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.8)

result = llm.invoke("What is the difference between Rajyasabha and Lok Sabha in India?")
print(result)
