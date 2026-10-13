from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8)

result = model.invoke("What is the difference between Rajyasabha and Lok Sabha in India?")
print(result.content)

