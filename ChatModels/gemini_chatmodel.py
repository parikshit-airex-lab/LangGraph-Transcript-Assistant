from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result = gemini.invoke("What is the difference between Rajyasabha and Lok Sabha in India?")
print(result.content)