from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

model = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

while True:
    user_input=input("You: ")
    if user_input==exit:
        break
    else:
        result = model.invoke(user_input)
        print("Bot:", result)   