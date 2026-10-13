from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

model = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

chat_history=[] #In future, LLM will get confused with messages, means won't be able to understand which is bot's msg and which one is from human, so here messages comes in picture.

while True:
    user_input=input("You: ")
    chat_history.append(user_input)
    if user_input==exit:
        break
    result = model.invoke(chat_history)
    chat_history.append(result)
    print("Bot:", result) 

      
print(chat_history)  