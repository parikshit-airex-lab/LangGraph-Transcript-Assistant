from langchain_core.prompts import ChatPromptTemplate   
# from langchain_google_genai import GoogleGenerativeAI
# from langchain_core.messages import HumanMessage, SystemMessage
# from dotenv import load_dotenv
# load_dotenv()   

# model = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

chat_template=ChatPromptTemplate.ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert "),
    ('human', "Tell me about {topic}.")
])

prompt = chat_template.invoke({'domain': 'Sports','topic': 'Cricket'})

print(prompt)

