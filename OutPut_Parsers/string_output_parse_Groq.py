from langchain_groq import ChatGroq
#from langchain.messages import HumanMessage
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


import os
load_dotenv()

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
template1=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)
template2=PromptTemplate(
    template="Write a 5 line summary on {text}",
    input_variables=["text"]
)

parser = StrOutputParser()
chain = template1 | model | template2 | model | parser
result = chain.invoke({"topic":"The impact of AI on society"})  
print(result)


