from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    task="text-generation",
)
hf_model = ChatHuggingFace(llm=llm)
template1=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)
template2=PromptTemplate(
    template="Write a 5 line summary on {text}",
    input_variables=["text"]
)

parser = StrOutputParser()
chain = template1 | hf_model | template2 | parser
result = chain.invoke({"topic":"The impact of AI on society"})  
print(result)
