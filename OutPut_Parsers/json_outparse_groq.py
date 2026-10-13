from langchain_groq import ChatGroq
#from langchain.messages import HumanMessage
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


import os
load_dotenv()
parser = JsonOutputParser()
model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
template=PromptTemplate(
    template="Give me the name,age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({})#there is no input variable in template hence empty
print(result)