from groq import BaseModel
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()

model = ChatGroq()
model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

class Person(BaseModel):
    name : str = Field(description= 'Name of person')
    age : int = Field(gt=18, description='Age of Person')
    city : str = Field(description ='Name of city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me the name,age and city of a fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'place':'India'})
print(result)