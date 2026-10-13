from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model = "meta-llama/llama-4-scout-17b-16e-instruct")

prompt = PromptTemplate(
    template ="Write answers of the quetions provided in this text {text}",
    input_variables=['text']
)   
parser = StrOutputParser()

loader = PyPDFLoader("documents\DSA_Question.pdf")
docs =  loader.load()
# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt | model | parser
result = chain.invoke({'text':docs[0].page_content})
print(result)   