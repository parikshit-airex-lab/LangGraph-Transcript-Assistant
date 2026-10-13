from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

model = ChatGroq(model ="meta-llama/llama-4-scout-17b-16e-instruct")
prompt1 = PromptTemplate(
  template = "Give 3 different jokes about {topic}",
  input_variables=['topic']
 )

parser = StrOutputParser()

simpleSeqChain = prompt1 | model | parser
result = simpleSeqChain.invoke({'topic':'Food'})
print(result)
simpleSeqChain.get_graph().print_ascii()