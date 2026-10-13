from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

model = ChatGroq(model ="meta-llama/llama-4-scout-17b-16e-instruct")

prompt1 = PromptTemplate(
    template= "Generate report on the given topic \n {topic}",
    input_variables = ['topic']
)
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = "Give 5 points summary for the given report {report}",
    input_variables = ['report']
)

SeqChain = prompt1 | model | parser | prompt2 | model | parser
Result = SeqChain.invoke({'topic':'Future of AI'})
print(Result)
SeqChain.get_graph().print_ascii()