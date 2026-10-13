from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model = "meta-llama/llama-4-scout-17b-16e-instruct")
prompt1 = PromptTemplate(
    template = "Write a joke on {topic}", 
    input_variables = ['topic']
    )
parser = StrOutputParser()
seqRunnable = RunnableSequence(prompt1, model, parser)
result = seqRunnable.invoke({'topic': 'programming'})
print(result)
