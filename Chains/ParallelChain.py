from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel

load_dotenv()
model  = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template = "Write short notes on {topic}",
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template = "Give 5 question answer quiz on following topic \n {topic}",
    input_variables = ['topic']
)
prompt3 = PromptTemplate(
    template = "Merge notes and document into single document \n {notes} and {quiz}",
    input_variables = ['notes','quiz']
)
parallelChain = RunnableParallel({
 'notes': prompt1 | model |parser,
 'quiz': prompt2 | model | parser
})

mergeChain = prompt3 | model | parser
finalChain = parallelChain | mergeChain
result = finalChain.invoke({'topic':"Basics of AI"})
print(result)
finalChain.get_graph().print_ascii()