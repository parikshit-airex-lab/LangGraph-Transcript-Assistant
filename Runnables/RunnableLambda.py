from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model = "meta-llama/llama-4-scout-17b-16e-instruct")

def word_count(text):
    return len(text.split())

prompt1 = PromptTemplate(
    template = "Write a report on this topic {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Write a short summary on this report {report}",
    input_variables=['report']
)
parser = StrOutputParser()
chain_report = RunnableSequence(prompt1,model,parser)
word_count_chain = RunnableLambda(word_count)

Branch_Chain = RunnableBranch(
    (lambda x:x['word_count']>=500, RunnableSequence(prompt2,model,parser)),
    (lambda x:x['word_count']<500, RunnablePassthrough()),
    RunnableLambda(lambda x: "Report is too short to summarise")
)

final_chain = RunnableSequence(chain_report, word_count_chain, Branch_Chain)
result = final_chain.invoke({'topic':'Artificial Intelligence'})    
print(result)
final_chain.get_graph().print_ascii()