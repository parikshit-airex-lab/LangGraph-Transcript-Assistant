from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model = "meta-llama/llama-4-scout-17b-16e-instruct")

prompt1 = PromptTemplate(
    template = "Write a joke on this {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Write an explaination of this joke {joke}",
    input_variables=["joke"]
)

parser = StrOutputParser()
gen_joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'Explaination':RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(gen_joke_chain,parallel_chain)
result = final_chain.invoke({'topic':'School'})
print(result)
final_chain.get_graph().print_ascii()