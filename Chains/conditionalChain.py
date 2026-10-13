from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGroq(model = "meta-llama/llama-4-scout-17b-16e-instruct")
parser = StrOutputParser()
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"]= Field(description = "Give the sentiment of feedback")

parser2 = PydanticOutputParser(pydantic_object = Feedback)

template1 = PromptTemplate(
    template = "Classify the feedback into positive or negative \n {feedback} \n {format_instruction}",
    input_variables = ['feedback'],
    partial_variables = ({'format_instruction': parser2.get_format_instructions()})
)


prompt2 = PromptTemplate(
    template = "Based on positive sentiment feedback, write thank you mesage \n {feedback}",
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = "Based on negative sentiment feedback, write an apology message \n {feedback}",  
    input_variables = ['feedback']
)

sentiment_chain = template1 | model | parser2
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Neutraal Feedback Received")
)

final_chain = sentiment_chain | branch_chain
result = final_chain.invoke({'feedback': "The product quality is very bad and I am not happy with the purchase"})
print(result)
final_chain.get_graph().print_ascii()