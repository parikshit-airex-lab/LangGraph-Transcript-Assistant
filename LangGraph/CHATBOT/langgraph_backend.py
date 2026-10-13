#langgraph import libraries
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile")

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages':[response]}

#checkpointer
checkpointer = InMemorySaver()
graph = StateGraph(ChatState)
graph.add_node('ChatNode',chat_node)
graph.add_edge(START, 'ChatNode')
graph.add_edge('ChatNode',END)
chatbot = graph.compile(checkpointer=checkpointer)