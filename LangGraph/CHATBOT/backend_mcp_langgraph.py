from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import display, Image
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

load_dotenv()
#-------1. connect to the MCP server and discover its tools
client = MultiServerMCPClient({
    "calculator":{
        "command":"Python",
        "args":["E:/Lang_Chain/LangGraph/CHATBOT/mcp_server.py"],
        "transport":"stdio"
    }
})
## client.get_tools() asks the server "what tools do you have?" and converts each into a LangChain-compatible tool. It's a coroutine, so it must be awaited. Since your module loads at import time (not inside an async function), wrap it with asyncio.run():
tools = asyncio.run(client.get_tools())

#--------2. LLM + Tool Binding -----
llm = ChatGroq(model = "llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(tools)

#--------3. State Schema --------
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

#------4. Add Nodes ----------
def chat_node(state:ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages':[response]}

tool_node = ToolNode(tools)

#-------5. Graph Writing -------
checkpointer = InMemorySaver()
graph = StateGraph(ChatState)
#nodes
graph.add_node("ChatNode", chat_node)
graph.add_node("tools", tool_node)
#edges
graph.add_edge(START,"ChatNode")
graph.add_conditional_edges("ChatNode",tools_condition)
graph.add_edge('tools','ChatNode')

chatbot = graph.compile(checkpointer=checkpointer)

#-------6. Standalone Test -----
if __name__ == "__main__":
    async def main():
        result = await chatbot.ainvoke(
            {"messages": [("user", "what is 7 plus 8?")]},
            config={"configurable": {"thread_id": "test-1"}},
        )
        print(result["messages"][-1].content)

    asyncio.run(main())