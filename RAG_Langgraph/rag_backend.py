from typing import TypedDict, Annotated
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant answering questions about a meeting/video transcript. "
     "Use ONLY the transcript excerpts below to answer. If the answer isn't in the "
     "excerpts, say you don't know based on the transcript.\n\n"
     "Transcript excerpts:\n{context}"),
    ("placeholder", "{messages}"),
])


class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: list[Document]

class TranscriptSummary(TypedDict):
    summary: str
    key_points: list[str]
    action_items: list[str]


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant that analyzes meeting/video transcripts. Given the full "
     "transcript below, produce a concise 3-5 sentence summary, a list of key discussion "
     "points, and a list of any action items or to-dos that were mentioned (include the "
     "owner's name if one was stated). If no action items were discussed, return an "
     "empty list for action_items.\n\nTranscript:\n{transcript}"),
])

summary_llm = llm.with_structured_output(TranscriptSummary)


def generate_summary(full_text: str) -> TranscriptSummary:
    """One-shot structured summary of the whole transcript, run once right after upload."""
    prompt = SUMMARY_PROMPT.invoke({"transcript": full_text})
    return summary_llm.invoke(prompt)


def build_vectorstore(docs: list[Document]) -> Chroma:
    """Embeds transcript chunks into a fresh, in-memory Chroma collection."""
    return Chroma.from_documents(documents=docs, embedding=embedding_model)


def build_rag_graph(vectorstore: Chroma):
    """Compiles a retrieve -> generate LangGraph chatbot bound to this transcript's vectorstore."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def retrieve_node(state: RAGState):
        question = state["messages"][-1].content
        retrieved_docs = retriever.invoke(question)
        return {"context": retrieved_docs}

    def generate_node(state: RAGState):
        context_text = "\n\n".join(
            f"[{doc.metadata.get('start_time') or doc.metadata.get('source', 'transcript')}] {doc.page_content}"
            for doc in state["context"]
        )
        prompt = RAG_PROMPT.invoke({"context": context_text, "messages": state["messages"]})
        response = llm.invoke(prompt)
        return {"messages": [response]}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=InMemorySaver())
