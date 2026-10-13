import hashlib
import json
import os
import sqlite3
import time
from typing import TypedDict, Annotated
from groq import RateLimitError
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

CHROMA_DIR = os.path.join(STORAGE_DIR, "chroma_store")
CHAT_DB_PATH = os.path.join(STORAGE_DIR, "chat_history.db")

conn = sqlite3.connect(CHAT_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Tracks every transcript ever indexed (filename + its cached summary) so a
# transcript can be RESUMED later using only its transcript_id - no need to
# re-upload the file, which matters after a browser refresh wipes st.session_state.
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS transcript_meta (
        transcript_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        summary_json TEXT NOT NULL,
        indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """
)
conn.commit()

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

summary_llm = llm.with_structured_output(TranscriptSummary, method="json_schema")

# Groq's free/on-demand tier caps openai/gpt-oss-120b at a low tokens-per-minute
# budget. A full transcript in one prompt can easily exceed it, so long transcripts
# are map-reduced: summarized in small chunks first, then those short partial
# summaries (not the raw transcript) are fed into the final structured summary.
MAP_CHUNK_CHARS = 3000

MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are summarizing one excerpt of a longer meeting/video transcript. In 2-4 "
     "sentences, note the key points discussed and any action items or to-dos mentioned "
     "(with owner if stated) in THIS excerpt only.\n\nExcerpt:\n{excerpt}"),
])


def _invoke_with_backoff(runnable, prompt, max_attempts: int = 4, base_delay: float = 3.0):
    """
    Retries a Groq call with growing backoff if the account's tokens-per-minute
    budget is hit. Several small calls fired back-to-back (e.g. map-reduce chunks)
    can still add up past the per-minute cap even though each one is small on its own.
    """
    for attempt in range(max_attempts):
        try:
            return runnable.invoke(prompt)
        except RateLimitError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (attempt + 1))


def _summarize_chunk(excerpt: str) -> str:
    prompt = MAP_PROMPT.invoke({"excerpt": excerpt})
    return _invoke_with_backoff(llm, prompt).content


def _condense(full_text: str) -> str:
    """Reduces a long transcript to a short list of partial summaries, one per chunk."""
    if len(full_text) <= MAP_CHUNK_CHARS:
        return full_text

    chunks = [full_text[i:i + MAP_CHUNK_CHARS] for i in range(0, len(full_text), MAP_CHUNK_CHARS)]
    partial_summaries = []
    for chunk in chunks:
        partial_summaries.append(_summarize_chunk(chunk))
        time.sleep(1.5)  # pace calls so cumulative usage stays under the 8000 TPM cap
    return "\n\n".join(partial_summaries)


def generate_summary(full_text: str) -> TranscriptSummary:
    """Structured summary of the whole transcript, run once right after upload."""
    condensed_text = _condense(full_text)
    prompt = SUMMARY_PROMPT.invoke({"transcript": condensed_text})
    return _invoke_with_backoff(summary_llm, prompt)


def get_transcript_id(filename: str) -> str:
    """
    Stable identifier for a transcript, derived from its filename.
    Reused as BOTH the Chroma collection name and the LangGraph thread_id,
    so a transcript's embeddings and its conversation history are tied together.
    """
    return "t_" + hashlib.sha256(filename.encode()).hexdigest()[:16]


def save_transcript_meta(transcript_id: str, filename: str, summary: TranscriptSummary) -> None:
    """Remembers a transcript's filename + summary so it can be resumed later without the file."""
    conn.execute(
        "INSERT INTO transcript_meta (transcript_id, filename, summary_json) VALUES (?, ?, ?) "
        "ON CONFLICT(transcript_id) DO UPDATE SET filename=excluded.filename, summary_json=excluded.summary_json",
        (transcript_id, filename, json.dumps(summary)),
    )
    conn.commit()


def get_transcript_meta(transcript_id: str) -> dict | None:
    """Returns {'filename', 'summary'} for a previously indexed transcript, or None."""
    row = conn.execute(
        "SELECT filename, summary_json FROM transcript_meta WHERE transcript_id = ?",
        (transcript_id,),
    ).fetchone()
    if row is None:
        return None
    filename, summary_json = row
    return {"filename": filename, "summary": json.loads(summary_json)}


def list_indexed_transcripts() -> list[dict]:
    """Every transcript ever indexed, most recently indexed first - powers the 'resume' picker."""
    rows = conn.execute(
        "SELECT transcript_id, filename FROM transcript_meta ORDER BY indexed_at DESC"
    ).fetchall()
    return [{"transcript_id": tid, "filename": filename} for tid, filename in rows]


def has_existing_index(transcript_id: str) -> bool:
    """True if this transcript was already embedded and persisted in a previous run."""
    vectorstore = Chroma(
        collection_name=transcript_id,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    return len(vectorstore.get(limit=1)["ids"]) > 0


def build_vectorstore(docs: list[Document], transcript_id: str) -> Chroma:
    """Embeds transcript chunks into a Chroma collection persisted to disk under CHROMA_DIR."""
    return Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name=transcript_id,
        persist_directory=CHROMA_DIR,
    )


def load_existing_vectorstore(transcript_id: str) -> Chroma:
    """Reopens a previously-persisted collection without re-embedding anything."""
    return Chroma(
        collection_name=transcript_id,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )


def get_message_history(chatbot, thread_id: str) -> list[BaseMessage]:
    """Reads back any messages already checkpointed for this thread (e.g. from a previous session)."""
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    return state.values.get("messages", []) if state.values else []


def list_chat_sessions(transcript_id: str) -> list[str]:
    """
    Returns the chat_ids of every conversation previously held about this transcript,
    most recently active first. Threads are named f"{transcript_id}::{chat_id}", so this
    scans all checkpointed threads and keeps only the ones belonging to this transcript.
    """
    prefix = f"{transcript_id}::"
    latest_ts: dict[str, str] = {}
    for checkpoint_tuple in checkpointer.list(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        if not thread_id.startswith(prefix):
            continue
        chat_id = thread_id[len(prefix):]
        ts = checkpoint_tuple.checkpoint["ts"]
        if chat_id not in latest_ts or ts > latest_ts[chat_id]:
            latest_ts[chat_id] = ts
    return [chat_id for chat_id, _ in sorted(latest_ts.items(), key=lambda kv: kv[1], reverse=True)]


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
        response = _invoke_with_backoff(llm, prompt)
        return {"messages": [response]}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)
