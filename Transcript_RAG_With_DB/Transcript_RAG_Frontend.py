import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from Transcript_RAG_Loader import load_transcript
from Transcript_RAG_backend import (
    build_vectorstore,
    build_rag_graph,
    generate_summary,
    get_transcript_id,
    has_existing_index,
    load_existing_vectorstore,
    get_message_history,
    list_chat_sessions,
    save_transcript_meta,
    get_transcript_meta,
    list_indexed_transcripts,
)

st.set_page_config(page_title="Transcript RAG Chatbot")
st.title("📝 Meeting / Video Transcript Chatbot")

if "chatbot" not in st.session_state:
    st.session_state["chatbot"] = None
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "transcript_name" not in st.session_state:
    st.session_state["transcript_name"] = None
if "transcript_id" not in st.session_state:
    st.session_state["transcript_id"] = None
if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = None
if "summary_message" not in st.session_state:
    st.session_state["summary_message"] = None


def format_summary_message(summary) -> str:
    lines = ["### 📋 Transcript Summary", "", summary["summary"], "", "**Key Points:**"]
    lines += [f"- {point}" for point in summary["key_points"]] or ["- (none found)"]

    lines += ["", "**Action Items:**"]
    if summary["action_items"]:
        lines += [f"- [ ] {item}" for item in summary["action_items"]]
    else:
        lines += ["_No action items were discussed._"]

    return "\n".join(lines)


def new_chat_id() -> str:
    return uuid.uuid4().hex[:8]


def load_chat_history(transcript_id: str, chat_id: str) -> list[dict]:
    """Summary first, then any messages already checkpointed for this specific chat thread."""
    thread_id = f"{transcript_id}::{chat_id}"
    history = [{"role": "assistant", "content": st.session_state["summary_message"]}]
    prior_messages = get_message_history(st.session_state["chatbot"], thread_id)
    history += [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in prior_messages
    ]
    return history


def start_new_chat():
    st.session_state["chat_id"] = new_chat_id()
    st.session_state["message_history"] = [
        {"role": "assistant", "content": st.session_state["summary_message"]}
    ]


def switch_chat():
    chat_id = st.session_state["chat_selector"]
    st.session_state["chat_id"] = chat_id
    st.session_state["message_history"] = load_chat_history(
        st.session_state["transcript_id"], chat_id
    )


def activate_transcript(transcript_id: str, filename: str, vectorstore, summary: dict):
    """Wires up session_state for a transcript, whether just uploaded or resumed from disk."""
    st.session_state["chatbot"] = build_rag_graph(vectorstore)
    st.session_state["transcript_id"] = transcript_id
    st.session_state["transcript_name"] = filename
    st.session_state["summary_message"] = format_summary_message(summary)

    past_chats = list_chat_sessions(transcript_id)
    st.session_state["chat_id"] = past_chats[0] if past_chats else new_chat_id()
    st.session_state["message_history"] = load_chat_history(
        transcript_id, st.session_state["chat_id"]
    )


with st.sidebar:
    st.header("Upload transcript")
    uploaded_file = st.file_uploader(
        "Meeting/video transcript",
        type=["txt", "pdf", "srt", "vtt"],
    )

    if uploaded_file is not None and uploaded_file.name != st.session_state["transcript_name"]:
        transcript_id = get_transcript_id(uploaded_file.name)
        cached_meta = get_transcript_meta(transcript_id)

        with st.spinner("Reading transcript..."):
            chunks, full_text = load_transcript(uploaded_file)

        with st.spinner("Indexing transcript..."):
            if has_existing_index(transcript_id):
                vectorstore = load_existing_vectorstore(transcript_id)
                st.info("Found a previously indexed copy of this transcript — reusing it.")
            else:
                vectorstore = build_vectorstore(chunks, transcript_id)

        if cached_meta is not None:
            summary = cached_meta["summary"]
        else:
            with st.spinner("Summarizing..."):
                summary = generate_summary(full_text)
            save_transcript_meta(transcript_id, uploaded_file.name, summary)

        activate_transcript(transcript_id, uploaded_file.name, vectorstore, summary)
        st.success(f"Ready — {uploaded_file.name}")

    indexed_transcripts = list_indexed_transcripts()
    if indexed_transcripts:
        st.divider()
        st.caption("Or resume a previous transcript (no re-upload needed):")
        labels = {t["transcript_id"]: t["filename"] for t in indexed_transcripts}
        resume_options = ["— Select —"] + list(labels.keys())
        resume_choice = st.selectbox(
            "Previously indexed transcripts",
            resume_options,
            format_func=lambda tid: labels.get(tid, tid),
            key="resume_selector",
            label_visibility="collapsed",
        )
        if resume_choice != "— Select —" and resume_choice != st.session_state["transcript_id"]:
            meta = get_transcript_meta(resume_choice)
            vectorstore = load_existing_vectorstore(resume_choice)
            activate_transcript(resume_choice, meta["filename"], vectorstore, meta["summary"])
            st.success(f"Resumed — {meta['filename']}")

    if st.session_state["transcript_id"]:
        st.caption(f"Active transcript: {st.session_state['transcript_name']}")

        st.button("🆕 New Chat", on_click=start_new_chat, use_container_width=True)

        past_chats = list_chat_sessions(st.session_state["transcript_id"])
        if past_chats:
            default_index = (
                past_chats.index(st.session_state["chat_id"])
                if st.session_state["chat_id"] in past_chats
                else 0
            )
            st.selectbox(
                "Previous chats",
                past_chats,
                index=default_index,
                key="chat_selector",
                on_change=switch_chat,
            )

        if st.session_state["chat_id"] not in past_chats:
            st.caption("🆕 New conversation (not yet saved)")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about the transcript...")

if user_input:
    if st.session_state["chatbot"] is None:
        st.warning("Upload a transcript first.")
    else:
        thread_id = f"{st.session_state['transcript_id']}::{st.session_state['chat_id']}"
        config = {"configurable": {"thread_id": thread_id}}
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            def response_generator():
                for chunk, metadata in st.session_state["chatbot"].stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages",
                ):
                    if metadata.get("langgraph_node") == "generate" and hasattr(chunk, "content") and chunk.content:
                        yield chunk.content

            ai_message = st.write_stream(response_generator())

        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
