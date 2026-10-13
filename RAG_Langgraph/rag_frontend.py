import streamlit as st
from langchain_core.messages import HumanMessage
from transcript_loader import load_transcript
from rag_backend import build_vectorstore, build_rag_graph, generate_summary

st.set_page_config(page_title="Transcript RAG Chatbot")
st.title("📝 Meeting / Video Transcript Chatbot")

CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "chatbot" not in st.session_state:
    st.session_state["chatbot"] = None
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "transcript_name" not in st.session_state:
    st.session_state["transcript_name"] = None


def format_summary_message(summary) -> str:
    lines = ["### 📋 Transcript Summary", "", summary["summary"], "", "**Key Points:**"]
    lines += [f"- {point}" for point in summary["key_points"]] or ["- (none found)"]

    lines += ["", "**Action Items:**"]
    if summary["action_items"]:
        lines += [f"- [ ] {item}" for item in summary["action_items"]]
    else:
        lines += ["_No action items were discussed._"]

    return "\n".join(lines)


with st.sidebar:
    st.header("Upload transcript")
    uploaded_file = st.file_uploader(
        "Meeting/video transcript",
        type=["txt", "pdf", "srt", "vtt"],
    )

    if uploaded_file is not None and uploaded_file.name != st.session_state["transcript_name"]:
        with st.spinner("Reading and indexing transcript..."):
            chunks, full_text = load_transcript(uploaded_file)
            vectorstore = build_vectorstore(chunks)
            st.session_state["chatbot"] = build_rag_graph(vectorstore)
            st.session_state["transcript_name"] = uploaded_file.name

        with st.spinner("Summarizing..."):
            summary = generate_summary(full_text)
            st.session_state["message_history"] = [
                {"role": "assistant", "content": format_summary_message(summary)}
            ]

        st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")

    if st.session_state["transcript_name"]:
        st.caption(f"Active transcript: {st.session_state['transcript_name']}")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about the transcript...")

if user_input:
    if st.session_state["chatbot"] is None:
        st.warning("Upload a transcript first.")
    else:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            def response_generator():
                for chunk, metadata in st.session_state["chatbot"].stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    if metadata.get("langgraph_node") == "generate" and hasattr(chunk, "content") and chunk.content:
                        yield chunk.content

            ai_message = st.write_stream(response_generator())

        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
