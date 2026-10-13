import streamlit as st
from typer import prompt
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {"configurable":{"thread_id":"thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Type Here")

#code for streaming
if user_input:
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
    
    with st.chat_message('assistant'):
        def response_generator():
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content

        ai_message = st.write_stream(response_generator())
    
    st.session_state['message_history'].append({'role':'assistant','content':ai_message})
