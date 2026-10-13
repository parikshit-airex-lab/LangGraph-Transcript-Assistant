import streamlit as st
from sympy import im
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

#Use session state of streamlit -> session state is dict in which key and value and Value is list of dict

CONFIG = {"configurable":{"thread_id":"thread-1"}}

if 'message_history' not in st.session_state: # message_history is key in dict session state
    st.session_state["message_history"]=[] #if not then append

#loading conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message['role']):
        st.text(message['content'])

        #{'role':"User","content":"Hii"}
        #{"role":"Assistant","content":"Hello"}

user_input = st.chat_input("Type Here")

if user_input:
    #add message to message history
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    #lets take response from llm
    response = chatbot.invoke({'messages':[HumanMessage(content=user_input)]},config = CONFIG)
    ai_message = response['messages'][-1].content # to fetch last message
    #first add ai message to message history
    st.session_state['message_history'].append({'role':'assistant','content':ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)



