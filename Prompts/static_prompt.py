from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model=GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
st.title("Static prompt Demonstration")
input = st.text_input=st.text_area("Enter your text prompt here")

if st.button("Summary"):
    result = model.invoke(input)
    st.write(result)

