from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt
import streamlit as st

load_dotenv()

model=GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
st.header("Dynamic prompt Demonstration")
paper_input=st.selectbox("select research paper name", ["Attention is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language models are few shot learners", "Diffusion Models Beat GANS on Image Synthesis"])

style_input=st.selectbox("Select Explaination style", ["Beginner Friendly", "Mathematical", "Code-oriented"])

length_input=st.selectbox("Select length of response",["short(1-2 paragraph)","medium(3-4 paragraph)","long(detailed explaination)"])

template = load_prompt('template.json')

prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button("summary"):
    result=model.invoke(prompt)
    st.write(result)