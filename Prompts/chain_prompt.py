from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st

load_dotenv()

model=GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
st.header("Dynamic prompt Demonstration")
paper_input=st.selectbox("select research paper name", ["Attention is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language models are few shot learners", "Diffusion Models Beat GANS on Image Synthesis"])

style_input=st.selectbox("Select Explaination style", ["Beginner Friendly", "Mathematical", "Code-oriented"])

length_input=st.selectbox("Select length of response",["short(1-2 paragraph)","medium(3-4 paragraph)","long(detailed explaination)"])

template= PromptTemplate(template="""
Please summarize the research paper titled "{paper_input}" with the following 
specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details: - Include relevant mathematical equations if present in the paper. - Explain the mathematical concepts using simple, intuitive code snippets 
where applicable. 
2. Analogies: - Use relatable analogies to simplify complex ideas. 
If certain information is not available in the paper, respond with: "Insufficient 
information available" instead of guessing. 
Ensure the summary is clear, accurate, and aligned with the provided style and 
length.
""",
input_variables=["paper_input","style_input","length_input"],
validate_template=True #to ensure all input variables are used in the template
)

# prompt = template.invoke({
#     "paper_input": paper_input,
#     "style_input": style_input,
#     "length_input": length_input
# })

if st.button("summary"):
    chain = template | model
    result=chain.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})
    st.write(result)