from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    task="text-generation",
)

hf_model = ChatHuggingFace(llm=llm)
result = hf_model.invoke("Give me details of fixed deposit?")
print(result.content)

#You did not load the model. You created wrappers and downloaded tokenizer/config. (LangChain Docs)
#invoke() calls Hugging Face chat completion routing, which needs at least one deployed provider. (Hugging Face)
#TinyLlama explicitly has no Inference Provider deployment, so routing returns an empty provider list and you get StopIteration. (Hugging Face)