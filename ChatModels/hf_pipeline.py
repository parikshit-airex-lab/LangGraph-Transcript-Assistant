from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv


load_dotenv()
import os
os.environ["HF_HOME"] = "./hf_cache"  # Optional: Set custom cache directory for Hugging Face models
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

hf_model = ChatHuggingFace(llm=llm)
result = hf_model.invoke("Give me details of fixed deposit?")
print(result.content)

#You did not load the model. You created wrappers and downloaded tokenizer/config. (LangChain Docs)
#invoke() calls Hugging Face chat completion routing, which needs at least one deployed provider. (Hugging Face)
#TinyLlama explicitly has no Inference Provider deployment, so routing returns an empty provider list and you get StopIteration. (Hugging Face)