# Python
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load a free sentence embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Example usage
text = ["LangChain is awesome!", "Free embedding models are available."]
embeddings = embedding_model.embed_documents(text)
print(embeddings)