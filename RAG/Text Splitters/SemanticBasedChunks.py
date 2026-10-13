from langchain_experimental.text_splitter import SemanticChunker
from langchain_nomic import NomicEmbeddings
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from getpass import getpass
load_dotenv()
# embedding = NomicEmbeddings(model="nomic-ai/nomic-embed-text-v2-moe")
splitter = SemanticChunker(
    HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2"),
    breakpoint_threshold_type = "standard deviation",
    breakpoint_threshold_amount = 1
)
text = """ I can show you how to wrap a Groq LLM as a pseudo-embedding model using text-to-vector techniques so you can keep everything on Groq hardware — but it won’t be as efficient as a true embedding model.
officially the Republic of India,[j][19] is a country in South Asia. It is the seventh-largest country by area; the most populous country in the world[20] and, since its independence in 1947, the world's most populous democracy.
Merge Sort is a classic divide and conquer algorithm. The core idea is based on breaking down a big problem into smaller, manageable sub-problems i.e. sorting smaller arrays and then merging those solutions to get the final sorted result."""

chunks = splitter.create_documents([text])
print(chunks)
print(len(chunks))