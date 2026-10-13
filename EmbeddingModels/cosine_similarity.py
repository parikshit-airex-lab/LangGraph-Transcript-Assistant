from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", dimensions=300)

document= [
    "Ankita is working in Capgemini",
    "Pragati is working in Interior Designing firm",
    "Madhavi is working in Tata Consultancy Services",
    "Rutika is working as a teacher in HighSchool"
]

query = "Where does Rutika work?"

doc_embeddings = embedding.embed_documents(document)

query_embeddings = embedding.embed_query(query)

similarity_scores = cosine_similarity([query_embeddings],doc_embeddings)[0]# value in cosine similarity should be 2d list


index, scores = sorted(list(enumerate(similarity_scores)),key=lambda x:x[1])[-1]
print(sorted(list(enumerate(similarity_scores)), key=lambda x:x[1])[-1])

print(query)
print(document[index])
print("similarity score is",scores)
