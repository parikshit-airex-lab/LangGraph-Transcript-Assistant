from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "documents",
    glob = "*.pdf",
    loader_cls = PyPDFLoader
)

docs = loader.load()
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)