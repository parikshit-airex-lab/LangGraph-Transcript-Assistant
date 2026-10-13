from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "documents",
    glob = "*.pdf",
    loader_cls = PyPDFLoader
)

docs = loader.lazy_load()
for document in docs:
    print(document.page_content)
