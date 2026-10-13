from langchain_community.document_loaders import WebBaseLoader
urls = ["https://www.carwale.com/maruti-suzuki-cars/alto-k10/", "https://in.bookmyshow.com/offers"]
loader = WebBaseLoader(urls)
docs = loader.load()
# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)