from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size =  100, 
    chunk_overlap = 30,
    separator = " "
)

text = """India, officially the Republic of India,[j][19] is a country in South Asia. It is the seventh-largest country by area; the most populous country in the world[20] and, since its independence in 1947, the world's most populous democracy.[21][22][23] Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west;[k] China, Nepal and Bhutan to the north; Bangladesh and Myanmar to the east. In the Indian Ocean, India is near Sri Lanka and the Maldives. Its Andaman and Nicobar Islands share a maritime border with Myanmar, Thailand and Indonesia."""

Chunks = text_splitter.split_text(text)
print(Chunks)