from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size =  30, 
    chunk_overlap = 3
)

text = """The Northern Plains, south of the Himalayan Region, is made up of the Gangetic Plains formed by the Indus, Ganges, and Brahmaputra river systems. This region is a large and fertile alluvial plain that stretches across several states and is the country's primary agricultural hub.

The Peninsular Plateau: South of the Northern Plains is the Peninsular Plateau, an ancient landmass composed of the Malwa plateau in the north and the Deccan plateau in the south. This region is rich in minerals and is bordered by the Western and Eastern Ghats, a range of hills running along the coasts.

The Coastal Plains include the Western Coastal Plains, which stretch from Gujarat to Kerala, and the Eastern Coastal Plains, extending from West Bengal to Tamil Nadu. These regions, located between the peninsular plateau and the respective seas, feature wide and fertile deltas formed by the rivers flowing into the Arabian Sea and the Bay of Bengal.

"""
Chunks = text_splitter.split_text(text)
print(Chunks)