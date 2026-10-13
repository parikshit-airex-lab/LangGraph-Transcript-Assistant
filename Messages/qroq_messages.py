from langchain_groq import ChatGroq
from langchain.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this image in detail."},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"},
        },
    ]
)

response = llm.invoke([message])
print(response.content)