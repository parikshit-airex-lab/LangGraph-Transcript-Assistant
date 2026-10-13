from langchain_google_genai import GoogleGenerativeAI
from typing import TypedDict, Annotated

model = GoogleGenerativeAI(model="gemini-1.5-flash")

class Review(TypedDict):

    summary: str
    sentiment:str

structured_model = model.with_structured_output(Review)

result = model.invoke("The hardware is great but software is bloated. There are too many  pre-installed apps that I don't need. Also, the UI looks outdated compared to other brands. Hoping  forsoftware updates to improve the experience.")
print(result)