from langchain_anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

anthropic_model = Anthropic(model_name="claude-2", temperature=0.7)

result=anthropic_model.invoke("Create a poem on woman")
print(result.content)
