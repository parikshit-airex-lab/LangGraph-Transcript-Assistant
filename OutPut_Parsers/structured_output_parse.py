from langchain_groq import ChatGroq
#from langchain.messages import HumanMessage
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.structured import (
    StructuredOutputParser,
    ResponseSchema,
)

load_dotenv()
model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

schema=[
    ResponseSchema(name='fact1',description='fact1 about topic'),
    ResponseSchema(name='fact2',description='fact2 about topic'),
    ResponseSchema(name='fact3',description='fact3 about topic')
    ]
parser=StructuredOutputParser.from_response_schema(schema)
template=PromptTemplate(
    template="Give 3 facts about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'blackhole'})
result = model.invoke(prompt)
final_res = parser.parse(result.content)
print(final_res)