from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#chat_template
chat_template = ChatPromptTemplate([
    ('system', "You are a helpful customer assistant."),
    MessagesPlaceholder(variable_name="chat_history"),#to load chat history dynamically
    ('human', '{query}')
])

#load history
chat_history = []
with open('Messages\chat_history.txt', 'r') as file:
    chat_history.extend(file.readlines())
    

#print prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Can you help me with my order?'})
print(prompt)   