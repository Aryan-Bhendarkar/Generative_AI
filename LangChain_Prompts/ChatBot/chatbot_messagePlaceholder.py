from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Chat Template 
chat_template = ChatPromptTemplate([
    ('system', 'You are Helpful customer support agent.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# load chat history
chat_history = []
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())


# Create Prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'where is my refund ?'})

print(prompt)