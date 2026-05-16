from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header('Research Tool')

user_input = st.text_input('Enter your Prompt: ')
model = ChatOpenRouter(model='openai/gpt-4o-mini')

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)
