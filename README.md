# Generative AI with LangChain

My notes and code from learning to build LLM applications with LangChain — models, prompts, RAG, and agents — one small step at a time.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C)
![RAG](https://img.shields.io/badge/RAG-orange)
![Agents](https://img.shields.io/badge/Agents-ReAct-6E40C9)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

---

## About this repo

I'm learning Generative AI, and instead of following one big tutorial end to end, I tried to build each piece myself and understand it before moving on. Every topic lives in its own folder as a small, runnable example, so the repo roughly follows the order I learned things — starting from a single model call and slowly working up to a RAG chatbot and a simple agent.

It's a work in progress and mostly a learning record, but the examples do run.

## What's here

A few small projects I'm happy with:

**▸ YouTube RAG Chatbot** — `LangChain RAG/YT Chatbot/`
Paste a YouTube link and ask questions about the video. It answers from the transcript and says "I don't know" when the answer isn't there. Built with Streamlit, ChromaDB, and HuggingFace embeddings.

**▸ ReAct Agent** — `LangChain Agents/agent.py`
A small agent that answers step-by-step questions by using web search and a weather API.

**▸ Currency Converter** — `LangChain Tools/currency_conversion_tool.py`
Two tools working together — one looks up the live exchange rate, the other does the conversion.

**▸ Research Paper Summarizer** — `LangChain_Prompts/Reseach Paper Summarizer/`
Summarizes a research paper using reusable prompt templates, where you pick the paper, the explanation style, and the length.

**▸ Document Similarity** — `LangChain Models/Document Similarity - Project/`
A first try at semantic search using embeddings and cosine similarity.

## Topics I worked through

| Topic | What I tried |
| ----- | ------------ |
| **Models** | LLMs, chat & embedding models across OpenAI, Anthropic, HuggingFace, OpenRouter, Gemini; a local model with Qwen |
| **Prompts** | Static & dynamic templates; a small chatbot with memory |
| **Structured Output** | Typed results with TypedDict, Pydantic & JSON Schema |
| **Output Parsers** | String, JSON, Structured & Pydantic parsers |
| **Chains** | Simple, sequential, parallel & conditional chains (LCEL) |
| **Runnables** | Rebuilt LangChain's Runnable from scratch to understand how it works |
| **Document Loaders** | Text, PDF, directory & web loaders |
| **Text Splitting** | Length, structure-based & semantic chunking |
| **Vector DB** | A ChromaDB vector store |
| **Retrievers** | Vector, MMR, Multi-Query, Compression & Wikipedia |
| **Tools** | Built-in, custom, structured & BaseTool tools |
| **Agents** | A basic ReAct agent |

## Tools used

Python · LangChain (LCEL) · ChromaDB · Streamlit · Pydantic · sentence-transformers · scikit-learn
OpenAI · Anthropic · HuggingFace · OpenRouter · Gemini

## Running the examples

```bash
git clone https://github.com/<your-username>/GenAI.git
cd GenAI
python -m venv venv && venv\Scripts\activate     # Windows
pip install -r "LangChain Models/requirements.txt"
# add your API keys to a .env file
```

To try the RAG chatbot:
```bash
cd "LangChain RAG/YT Chatbot"
pip install -r requirements.txt
streamlit run app.py
```

API keys are read from a `.env` file and aren't committed.

## About me

I'm still learning GenAI and figuring things out as I go. This repo is mostly me keeping track of what I've understood so far.
📫 [bhendarkararyan@gmail.com](mailto:bhendarkararyan@gmail.com)
