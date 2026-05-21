from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"a:\AI-ML Track\GenAI\LangChain DocumentLoader\dl-curriculum.pdf")

docs = loader.load()
print(docs)
print(len(docs))
print(docs[0].page_content)

