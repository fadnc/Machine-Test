import os
from dotenv import load_dotenv
load_dotenv()

#rag chatbot for resume analysis

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(r"D:\PRJKT\test\FADHIL MUHAMMED N C.pdf") 
document = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50
)
documents = splitter.split_documents(document)

from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_chroma import Chroma
db = Chroma.from_documents(documents=documents, embedding=embedding)

question = input("Ask your question:... ")

docs = db.similarity_search(question, k=3) 

from langchain_groq import ChatGroq
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

context = "\n\n".join(doc.page_content for doc in docs)

prompt = f"""
You are a helpful Assistant

Answer the user's question only usin gthe provide context.

If the answer cannot be  found within the provided context, reply exactly:
"Sorry! The answer to that query is not found in the given context.

Context: {context}

Question: {question}

Answer:
"""

response = llm.invoke(prompt)
print("\nAnswer")
print(response.content)