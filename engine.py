#creating a .env file to keep api key secure
env_content="""
GEMINI_API_KEY="your_api_key"
"""
with open(".env","w") as file:
  file.write(env_content)
print("file created successfully")

#setting up the model
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=os.getenv("GEMINI_API_KEY"))

#taking multiple types of files and urls as input and segregating to functions accordingly
import os

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, CSVLoader, TextLoader

def sort_input(source: str) -> list[Document]:
    if source.startswith(("http://","https://")):
        loader=WebBaseLoader(source)
        return loader.load()
    else:
      return load_uploaded_file(source)

def load_uploaded_file(file_path: str) -> list[Document]:
  _,extension=os.path.splitext(file_path) #taking only the extension and storing from created tuple
  extension=extension.lower()
  if extension == ".pdf":
    loader=PyPDFLoader(file_path)
  elif extension == ".csv":
    loader=CSVLoader(file_path)
  elif extension == ".txt":
    loader=TextLoader(file_path)
  else:
    raise ValueError(f"unsupported file type: {extension}")
  
  return loader.load()

#storing uploaded file as data
data=sort_input()

print(len(splitted))

#setting the modeland database
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
load_dotenv()

embedding_model=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    batch_size=10,
    google_api_key=os.getenv("GEMINI_API_KEY")
    )

vectorstore=Chroma.from_documents(
    documents=splitted,
    embedding=embedding_model
)

print(vectorstore._collection.count())

print(vectorstore._collection.get())

#creating the prompt to be used for question answering
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nContext: {context}"),
    ("human", "{question}"),
])

#creating the runnables
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def formatdocs(docs):
  return "\n".join(doc.page_content for doc in docs)

rag_chain=({"context":retriever | formatdocs,"question":RunnablePassthrough()}
           | prompt
           | llm
           | StrOutputParser())

def run_rag_chain(user_message: str) -> str:
    # Your working vector store retrieval and Gemini code goes here!
    # example: return rag_chain.invoke(user_message)
    return f"Response from engine for: {user_message}"