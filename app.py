import streamlit as st
import os

# ==========================================
# 1. CORE ENGINE CODE (YOUR ORIGINAL LOGIC)
# ==========================================
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, CSVLoader, TextLoader

def sort_input(source: str) -> list[Document]:
    if source.startswith(("http://","https://")):
        loader = WebBaseLoader(source)
        return loader.load()
    else:
        return load_uploaded_file(source)

def load_uploaded_file(file_path: str) -> list[Document]:
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension == ".csv":
        loader = CSVLoader(file_path)
    elif extension == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"unsupported file type: {extension}")
  
    return loader.load()

# Model and database configurations
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    batch_size=10,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def formatdocs(docs):
    return "\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nContext: {context}"),
    ("human", "{question}"),
])


# ==========================================
# 2. STATE INTERFACE PERSISTENCE HANDLING
# ==========================================
# Keeps your database pipeline intact during page reruns
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

def run_rag_chain(user_message: str) -> str:
    if st.session_state.rag_chain is None:
        return "No background knowledge base has been established yet. Please paste a URL or use the '+' button to submit documents."
    return st.session_state.rag_chain.invoke(user_message)


# ==========================================
# 3. STREAMLIT INTERFACE & UI WORKFLOW
# ==========================================
st.set_page_config(page_title="RAG Chat Application", layout="wide")

# Persistent message container initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Send me a message, a web URL, or click the '+' button to upload documents."}
    ]

# --- Left Hand Sidebar ---
with st.sidebar:
    st.title("⚙️ Controls")
    st.write("Manage your running chat context.")
    
    # Fully functional chat clean wipe button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you now?"}
        ]
        st.session_state.vectorstore = None
        st.session_state.retriever = None
        st.session_state.rag_chain = None
        st.rerun()

# --- Center Screen Chat Window ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


# --- File Upload Modal Pop-Up Dialog ---
@st.dialog("📤 Upload Files to RAG Engine")
def upload_file_popup():
    st.write("Select local files to feed directly into the `sort_input` router.")
    uploaded_files = st.file_uploader(
        "Supported formats: PDF, CSV, TXT", 
        type=["pdf", "csv", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Submit Files"):
        if uploaded_files:
            all_documents = []
            
            for uploaded_file in uploaded_files:
                # Store stream buffers onto disk temporarily for path-based LangChain loaders
                temp_path = os.path.join(".", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Execute your sort_input pipeline logic
                data = sort_input(temp_path)
                all_documents.extend(data)
                
                # Immediate temporary disk storage cleaning
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Integrating the text splitter to compute the necessary 'splitted' documents array
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splitted = text_splitter.split_documents(all_documents)
            
            # Prints original validation diagnostics to console terminal
            print(len(splitted))
            
            # Build and initialize vectorstore references within the session state
            st.session_state.vectorstore = Chroma.from_documents(
                documents=splitted,
                embedding=embedding_model
            )
            
            print(st.session_state.vectorstore._collection.count())
            print(st.session_state.vectorstore._collection.get())
            
            st.session_state.retriever = st.session_state.vectorstore.as_retriever()
            
            # Build running execution runnable pipeline chain mapping
            st.session_state.rag_chain = (
                {"context": st.session_state.retriever | formatdocs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
                
            st.success(f"Successfully processed {len(uploaded_files)} file(s) into database!")
            st.rerun()
        else:
            st.error("Please upload at least one file before submitting.")


# --- Bottom Menu: Input Interface Control Rows ---
bottom_menu = st.container()
with bottom_menu:
    # Explicit integer ratio column mapping layout structures
    col_button, col_input = st.columns([1, 15])
    
    with col_button:
        if st.button("➕", use_container_width=True, help="Click to upload local documents"):
            upload_file_popup()
            
    with col_input:
        user_text = st.chat_input("Ask a question or paste a website URL (http://...)")


# --- Input Router Parsing Evaluator Rules ---
if user_text:
    # Append and render user text entry immediately into center window layout
    st.session_state.messages.append({"role": "user", "content": user_text})
    with chat_container:
        with st.chat_message("user"):
            st.write(user_text)
            
    # Process Web URL parsing routine if input is web address string
    if user_text.startswith(("http://", "https://")):
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Scraping webpage URL via sort_input..."):
                    data = sort_input(user_text)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splitted = text_splitter.split_documents(data)
                    
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=splitted,
                        embedding=embedding_model
                    )
                    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
                    st.session_state.rag_chain = (
                        {"context": st.session_state.retriever | formatdocs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    response_text = f"URL context parsed successfully! I have incorporated '{user_text}' into my knowledge base."
                    st.write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
    # Execute RAG pipeline invoke chain response for text queries
    else:
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_text = run_rag_chain(user_text)
                    st.write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
