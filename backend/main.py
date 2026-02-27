from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
from pptx import Presentation
from dotenv import load_dotenv
import io

# SSL Bypass for strict networks
import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = "" # <--- Add this line to force httpx to bypass

load_dotenv()
app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False, # Must be False when origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to hold vector stores in memory
active_databases = {}

def read_excel(file_bytes):
    df = pd.read_excel(io.BytesIO(file_bytes))
    return df.to_markdown()

def read_csv(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.to_markdown()

def read_ppt(file_bytes):
    prs = Presentation(io.BytesIO(file_bytes))
    text_runs = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text_runs.append(shape.text)
    print(text_runs)
    return "\n".join(text_runs)

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...), model: str = Form(...)):
    documents = []
    
    try:
        for file in files:
            contents = await file.read()
            ext = file.filename.split('.')[-1].lower()
            raw_text = ""
            
            # Now supports CSVs alongside Excel and PPT
            if ext in ['xlsx', 'xls']:
                raw_text = read_excel(contents)
            elif ext == 'csv':
                raw_text = read_csv(contents)
            elif ext in ['pptx', 'ppt']:
                raw_text = read_ppt(contents)
            else:
                # Fallback to try reading as plain text if format is unknown
                raw_text = contents.decode('utf-8', errors='ignore')
                
            if raw_text:
                documents.append(Document(page_content=raw_text, metadata={"source": file.filename}))
                
    except Exception as e:
        # If pandas fails (e.g., missing tabulate library), report it cleanly
        raise HTTPException(status_code=500, detail=f"File parsing error: {str(e)}")
            
    if not documents:
        raise HTTPException(status_code=400, detail="No readable text found in the uploaded files.")

    try:
        # Keep chunks massive for tabular data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_documents(documents)
        
        if "gemini" in model.lower():
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", transport="rest")
            db_key = "gemini"
        else:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            db_key = "openai"
            
        vector_store = FAISS.from_documents(chunks, embeddings)
        active_databases[db_key] = vector_store
        
        return {"status": "success", "message": f"Data synced to {db_key.upper()}."}
        
    except Exception as e:
        # Catch AI/API Key errors and send them to the frontend
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@app.post("/chat")
async def chat(message: str = Form(...), model: str = Form(...)):
    db_key = "gemini" if "gemini" in model.lower() else "openai"
    
    if db_key not in active_databases:
        raise HTTPException(status_code=400, detail=f"Database not synced for {db_key.upper()}. Please upload files first.")
        
    vector_store = active_databases[db_key]
    
    try:
        if "gemini" in db_key:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, transport="rest")
        else:
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            
        system_prompt = (
            "You are a meticulous enterprise data assistant. Use the provided context to answer the user's question. "
            "CRITICAL INSTRUCTIONS:\n"
            "1. TABLES: When asked to list, group, or show all items, output a COMPLETE and exhaustive Markdown table. Do not truncate or summarize.\n"
            "2. CHARTS: If the user explicitly asks for a chart or graph, output a valid JSON object wrapped EXACTLY in a ```chart ... ``` code block. "
            "The JSON must be a valid Chart.js configuration object containing 'type' (e.g., 'bar', 'pie', 'line', 'doughnut'), 'data' (labels and datasets), and 'options'. "
            "Keep the chart design modern, minimal, and use aesthetic colors.\n"
            "If you don't know the answer based on the context, say you don't know.\n\n"
            "Context: {context}"
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 50})
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        answer = rag_chain.invoke(message)
        return {"answer": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation error: {str(e)}")