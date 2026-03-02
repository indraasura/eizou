from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
from pptx import Presentation
from dotenv import load_dotenv, find_dotenv
import io
from pathlib import Path
from supabase import create_client, Client

# --- LANGCHAIN IMPORTS ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SUPABASE & STORAGE INIT ---
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

STORAGE_DIR = Path(__file__).parent.parent / "storage" / "vectors"
os.makedirs(STORAGE_DIR, exist_ok=True)

# --- AUTH & RBAC DEPENDENCIES ---
def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Token")
    token = authorization.split(" ")[1]
    
    try:
        # Verify JWT with Supabase
        user_response = supabase.auth.get_user(token)
        user_id = user_response.user.id
        email = user_response.user.email
        
        # Fetch Role from Profiles
        profile = supabase.table("profiles").select("role").eq("id", user_id).execute()
        role = profile.data[0]["role"] if profile.data else "user"
        
        return {"id": user_id, "email": email, "role": role}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or Expired Token")

def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# --- AUTH ENDPOINTS ---
@app.post("/login")
def login(email: str = Form(...), password: str = Form(...)):
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        token = res.session.access_token
        
        # Get role
        profile = supabase.table("profiles").select("role").eq("id", res.user.id).execute()
        role = profile.data[0]["role"] if profile.data else "user"
        
        return {"token": token, "role": role, "email": res.user.email}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# --- ADMIN ENDPOINTS ---
@app.post("/admin/users")
def create_user(email: str = Form(...), password: str = Form(...), role: str = Form(...), admin: dict = Depends(require_admin)):
    try:
        # Create user in Auth using Admin API
        res = supabase.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True
        })
        new_user_id = res.user.id
        
        # Update their role in the profiles table (auto-created by the trigger)
        supabase.table("profiles").update({"role": role}).eq("id", new_user_id).execute()
        return {"status": "User created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/admin/projects")
def create_project(name: str = Form(...), admin: dict = Depends(require_admin)):
    res = supabase.table("projects").insert({"name": name}).execute()
    return {"status": "Project created", "data": res.data}

@app.post("/admin/assign")
def assign_user(user_email: str = Form(...), project_id: int = Form(...), admin: dict = Depends(require_admin)):
    # Find user ID by email
    profile = supabase.table("profiles").select("id").eq("email", user_email).execute()
    if not profile.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_id = profile.data[0]["id"]
    try:
        supabase.table("project_users").insert({"project_id": project_id, "user_id": user_id}).execute()
        return {"status": "User assigned"}
    except Exception:
        return {"status": "User already assigned"}

@app.get("/admin/list_users")
def list_users(admin: dict = Depends(require_admin)):
    profiles = supabase.table("profiles").select("*").execute()
    return {"users": profiles.data}

# --- USER ENDPOINTS ---
@app.get("/projects")
def get_user_projects(user: dict = Depends(get_current_user)):
    if user["role"] == "admin":
        projects = supabase.table("projects").select("*").execute()
        return {"projects": projects.data}
    else:
        # Query projects mapped to this user
        assignments = supabase.table("project_users").select("project_id").eq("user_id", user["id"]).execute()
        project_ids = [a["project_id"] for a in assignments.data]
        
        if not project_ids:
            return {"projects": []}
            
        projects = supabase.table("projects").select("*").in_("id", project_ids).execute()
        return {"projects": projects.data}

# --- AI DATA ENDPOINTS (PERSISTENT MEMORY) ---
@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...), project_id: int = Form(...), model: str = Form(...), admin: dict = Depends(require_admin)):
    documents = []
    for file in files:
        contents = await file.read()
        ext = file.filename.split('.')[-1].lower()
        if ext in ['xlsx', 'xls']:
            raw_text = pd.read_excel(io.BytesIO(contents)).to_markdown()
        elif ext == 'csv':
            raw_text = pd.read_csv(io.BytesIO(contents)).to_markdown()
        elif ext in ['pptx', 'ppt']:
            prs = Presentation(io.BytesIO(contents))
            raw_text = "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")])
        else:
            raw_text = contents.decode('utf-8', errors='ignore')
            
        if raw_text:
            documents.append(Document(page_content=raw_text, metadata={"source": file.filename}))
            
    if not documents:
        raise HTTPException(status_code=400, detail="No readable text found.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(documents)
    
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    db_key = "gemini" if "gemini" in model.lower() else "openai"
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_key, transport="rest") if db_key == "gemini" else OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key)
        
    vector_store = FAISS.from_documents(chunks, embeddings)
    save_path = str(STORAGE_DIR / f"project_{project_id}_{db_key}")
    vector_store.save_local(save_path)
    
    return {"status": "success", "message": f"Data permanently saved to Project {project_id}."}

def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)

@app.post("/chat")
async def chat(message: str = Form(...), project_id: int = Form(...), model: str = Form(...), user: dict = Depends(get_current_user)):
    db_key = "gemini" if "gemini" in model.lower() else "openai"
    save_path = str(STORAGE_DIR / f"project_{project_id}_{db_key}")
    
    if not os.path.exists(save_path):
        raise HTTPException(status_code=400, detail="No data uploaded for this project yet.")
        
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_key, transport="rest") if db_key == "gemini" else OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key)
    vector_store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_key, temperature=0, transport="rest") if db_key == "gemini" else ChatOpenAI(model="gpt-4o", openai_api_key=openai_key, temperature=0)
        
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an enterprise data assistant. Answer based on the context. If charts are requested, output a JSON array wrapped in a ```chart ``` block for Chart.js. Context: {context}"),
        ("human", "{input}"),
    ])
    
    rag_chain = ({"context": vector_store.as_retriever(search_kwargs={"k": 50}) | format_docs, "input": RunnablePassthrough()} | prompt_template | llm | StrOutputParser())
    
    return {"answer": rag_chain.invoke(message)}