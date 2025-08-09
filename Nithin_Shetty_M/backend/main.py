# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import uuid
import json
from dotenv import load_dotenv

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatPerplexity
from langchain.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter

# State management for LangGraph
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# Pydantic models for API requests
class StartLessonRequest(BaseModel):
    topic: str

class ContinueLessonRequest(BaseModel):
    session_id: str

class DoubtRequest(BaseModel):
    session_id: str
    question: str

class ResumeRequest(BaseModel):
    session_id: str

# --- Initial setup ---
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the pre-built FAISS vector store
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
try:
    faiss_db = FAISS.load_local("data/vectorstore", embeddings, allow_dangerous_deserialization=True)
except FileNotFoundError:
    raise FileNotFoundError("FAISS vector store not found. Run `build_vectorstore.py` first.")

# In-memory dictionary to store session state
# In a production environment, use Redis or a database
sessions = {}

# --- LangGraph State and Nodes ---
class AgentState(TypedDict):
    """
    Represents the state of our tutor agent.
    """
    session_id: str
    current_topic: str
    current_chunk_index: int
    lesson_chunks: List[str]
    mode: str  # "lesson" or "doubt"
    doubt_history: List[str]

def get_next_lesson_chunk(state: AgentState):
    """Retrieves the next chunk of the lesson."""
    if state["current_chunk_index"] < len(state["lesson_chunks"]):
        chunk = state["lesson_chunks"][state["current_chunk_index"]]
        state["current_chunk_index"] += 1
        return {"response": chunk}
    else:
        return {"response": "That concludes the lesson on this topic."}

def handle_doubt(state: AgentState, question: str):
    """Handles questions within the doubt modal using RAG."""
    docs = faiss_db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a 10th-grade science teacher. Use the following context to answer the student's question clearly and concisely. If the answer is not in the context, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    llm = ChatPerplexity(
    model="sonar-pro",
    pplx_api_key=PERPLEXITY_API_KEY
)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    response = chain.run(context=context, question=question)
    state["doubt_history"].append(f"Q: {question}\nA: {response}")
    return {"response": response}

# --- LangGraph Workflow Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("lesson", get_next_lesson_chunk)
workflow.add_node("doubt", handle_doubt)

# Define edges and conditional routing
workflow.add_edge("lesson", END) # The lesson simply ends for now
# We will manually transition between modes in our FastAPI endpoints

# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start_lesson")
async def start_lesson(request_body: StartLessonRequest):
    # Retrieve the most relevant chapter/subtopic from the vector store
    retrieved_docs = faiss_db.similarity_search(request_body.topic, k=1)
    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="Topic not found in knowledge base.")
    
    # Get the full content for the retrieved topic
    # This is a simplification; a more robust system would get all chunks for the relevant subtopic
    full_content = retrieved_docs[0].page_content
    
    # Chunk the retrieved content for incremental teaching
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    lesson_chunks = text_splitter.split_text(full_content)
    
    # Initialize a new session state
    session_id = str(uuid.uuid4())
    sessions[session_id] = AgentState(
        session_id=session_id,
        current_topic=request_body.topic,
        current_chunk_index=0,
        lesson_chunks=lesson_chunks,
        mode="lesson",
        doubt_history=[]
    )
    
    # Get the first chunk to start the lesson
    first_chunk = get_next_lesson_chunk(sessions[session_id])
    
    return {"session_id": session_id, "message": first_chunk["response"]}

@app.post("/continue_lesson")
async def continue_lesson(request_body: ContinueLessonRequest):
    session_id = request_body.session_id
    state = sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    # Check if we are in lesson mode
    if state["mode"] != "lesson":
        raise HTTPException(status_code=400, detail="Cannot continue lesson while in doubt mode.")

    response = get_next_lesson_chunk(state)
    return {"message": response["response"]}

@app.post("/ask_doubt")
async def ask_doubt(request_body: DoubtRequest):
    session_id = request_body.session_id
    state = sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    state["mode"] = "doubt"
    
    # Use the LangGraph node to handle the doubt
    response = handle_doubt(state, request_body.question)
    
    return {"message": response["response"]}
    
@app.post("/resume_lesson")
async def resume_lesson(request_body: ResumeRequest):
    session_id = request_body.session_id
    state = sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    state["mode"] = "lesson"
    # The next "continue_lesson" call will pick up from the correct index
    
    return {"message": "Lesson resumed."}