import os
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.core.config import settings
from backend.core.ingestion import ingest_and_index_file
from backend.core.agent_graph import build_workflow
from backend.core.evaluation import run_batch_evaluation

app = FastAPI(
    title="Agentic Research System API",
    description="Backend service for RAG + LangGraph Multi-Agent Orchestration + Evaluation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories setup
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)


class QueryRequest(BaseModel):
    query: str


class SettingsRequest(BaseModel):
    mock_mode: bool
    openai_api_key: Optional[str] = ""
    anthropic_api_key: Optional[str] = ""
    default_llm_provider: str
    default_model: str
    temperature: float
    max_retries: int


class EvaluationTestCase(BaseModel):
    query: str


class EvaluationRequest(BaseModel):
    cases: Optional[List[EvaluationTestCase]] = None


# --- Endpoints ---

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "mock_mode": settings.MOCK_MODE,
        "openai_key_configured": bool(settings.OPENAI_API_KEY),
        "anthropic_key_configured": bool(settings.ANTHROPIC_API_KEY)
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads and ingests a PDF or TXT file into the vector store."""
    filename = file.filename
    _, ext = os.path.splitext(filename.lower())
    if ext not in (".pdf", ".txt"):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Save file to uploads folder
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Ingest and embed file
        ingest_info = ingest_and_index_file(file_path, filename)
        
        return {
            "status": "success",
            "message": f"Successfully ingested {filename}",
            "data": ingest_info
        }
    except Exception as e:
        # Clean up if failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/api/documents")
async def list_documents():
    """Lists all uploaded documents."""
    if not os.path.exists(UPLOAD_DIR):
        return []
    
    files = []
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if os.path.isfile(fpath) and fname.lower().endswith((".pdf", ".txt")):
            stats = os.stat(fpath)
            files.append({
                "filename": fname,
                "size": stats.st_size,
                "created_at": stats.st_ctime
            })
    return files


@app.post("/api/query")
async def query_research_system(payload: QueryRequest):
    """Executes the LangGraph multi-agent loop over the vector store to answer queries."""
    query = payload.query
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        workflow = build_workflow()
        
        # Initialize LangGraph state
        initial_state = {
            "query": query,
            "retrieved_chunks": [],
            "current_response": "",
            "validation_result": {},
            "feedback": "",
            "agent_trace": [],
            "retry_count": 0
        }
        
        # Run graph
        output = workflow.invoke(initial_state)
        
        return {
            "query": query,
            "answer": output.get("current_response", ""),
            "retrieved_chunks": output.get("retrieved_chunks", []),
            "validation": output.get("validation_result", {}),
            "trace": output.get("agent_trace", []),
            "retry_count": output.get("retry_count", 0)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/api/settings")
async def get_settings():
    """Returns current system configuration settings."""
    return {
        "mock_mode": settings.MOCK_MODE,
        "openai_api_key": "***" if settings.OPENAI_API_KEY else "",
        "anthropic_api_key": "***" if settings.ANTHROPIC_API_KEY else "",
        "default_llm_provider": settings.DEFAULT_LLM_PROVIDER,
        "default_model": settings.DEFAULT_MODEL,
        "temperature": settings.TEMPERATURE,
        "max_retries": settings.MAX_RETRIES
    }


@app.post("/api/settings")
async def update_settings(payload: SettingsRequest):
    """Updates system configuration settings in-memory."""
    settings.MOCK_MODE = payload.mock_mode
    
    # Only update API keys if provided (not placeholder mask)
    if payload.openai_api_key and payload.openai_api_key != "***":
        settings.OPENAI_API_KEY = payload.openai_api_key
        os.environ["OPENAI_API_KEY"] = payload.openai_api_key
    elif payload.openai_api_key == "":
        settings.OPENAI_API_KEY = ""
        os.environ["OPENAI_API_KEY"] = ""

    if payload.anthropic_api_key and payload.anthropic_api_key != "***":
        settings.ANTHROPIC_API_KEY = payload.anthropic_api_key
        os.environ["ANTHROPIC_API_KEY"] = payload.anthropic_api_key
    elif payload.anthropic_api_key == "":
        settings.ANTHROPIC_API_KEY = ""
        os.environ["ANTHROPIC_API_KEY"] = ""
        
    settings.DEFAULT_LLM_PROVIDER = payload.default_llm_provider
    settings.DEFAULT_MODEL = payload.default_model
    settings.TEMPERATURE = payload.temperature
    settings.MAX_RETRIES = payload.max_retries
    
    return {"status": "success", "message": "Settings updated successfully"}


@app.post("/api/evaluate")
async def batch_evaluate(payload: EvaluationRequest):
    """Runs a batch evaluation of the system against a list of test queries."""
    cases = payload.cases
    
    # Default Q&A evaluation dataset if none provided
    if not cases or len(cases) == 0:
        cases = [
            EvaluationTestCase(query="How does local task scheduling work?"),
            EvaluationTestCase(query="Explain the multi-agent validation loop in LangGraph."),
            EvaluationTestCase(query="What is faithfulness in RAGAS evaluation?"),
            EvaluationTestCase(query="How does the system handle hallucination detection?")
        ]
        
    test_cases_dicts = [{"query": c.query} for c in cases]
    
    try:
        report = run_batch_evaluation(test_cases_dicts)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# Mount static files or index.html if needed in production
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
