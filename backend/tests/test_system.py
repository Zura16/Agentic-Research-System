import os
import shutil
import pytest
from backend.core.config import settings
from backend.core.ingestion import ingest_and_index_file, retrieve_context
from backend.core.agent_graph import build_workflow
from backend.core.evaluation import (
    evaluate_faithfulness, 
    evaluate_answer_relevance, 
    evaluate_context_precision
)

@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """Initializes settings and cleans up test folders before/after tests."""
    # Force Mock Mode for testing
    settings.MOCK_MODE = True
    settings.VECTOR_STORE_PATH = "test_vector_store"
    
    # Ensure test directories exist
    os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
    os.makedirs("test_uploads", exist_ok=True)
    
    yield
    
    # Clean up test directories
    if os.path.exists(settings.VECTOR_STORE_PATH):
        shutil.rmtree(settings.VECTOR_STORE_PATH)
    if os.path.exists("test_uploads"):
        shutil.rmtree("test_uploads")


def test_document_ingestion_and_retrieval():
    """Tests writing a file, indexing it with FAISS, and retrieving matches."""
    test_file_path = "test_uploads/sample.txt"
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write("Local task scheduling is managed using cron and timers. It supports high reliability and progressive push strategies.")
        
    # Ingest
    info = ingest_and_index_file(test_file_path, "sample.txt")
    assert info["filename"] == "sample.txt"
    assert info["chunks"] > 0
    assert info["char_count"] > 0
    
    # Retrieve
    results = retrieve_context("How does local task scheduling work?")
    assert len(results) > 0
    assert results[0]["source"] == "sample.txt"
    assert "cron" in results[0]["content"].lower()


def test_multi_agent_graph_execution():
    """Tests that the LangGraph workflow compiles, runs, and returns expected state keys."""
    workflow = build_workflow()
    initial_state = {
        "query": "Explain local task scheduling.",
        "retrieved_chunks": [],
        "current_response": "",
        "validation_result": {},
        "feedback": "",
        "agent_trace": [],
        "retry_count": 0
    }
    
    output = workflow.invoke(initial_state)
    
    assert "current_response" in output
    assert "validation_result" in output
    assert "agent_trace" in output
    assert "retry_count" in output
    assert len(output["agent_trace"]) > 0
    assert output["retry_count"] >= 1  # Should run the mock hallucination retry loop!
    assert output["validation_result"]["status"] == "APPROVED"


def test_evaluation_metrics():
    """Tests faithfulness, relevance, and precision calculations in Mock Mode."""
    response = "The system supports local task scheduling using cron."
    chunks = [
        {"content": "Local task scheduling is managed using cron and timers.", "source": "sample.txt", "score": 0.9}
    ]
    
    faith_score = evaluate_faithfulness(response, chunks)
    assert 0.0 <= faith_score <= 1.0
    
    rel_score = evaluate_answer_relevance("How does local task scheduling work?", response)
    assert 0.0 <= rel_score <= 1.0
    
    prec_score = evaluate_context_precision("How does local task scheduling work?", chunks)
    assert 0.0 <= prec_score <= 1.0
