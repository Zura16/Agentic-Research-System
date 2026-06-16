from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from backend.core.config import settings
from backend.core.agents import (
    run_retrieval_agent,
    run_reasoning_agent,
    run_validation_agent
)

class AgentState(TypedDict):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    current_response: str
    validation_result: Dict[str, Any]
    feedback: str
    agent_trace: List[Dict[str, Any]]
    retry_count: int

def validation_routing_edge(state: AgentState) -> Literal["retrieve", "__end__"]:
    """Decides whether to retry retrieval or end the workflow based on validation results."""
    val_res = state.get("validation_result", {})
    status = val_res.get("status", "REJECTED")
    retry_count = state.get("retry_count", 0)
    
    if status == "REJECTED" and retry_count < settings.MAX_RETRIES:
        # Loop back to retrieval
        return "retrieve"
    else:
        # Stop
        return END

def build_workflow() -> Any:
    """Builds and compiles the multi-agent RAG workflow."""
    workflow = StateGraph(AgentState)
    
    # Define nodes
    workflow.add_node("retrieve", run_retrieval_agent)
    workflow.add_node("generate", run_reasoning_agent)
    workflow.add_node("validate", run_validation_agent)
    
    # Define entry point
    workflow.set_entry_point("retrieve")
    
    # Define simple transitions
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "validate")
    
    # Define conditional transition from validate
    workflow.add_conditional_edges(
        "validate",
        validation_routing_edge,
        {
            "retrieve": "retrieve",
            "__end__": END
        }
    )
    
    return workflow.compile()
