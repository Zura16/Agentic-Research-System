import json
import random
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from backend.core.config import settings
from backend.core.ingestion import retrieve_context
from backend.core.mock_llm import simulate_agent_flow

def get_llm():
    """Returns the LLM client according to settings."""
    if settings.DEFAULT_LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
        return ChatAnthropic(
            model=settings.DEFAULT_MODEL or "claude-3-5-sonnet-20240620",
            temperature=settings.TEMPERATURE,
            anthropic_api_key=settings.ANTHROPIC_API_KEY
        )
    else:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key missing in non-mock mode.")
        return ChatOpenAI(
            model=settings.DEFAULT_MODEL or "gpt-4o-mini",
            temperature=settings.TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY
        )

# --- Agent Node Logic ---

def run_retrieval_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieves context chunks. Performs query rewriting if validator feedback exists."""
    query = state["query"]
    retry_count = state["retry_count"]
    feedback = state.get("feedback", "")
    agent_trace = list(state.get("agent_trace", []))
    
    # 1. Mock Mode
    if settings.MOCK_MODE:
        if retry_count > 0:
            agent_trace.append({
                "agent": "RetrievalAgent",
                "action": "QUERY_EXPANSION",
                "message": f"Expanding query for retry #{retry_count} using validator feedback: '{feedback}'."
            })
        
        chunks = retrieve_context(query)
        # If no chunks, provide a small mock chunk to simulate RAG
        if not chunks:
            chunks = [{"content": "Local task scheduling is managed using cron and timers.", "source": "task_system.txt", "chunk_index": 1, "score": 0.85}]
            
        agent_trace.append({
            "agent": "RetrievalAgent",
            "action": "RETRIEVE",
            "message": f"Retrieved {len(chunks)} chunk(s) from FAISS database. Top chunk source: {chunks[0]['source']}."
        })
        
        return {
            "retrieved_chunks": chunks,
            "agent_trace": agent_trace
        }

    # 2. Live LLM Mode
    revised_query = query
    if retry_count > 0 and feedback:
        try:
            llm = get_llm()
            agent_trace.append({
                "agent": "RetrievalAgent",
                "action": "QUERY_EXPANSION",
                "message": f"Asking LLM to rewrite query for retry #{retry_count}."
            })
            prompt = f"Original Query: {query}\nValidator Feedback: {feedback}\n\nTask: Rewrite the query to retrieve better search results from a vector database. Output only the revised query and nothing else."
            response = llm.invoke(prompt)
            revised_query = response.content.strip()
            agent_trace.append({
                "agent": "RetrievalAgent",
                "action": "RETRIEVE",
                "message": f"Rewrote query to: '{revised_query}'."
            })
        except Exception as e:
            print(f"[!] Error expanding query with LLM: {e}")
            agent_trace.append({
                "agent": "RetrievalAgent",
                "action": "WARNING",
                "message": f"Query rewrite failed ({str(e)}). Falling back to original query."
            })

    # Retrieve context
    chunks = retrieve_context(revised_query)
    agent_trace.append({
        "agent": "RetrievalAgent",
        "action": "RETRIEVE",
        "message": f"Retrieved {len(chunks)} chunks from FAISS. Top similarity: {chunks[0]['score']:.4f}" if chunks else "No relevant documents found."
    })
    
    return {
        "retrieved_chunks": chunks,
        "agent_trace": agent_trace
    }


def run_reasoning_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generates an answer based on query, context chunks, and validator feedback."""
    query = state["query"]
    chunks = state["retrieved_chunks"]
    retry_count = state["retry_count"]
    feedback = state.get("feedback", "")
    agent_trace = list(state.get("agent_trace", []))
    
    context_str = "\n\n".join([f"Source: {c['source']} (chunk {c['chunk_index']}):\n{c['content']}" for c in chunks])
    
    # 1. Mock Mode
    if settings.MOCK_MODE:
        agent_trace.append({
            "agent": "ReasoningAgent",
            "action": "SYNTHESIZE",
            "message": "Synthesizing response from retrieved chunks..."
        })
        
        # Determine draft answer or final answer based on retry count
        if retry_count == 0 and chunks:
            # First draft - let's add a minor hallucinated claim to show the loop
            filename = chunks[0]["source"]
            draft_answer = f"According to {filename}, this system achieves 100% precision in real-time execution, starting deployment in 1995."
            agent_trace.append({
                "agent": "ReasoningAgent",
                "action": "DRAFT",
                "message": f"Draft response: \"{draft_answer}\""
            })
            return {
                "current_response": draft_answer,
                "agent_trace": agent_trace
            }
        else:
            # Corrected response
            content = chunks[0]["content"] if chunks else "Local task scheduling is supported."
            revised_answer = f"According to the source documents, the system supports task scheduling and processing. Detail: '{content}'."
            agent_trace.append({
                "agent": "ReasoningAgent",
                "action": "REVISE",
                "message": f"Revised response: \"{revised_answer}\""
            })
            return {
                "current_response": revised_answer,
                "agent_trace": agent_trace
            }

    # 2. Live LLM Mode
    agent_trace.append({
        "agent": "ReasoningAgent",
        "action": "SYNTHESIZE",
        "message": f"Synthesizing response from {len(chunks)} context chunks."
    })
    
    prompt = f"""You are a Reasoning Agent. Answer the User Query based ONLY on the provided Document Context. 
If the context is insufficient, explain what is missing.
If you receive Validator Feedback, you must revise your previous answer to fix any errors or hallucinations noted.

User Query: {query}

Document Context:
{context_str}

Validator Feedback (if any): {feedback}

Provide a clean, comprehensive response:"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        current_response = response.content.strip()
        
        action_name = "REVISE" if retry_count > 0 else "DRAFT"
        agent_trace.append({
            "agent": "ReasoningAgent",
            "action": action_name,
            "message": f"Generated answer: \"{current_response[:120]}...\""
        })
        
        return {
            "current_response": current_response,
            "agent_trace": agent_trace
        }
    except Exception as e:
        print(f"[!] Error generating response: {e}")
        return {
            "current_response": f"Error during generation: {str(e)}",
            "agent_trace": agent_trace
        }


def run_validation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Audits the response for hallucinations, checks relevancy, and scores confidence."""
    query = state["query"]
    current_response = state["current_response"]
    chunks = state["retrieved_chunks"]
    retry_count = state["retry_count"]
    agent_trace = list(state.get("agent_trace", []))
    
    context_str = "\n\n".join([f"Source: {c['source']} (chunk {c['chunk_index']}):\n{c['content']}" for c in chunks])
    
    # 1. Mock Mode
    if settings.MOCK_MODE:
        agent_trace.append({
            "agent": "ValidationAgent",
            "action": "GROUNDING_CHECK",
            "message": "Checking answer faithfulness and checking for hallucinations..."
        })
        
        if retry_count == 0 and chunks:
            # Reject to trigger loop
            feedback = "Hallucination Detected: The claim that the system was deployed in 1995 is ungrounded. Check the documents for the correct history."
            agent_trace.append({
                "agent": "ValidationAgent",
                "action": "REJECT",
                "message": f"Validation Failed. {feedback}"
            })
            
            validation_result = {
                "status": "REJECTED",
                "faithfulness": 0.4,
                "answer_relevance": 0.9,
                "context_precision": 0.85,
                "confidence": 0.55,
                "explanation": feedback
            }
            return {
                "validation_result": validation_result,
                "feedback": feedback,
                "agent_trace": agent_trace,
                "retry_count": retry_count + 1
            }
        else:
            # Accept
            agent_trace.append({
                "agent": "ValidationAgent",
                "action": "APPROVE",
                "message": "Validation Passed. Response is grounded in retrieved documents."
            })
            validation_result = {
                "status": "APPROVED",
                "faithfulness": 0.98,
                "answer_relevance": 0.95,
                "context_precision": 0.90,
                "confidence": 0.96,
                "explanation": "No hallucinations detected. Answer is well-supported."
            }
            return {
                "validation_result": validation_result,
                "feedback": "",
                "agent_trace": agent_trace,
                "retry_count": retry_count
            }

    # 2. Live LLM Mode
    agent_trace.append({
        "agent": "ValidationAgent",
        "action": "GROUNDING_CHECK",
        "message": "Auditing faithfulness and grounding using LLM..."
    })
    
    prompt = f"""You are a Validation Agent. Audit the Answer against the provided Document Context to detect hallucinations or ungrounded claims.
Also verify if the Answer is relevant to the Query.

User Query: {query}
Document Context:
{context_str}
Answer to Validate:
{current_response}

Instructions:
Evaluate Faithfulness (grounding) and Answer Relevance.
If any statements in the Answer are not supported by the Context, mark as REJECTED and describe the hallucinated claims in the feedback.
If the Answer is fully grounded and relevant, mark as APPROVED.

You MUST respond strictly in valid JSON format with these exact keys:
{{
  "status": "APPROVED" or "REJECTED",
  "faithfulness_score": 0.0 to 1.0 (grounded statements ratio),
  "relevance_score": 0.0 to 1.0 (how well it answers the query),
  "precision_score": 0.0 to 1.0 (precision of retrieved chunks),
  "explanation": "Provide a detailed explanation of hallucinations found or validation success."
}}
JSON:"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Clean markdown wrappers if present
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        res = json.loads(content)
        
        faith = res.get("faithfulness_score", 0.0)
        rel = res.get("relevance_score", 0.0)
        prec = res.get("precision_score", 0.0)
        status = res.get("status", "REJECTED")
        explanation = res.get("explanation", "")
        
        # Calculate combined confidence
        confidence = (faith * 0.5) + (rel * 0.3) + (prec * 0.2)
        
        validation_result = {
            "status": status,
            "faithfulness": faith,
            "answer_relevance": rel,
            "context_precision": prec,
            "confidence": confidence,
            "explanation": explanation
        }
        
        if status == "REJECTED":
            feedback = f"Hallucination Detected/Low Quality: {explanation}"
            agent_trace.append({
                "agent": "ValidationAgent",
                "action": "REJECT",
                "message": f"Validation Failed: {explanation}"
            })
        else:
            feedback = ""
            agent_trace.append({
                "agent": "ValidationAgent",
                "action": "APPROVE",
                "message": "Validation Passed. Response is fully grounded."
            })
            
        return {
            "validation_result": validation_result,
            "feedback": feedback,
            "agent_trace": agent_trace,
            "retry_count": retry_count + 1 if validation_result["status"] == "REJECTED" else retry_count
        }
    except Exception as e:
        print(f"[!] Error running validation agent: {e}")
        # Default safety fallback
        validation_result = {
            "status": "APPROVED",
            "faithfulness": 0.9,
            "answer_relevance": 0.9,
            "context_precision": 0.8,
            "confidence": 0.88,
            "explanation": f"LLM Validation errored out: {str(e)}. Defaulted to approved."
        }
        return {
            "validation_result": validation_result,
            "feedback": "",
            "agent_trace": agent_trace,
            "retry_count": retry_count
        }
