import random
import re
from langchain_core.embeddings import Embeddings

class MockEmbeddings(Embeddings):
    """Deterministic mock embeddings for offline FAISS usage without API keys."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            # Create a simple deterministic float list of size 1536
            val = sum(ord(c) for c in text[:100]) / 10000.0
            results.append([((val + i * 0.001) % 1.0) for i in range(1536)])
        return results

    def embed_query(self, text: str) -> list[float]:
        val = sum(ord(c) for c in text[:100]) / 10000.0
        return [((val + i * 0.001) % 1.0) for i in range(1536)]

def simulate_agent_flow(query: str, context_chunks: list[dict]) -> dict:
    """
    Simulates a multi-agent validation loop with step-by-step traces.
    Showcases retrieval, hallucination detection, correction, and final response.
    """
    query_lower = query.lower()
    
    # 1. Identify key topics or keywords in query and context
    keywords = re.findall(r'\b\w{4,}\b', query_lower)
    context_text = " ".join([c.get("content", "") for c in context_chunks])
    
    # Extract some sentences from the context that might match keywords
    matched_sentences = []
    if context_text:
        sentences = re.split(r'(?<=[.!?])\s+', context_text)
        for s in sentences:
            if any(kw in s.lower() for kw in keywords) and len(s.strip()) > 10:
                matched_sentences.append(s.strip())
        
        # Fallback to first few sentences if no keyword matches
        if not matched_sentences and len(sentences) > 0:
            matched_sentences = sentences[:3]
    
    context_str = " | ".join(matched_sentences[:3]) if matched_sentences else "No document context uploaded."
    
    # 2. Setup pre-cooked template or dynamically construct response
    if not context_chunks:
        # No context uploaded
        initial_answer = "I couldn't find any relevant local documents to answer this question. Based on general knowledge, I can state that AI Research requires structured ingestion, vector databases, and evaluation."
        corrected_answer = initial_answer
        hallucination_detected = False
        feedback = ""
        faithfulness_score = 0.5
        relevance_score = 0.4
        precision_score = 0.0
        confidence = 0.3
    else:
        # Dynamic response simulation
        # Introduce a deliberate mock hallucination in the first step to demonstrate the correction loop!
        fact_match = "extracted information"
        if matched_sentences:
            first_match = matched_sentences[0]
            # Try to grab a word/number to mutate for a hallucination
            numbers = re.findall(r'\b\d+\b', first_match)
            if numbers:
                mutated_number = str(int(numbers[0]) + 10)
                initial_answer = f"According to the documents, we observe a key metric of {mutated_number}. In addition, {first_match[:120]}..."
                feedback = f"Hallucination Detected: The response stated the metric is {mutated_number}, but the source document says it is {numbers[0]}."
                corrected_answer = f"Based on the retrieved context, the correct metric value is {numbers[0]}. Additionally, the documents state: '{first_match}'."
                hallucination_detected = True
            else:
                initial_answer = f"The documents indicate that {first_match[:150]} (unverified claim that the author was Albert Einstein)."
                feedback = "Hallucination Detected: The answer claims the author was Albert Einstein, which is not mentioned in the context."
                corrected_answer = f"According to the source documents: {first_match}"
                hallucination_detected = True
        else:
            initial_answer = "The documents contain general information. I assume the system performs well in 100% of cases."
            feedback = "Hallucination Detected: Claim of '100% of cases' is ungrounded."
            corrected_answer = "The documents contain general information, suggesting task processing and retrieval are supported."
            hallucination_detected = True

        faithfulness_score = 0.95
        relevance_score = 0.92
        precision_score = 0.88
        confidence = 0.91

    # Build sequential traces
    traces = []
    
    # Step 1: Retrieval
    traces.append({
        "agent": "RetrievalAgent",
        "action": "QUERY_EXPANSION",
        "message": f"Expanding query '{query}' -> keywords: {', '.join(keywords[:4])}."
    })
    traces.append({
        "agent": "RetrievalAgent",
        "action": "RETRIEVE",
        "message": f"Found {len(context_chunks)} relevant chunk(s) in local FAISS vector store. Highest similarity score: {random.uniform(0.78, 0.89):.4f}."
    })
    
    # Step 2: Reasoning (Initial)
    traces.append({
        "agent": "ReasoningAgent",
        "action": "SYNTHESIZE",
        "message": f"Generating answer using {len(context_chunks)} context chunks. Temperature set to 0.2."
    })
    traces.append({
        "agent": "ReasoningAgent",
        "action": "DRAFT",
        "message": f"Draft response: \"{initial_answer}\""
    })
    
    # Step 3: Validation (Attempt 1)
    traces.append({
        "agent": "ValidationAgent",
        "action": "GROUNDING_CHECK",
        "message": "Auditing claims against retrieved chunks..."
    })
    
    if hallucination_detected:
        traces.append({
            "agent": "ValidationAgent",
            "action": "REJECT",
            "message": f"Validation Failed. {feedback} Re-routing back to ReasoningAgent for correction."
        })
        
        # Step 4: Reasoning (Revision)
        traces.append({
            "agent": "ReasoningAgent",
            "action": "CORRECT",
            "message": f"Applying validator feedback. Modifying response to ground claims precisely in text."
        })
        traces.append({
            "agent": "ReasoningAgent",
            "action": "REVISE",
            "message": f"Revised response: \"{corrected_answer}\""
        })
        
        # Step 5: Validation (Attempt 2)
        traces.append({
            "agent": "ValidationAgent",
            "action": "GROUNDING_CHECK",
            "message": "Re-auditing revised claims against context..."
        })
        traces.append({
            "agent": "ValidationAgent",
            "action": "APPROVE",
            "message": f"Validation Passed. Faithfulness score: {faithfulness_score:.2f}. No hallucinations detected."
        })
    else:
        traces.append({
            "agent": "ValidationAgent",
            "action": "APPROVE",
            "message": f"Validation Passed. Faithfulness score: {faithfulness_score:.2f}. Answer is fully grounded."
        })

    return {
        "answer": corrected_answer if hallucination_detected else initial_answer,
        "traces": traces,
        "evaluation": {
            "faithfulness": faithfulness_score,
            "answer_relevance": relevance_score,
            "context_precision": precision_score,
            "confidence": confidence
        }
    }
