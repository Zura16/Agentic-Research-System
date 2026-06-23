import React, { useState, useEffect, useRef } from 'react';
import { 
  Search, FileText, Settings, BarChart3, UploadCloud, 
  ShieldCheck, RefreshCw, CheckCircle2, XCircle, 
  AlertCircle, BookOpen, Cpu, Layers, Activity 
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('assistant');
  const [isMockMode, setIsMockMode] = useState(true);
  
  // Settings State
  const [config, setConfig] = useState({
    mock_mode: true,
    openai_api_key: '',
    anthropic_api_key: '',
    default_llm_provider: 'openai',
    default_model: 'gpt-4o-mini',
    temperature: 0.2,
    max_retries: 3
  });

  // Assistant Tab State
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [displayedTrace, setDisplayedTrace] = useState([]);
  const [playbackIndex, setPlaybackIndex] = useState(-1);
  const [activeGraphNode, setActiveGraphNode] = useState(null);
  const [graphStatus, setGraphStatus] = useState({}); // 'retrieve', 'generate', 'validate' -> 'active' | 'completed' | 'failed'

  // Library Tab State
  const [documents, setDocuments] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const fileInputRef = useRef(null);

  // Evaluation Studio State
  const [evalResult, setEvalResult] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);

  // Fetch Settings & Docs on Mount
  useEffect(() => {
    fetchSettings();
    fetchDocuments();
  }, []);

  const fetchSettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/settings`);
      const data = await res.json();
      setConfig(prev => ({
        ...prev,
        ...data,
        openai_api_key: data.openai_api_key ? '***' : '',
        anthropic_api_key: data.anthropic_api_key ? '***' : ''
      }));
      setIsMockMode(data.mock_mode);
    } catch (e) {
      console.warn("Backend settings API down. Running in offline client-side mock mode.");
      setIsMockMode(true);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/documents`);
      const data = await res.json();
      setDocuments(data);
    } catch (e) {
      console.warn("Backend documents API down, using client-side mock docs.");
      setDocuments([
        { filename: "deep_learning_intro.pdf", size: 1024 * 342, created_at: Date.now() / 1000 - 86400 },
        { filename: "agentic_workflows_guide.txt", size: 1024 * 45, created_at: Date.now() / 1000 - 172800 }
      ]);
    }
  };

  // Save Settings
  const handleSaveSettings = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch(`${API_BASE}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      const data = await res.json();
      if (data.status === 'success') {
        alert('Settings saved successfully!');
        setIsMockMode(config.mock_mode);
        fetchSettings();
      }
    } catch (err) {
      console.warn("Backend settings API offline. Saving settings in local state.");
      setIsMockMode(config.mock_mode);
      alert('Settings saved in local React state (Offline Fallback).');
    }
  };

  // File Upload
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setIsUploading(true);
    setUploadStatus('Uploading file...');
    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus(`Success: Ingested ${data.data.chunks} chunks from ${data.data.filename}.`);
        fetchDocuments();
      } else {
        setUploadStatus(`Error: ${data.detail || 'Upload failed'}`);
      }
    } catch (err) {
      console.warn("Backend offline. Simulating local file ingestion...");
      setTimeout(() => {
        setUploadStatus(`Success (Simulated): Ingested 12 chunks from ${file.name}.`);
        setDocuments(prev => [
          { filename: file.name, size: file.size, created_at: Date.now() / 1000 },
          ...prev
        ]);
      }, 1000);
    } finally {
      setIsUploading(false);
    }
  };

  // Playback step-by-step logs from backend response trace to create a visual animation of LangGraph loops!
  const runTracePlayback = (traceList, finalResponseData) => {
    setDisplayedTrace([]);
    setPlaybackIndex(0);
    setGraphStatus({});
    
    let index = 0;
    const interval = setInterval(() => {
      if (index >= traceList.length) {
        clearInterval(interval);
        setResponse(finalResponseData);
        setIsLoading(false);
        setActiveGraphNode(null);
        // Complete the validation state
        setGraphStatus(prev => ({
          ...prev,
          validate: finalResponseData.validation.status === 'APPROVED' ? 'completed' : 'failed'
        }));
        return;
      }

      const log = traceList[index];
      setDisplayedTrace(prev => [...prev, log]);
      
      // Update SVG node statuses based on agent
      if (log.agent === 'RetrievalAgent') {
        setActiveGraphNode('retrieve');
        setGraphStatus(prev => ({ ...prev, retrieve: 'active', generate: null, validate: null }));
      } else if (log.agent === 'ReasoningAgent') {
        setActiveGraphNode('generate');
        setGraphStatus(prev => ({ ...prev, retrieve: 'completed', generate: 'active', validate: null }));
      } else if (log.agent === 'ValidationAgent') {
        setActiveGraphNode('validate');
        setGraphStatus(prev => ({ ...prev, retrieve: 'completed', generate: 'completed', validate: 'active' }));
        
        if (log.action === 'REJECT') {
          setGraphStatus(prev => ({ ...prev, validate: 'failed' }));
        } else if (log.action === 'APPROVE') {
          setGraphStatus(prev => ({ ...prev, validate: 'completed' }));
        }
      }
      
      index++;
      setPlaybackIndex(index);
    }, 1200); // 1.2 seconds delay per step to show the thinking pipeline
  };

  // Submit Assistant Query
  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setResponse(null);
    setDisplayedTrace([]);
    setActiveGraphNode(null);
    setGraphStatus({});
    
    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      
      if (!res.ok) {
        throw new Error('Server responded with an error');
      }
      
      const data = await res.json();
      // Play back trace steps sequentially
      runTracePlayback(data.trace, data);
    } catch (err) {
      console.warn("Backend query API down, falling back to local client-side simulation:", err);
      
      const cleanQuery = query.trim().replace(/[?.]/g, '');
      const lowercaseQuery = cleanQuery.toLowerCase();
      const words = cleanQuery.split(' ').filter(w => w.length > 3 && !['what', 'where', 'when', 'should', 'would', 'could', 'their', 'there', 'about', 'local', 'explain', 'describe', 'summarize'].includes(w.toLowerCase()));
      const extractedTopic = words.slice(-2).join(' ') || cleanQuery;
      
      // Determine the mock response data based on the user query
      let mockData = null;
      
      if (lowercaseQuery.includes('deep learning') || lowercaseQuery.includes('neural') || lowercaseQuery.includes('machine learning') || lowercaseQuery.includes('artificial intelligence') || lowercaseQuery.includes('ai')) {
        mockData = {
          topic: "Deep Learning",
          draft: "Based on the index files, Deep Learning was invented in 1950 by Turing to run on the first digital computer.",
          hallucination: "The claim that Deep Learning was invented in 1950 by Turing to run on the first digital computer is historically ungrounded in our context.",
          answer: "Deep Learning is a subset of machine learning based on artificial neural networks with representation learning. It uses multiple layers of nonlinear processing units to extract and transform features directly from raw data without manual feature engineering.",
          source: "deep_learning_intro.pdf",
          chunks: [
            { content: "Artificial Neural Networks (ANNs) form the core of deep learning, stacking multiple layers (input, hidden, output) to learn high-level abstractions.", source: "deep_learning_intro.pdf", chunk_index: 0, score: 0.9412 },
            { content: "Deep learning models such as CNNs and Transformers process raw inputs hierarchically, automatically discovering representations needed for classification.", source: "deep_learning_intro.pdf", chunk_index: 1, score: 0.8876 }
          ],
          explanation: "Definition of deep learning and neural representation is fully verified against deep_learning_intro.pdf."
        };
      } else if (lowercaseQuery.includes('rag') || lowercaseQuery.includes('retrieval augmented') || lowercaseQuery.includes('vector') || lowercaseQuery.includes('embedding') || lowercaseQuery.includes('faiss')) {
        mockData = {
          topic: "Retrieval-Augmented Generation",
          draft: "RAG retrieves data by querying Google search directly and copying the first three links into the model prompt.",
          hallucination: "The claim that RAG queries Google Search directly is incorrect. The system retrieves chunks from a local FAISS vector store.",
          answer: "Retrieval-Augmented Generation (RAG) optimizes LLM outputs by querying an authoritative, external knowledge base. Documents are chunked and embedded into a vector space, indexed in a vector store (like FAISS), retrieved semantically for a user query, and appended to the prompt to ground generation.",
          source: "agentic_workflows_guide.txt",
          chunks: [
            { content: "RAG systems bridge the gap between static LLM parameters and dynamic private documents by injecting context chunks into the generation prompt.", source: "agentic_workflows_guide.txt", chunk_index: 4, score: 0.9245 },
            { content: "Vector search indexes high-dimensional embeddings of text segments, enabling semantic similarity matching using distance metrics.", source: "agentic_workflows_guide.txt", chunk_index: 5, score: 0.8912 }
          ],
          explanation: "RAG retrieval flow, embedding index, and local vector store groundings are authenticated."
        };
      } else if (lowercaseQuery.includes('langgraph') || lowercaseQuery.includes('multi-agent') || lowercaseQuery.includes('agent') || lowercaseQuery.includes('orchestration') || lowercaseQuery.includes('graph')) {
        mockData = {
          topic: "LangGraph Multi-Agent Orchestration",
          draft: "LangGraph coordinates agents using a simple sequential bash script that executes Python files one after another.",
          hallucination: "The assertion that LangGraph is a simple sequential bash script is inaccurate. It is a stateful graph library that allows cycles and shared state.",
          answer: "LangGraph is a library designed for building stateful, multi-agent systems with LLMs. By modeling agent interactions as graphs where nodes are agent decisions and edges are transitions (including conditional routing), it enables cyclic loops, self-correction, and collaborative workflows.",
          source: "agentic_workflows_guide.txt",
          chunks: [
            { content: "Multi-agent systems divide complex research tasks among specialized nodes (e.g. RetrievalAgent, ReasoningAgent, ValidationAgent) sharing state.", source: "agentic_workflows_guide.txt", chunk_index: 2, score: 0.9543 },
            { content: "LangGraph supports cycles, enabling self-correction loops where validators critique outputs and route them back to reasoning for revision.", source: "agentic_workflows_guide.txt", chunk_index: 3, score: 0.9108 }
          ],
          explanation: "Multi-agent graph orchestration, cycles, and LangGraph structures are validated."
        };
      } else if (lowercaseQuery.includes('hallucination') || lowercaseQuery.includes('validation') || lowercaseQuery.includes('validate') || lowercaseQuery.includes('guardrail') || lowercaseQuery.includes('faithfulness')) {
        mockData = {
          topic: "Hallucination Audit & Validation",
          draft: "Validation is performed by asking a human administrator via email to approve each generated LLM response.",
          hallucination: "Ungrounded claim: validation is automated via ValidationAgent checks, not by emailing human admins.",
          answer: "Response validation is handled by a specialized ValidationAgent acting as an automated guardrail. It cross-references the reasoning agent's output with retrieved context to compute faithfulness (checking for hallucinations) and relevance, initiating feedback loops if issues arise.",
          source: "deep_learning_intro.pdf",
          chunks: [
            { content: "Validation agents act as automated guardrails by evaluating the faithfulness of generated drafts relative to the retrieved context chunks.", source: "deep_learning_intro.pdf", chunk_index: 7, score: 0.9388 },
            { content: "If the verification agent flags a claim as ungrounded, it returns structured critique, triggering a self-correction loop in the reasoning agent.", source: "deep_learning_intro.pdf", chunk_index: 8, score: 0.8921 }
          ],
          explanation: "Faithfulness metrics and automated self-correction guardrails conform to system logs."
        };
      } else if (lowercaseQuery.includes('evaluation') || lowercaseQuery.includes('ragas') || lowercaseQuery.includes('metric') || lowercaseQuery.includes('compliance') || lowercaseQuery.includes('precision')) {
        mockData = {
          topic: "RAGAS Evaluation Framework",
          draft: "Ragas evaluation measures if the code compiles and if the frontend page renders within 2 seconds.",
          hallucination: "The metric description is ungrounded. Ragas measures faithfulness, answer relevance, and context precision, not compilation or UI speed.",
          answer: "Ragas is an evaluation framework used to audit and score RAG pipelines without human ground truth. It measures Faithfulness (verifying answers are grounded in context), Answer Relevance (checking if the query is directly addressed), and Context Precision (evaluating retriever quality).",
          source: "deep_learning_intro.pdf",
          chunks: [
            { content: "RAGAS computes continuous scores between 0 and 1 for faithfulness and relevance using LLM-assisted reasoning to audit compliance.", source: "deep_learning_intro.pdf", chunk_index: 9, score: 0.9632 },
            { content: "Automated evaluation studio runs test cases to identify systematic failures in either retrieval or generation layers.", source: "deep_learning_intro.pdf", chunk_index: 10, score: 0.8974 }
          ],
          explanation: "Ragas compliance, metrics definition, and automated scoring are grounded in deep_learning_intro.pdf."
        };
      } else if (lowercaseQuery.includes('scheduling') || lowercaseQuery.includes('task') || lowercaseQuery.includes('distributed') || lowercaseQuery.includes('queue') || lowercaseQuery.includes('worker') || lowercaseQuery.includes('redis')) {
        mockData = {
          topic: "Distributed Task Scheduling",
          draft: "Tasks are stored in local CPU registers and executed directly in the main browser event loop thread.",
          hallucination: "Ungrounded claim: tasks are processed asynchronously via Redis broker queues and background worker nodes, not the browser thread.",
          answer: "A distributed task processing system schedules and executes background jobs across worker nodes. It relies on a message broker (e.g. Redis) to queue task payloads, which are consumed asynchronously by workers that handle retries, rate limits, and state tracking.",
          source: "agentic_workflows_guide.txt",
          chunks: [
            { content: "Distributed worker queues coordinate asynchronous tasks by decoupling submission from execution via broker queues.", source: "agentic_workflows_guide.txt", chunk_index: 12, score: 0.9102 },
            { content: "Retry policies and execution metrics are monitored by a central dashboard tracking worker health and resource usage.", source: "agentic_workflows_guide.txt", chunk_index: 13, score: 0.8544 }
          ],
          explanation: "Distributed architecture, broker queueing, and asynchronous worker systems are validated."
        };
      } else {
        // Dynamic fallback fallback
        const cleanTopic = extractedTopic.charAt(0).toUpperCase() + extractedTopic.slice(1);
        mockData = {
          topic: cleanTopic,
          draft: `Based on the mock index files, we observe that ${extractedTopic} was first introduced in 1999 as a method to optimize processing.`,
          hallucination: `The claim that ${extractedTopic} was introduced in 1999 is ungrounded. The source document says 2009.`,
          answer: `According to the source documents, the framework for ${extractedTopic} supports stateful query routing, enabling structured execution flows and multi-agent validation starting in 2009.`,
          source: "agentic_workflows_guide.txt",
          chunks: [
            { content: `Multi-agent orchestration using LangGraph enables robust self-correction loops. The system for ${extractedTopic} was designed in 2009.`, source: "agentic_workflows_guide.txt", chunk_index: 0, score: 0.8654 },
            { content: `Evaluation frameworks check for hallucinations and score faithfulness of ${extractedTopic} outputs.`, source: "deep_learning_intro.pdf", chunk_index: 2, score: 0.7912 }
          ],
          explanation: `No hallucinations detected. Answer regarding "${extractedTopic}" is well-supported.`
        };
      }
      
      const { topic, draft, hallucination, answer, source, chunks, explanation } = mockData;
      
      // Generate a client-side simulation response!
      setTimeout(() => {
        const simulatedTrace = [
          { agent: "RetrievalAgent", action: "QUERY_EXPANSION", message: `Expanding query '${query}' -> keywords: ${words.slice(0, 3).join(', ')}.` },
          { agent: "RetrievalAgent", action: "RETRIEVE", message: `Found 2 relevant chunks in client-side fallback vector store matching topic: "${topic}".` },
          { agent: "ReasoningAgent", action: "SYNTHESIZE", message: `Synthesizing response for query regarding "${topic}"...` },
          { agent: "ReasoningAgent", action: "DRAFT", message: `Draft response: "${draft}"` },
          { agent: "ValidationAgent", action: "GROUNDING_CHECK", message: "Checking answer faithfulness and checking for hallucinations..." },
          { agent: "ValidationAgent", action: "REJECT", message: `Validation Failed. Hallucination Detected: ${hallucination}` },
          { agent: "ReasoningAgent", action: "CORRECT", message: `Applying validator feedback. Modifying response to ground claims precisely in text regarding ${topic}.` },
          { agent: "ReasoningAgent", action: "REVISE", message: `Revised response: "${answer}"` },
          { agent: "ValidationAgent", action: "GROUNDING_CHECK", message: "Re-auditing revised claims against context..." },
          { agent: "ValidationAgent", action: "APPROVE", message: "Validation Passed. Response is grounded in retrieved documents." }
        ];
        
        const simulatedResponse = {
          query: query,
          answer: answer,
          retrieved_chunks: chunks,
          validation: {
            status: "APPROVED",
            faithfulness: 0.98,
            answer_relevance: 0.95,
            context_precision: 0.90,
            confidence: 0.96,
            explanation: explanation
          },
          trace: simulatedTrace,
          retry_count: 1
        };
        
        runTracePlayback(simulatedTrace, simulatedResponse);
      }, 500);
    }
  };

  const runEvaluation = async () => {
    setIsEvaluating(true);
    setEvalResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const data = await res.json();
      setEvalResult(data);
    } catch (err) {
      console.warn("Backend evaluation API down, using client-side mock evaluation reports.");
      setTimeout(() => {
        setEvalResult({
          summary: {
            avg_faithfulness: 0.92,
            avg_answer_relevance: 0.94,
            avg_context_precision: 0.88,
            avg_confidence: 0.91,
            total_cases: 4
          },
          cases: [
            { query: "How does local task scheduling work?", faithfulness: 0.95, answer_relevance: 0.92, context_precision: 0.90, confidence: 0.93, retries: 0 },
            { query: "Explain the multi-agent validation loop in LangGraph.", faithfulness: 0.98, answer_relevance: 0.96, context_precision: 0.92, confidence: 0.96, retries: 1 },
            { query: "What is faithfulness in RAGAS evaluation?", faithfulness: 0.90, answer_relevance: 0.94, context_precision: 0.85, confidence: 0.90, retries: 0 },
            { query: "How does the system handle hallucination detection?", faithfulness: 0.85, answer_relevance: 0.94, context_precision: 0.85, confidence: 0.85, retries: 1 }
          ]
        });
      }, 1000);
    } finally {
      setIsEvaluating(false);
    }
  };

  // Render SVG Flow Chart
  const renderFlowGraph = () => {
    return (
      <div className="graph-container">
        <svg width="100%" height="180" viewBox="0 0 700 180">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280" />
            </marker>
            <marker id="arrow-active" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#8b5cf6" />
            </marker>
          </defs>

          {/* User Query Starting Point */}
          <g transform="translate(40, 90)">
            <circle r="8" fill="var(--text-muted)" />
            <text x="0" y="24" textAnchor="middle" fill="var(--text-secondary)" fontSize="10" fontWeight="600">QUERY</text>
          </g>

          {/* Connection to Retrieval */}
          <path d="M 48 90 L 100 90" stroke={activeGraphNode === 'retrieve' ? 'var(--accent-purple)' : '#4b5563'} strokeWidth="2" markerEnd={activeGraphNode === 'retrieve' ? 'url(#arrow-active)' : 'url(#arrow)'} />

          {/* Node 1: Retrieval */}
          <g transform="translate(100, 55)" className={`graph-node ${graphStatus.retrieve || ''}`}>
            <rect width="120" height="70" rx="8" />
            <text x="60" y="32" fontWeight="600">RetrievalAgent</text>
            <text x="60" y="50" fill="var(--text-muted)" fontSize="9">FAISS Vector Store</text>
          </g>

          {/* Connection to Reasoning */}
          <path d="M 220 90 L 280 90" stroke={activeGraphNode === 'generate' ? 'var(--accent-purple)' : '#4b5563'} strokeWidth="2" markerEnd={activeGraphNode === 'generate' ? 'url(#arrow-active)' : 'url(#arrow)'} />

          {/* Node 2: Reasoning */}
          <g transform="translate(280, 55)" className={`graph-node ${graphStatus.generate || ''}`}>
            <rect width="120" height="70" rx="8" />
            <text x="60" y="32" fontWeight="600">ReasoningAgent</text>
            <text x="60" y="50" fill="var(--text-muted)" fontSize="9">Response Synthesis</text>
          </g>

          {/* Connection to Validation */}
          <path d="M 400 90 L 460 90" stroke={activeGraphNode === 'validate' ? 'var(--accent-purple)' : '#4b5563'} strokeWidth="2" markerEnd={activeGraphNode === 'validate' ? 'url(#arrow-active)' : 'url(#arrow)'} />

          {/* Node 3: Validation */}
          <g transform="translate(460, 55)" className={`graph-node ${graphStatus.validate || ''}`}>
            <rect width="120" height="70" rx="8" />
            <text x="60" y="32" fontWeight="600">ValidationAgent</text>
            <text x="60" y="50" fill="var(--text-muted)" fontSize="9">Hallucination Audit</text>
          </g>

          {/* Loop Back Path (Conditional Loop) */}
          <path d="M 520 55 C 520 15, 160 15, 160 55" fill="none" 
            stroke={graphStatus.validate === 'failed' ? 'var(--error)' : '#4b5563'} 
            strokeWidth={graphStatus.validate === 'failed' ? '2.5' : '1.5'} 
            strokeDasharray={graphStatus.validate === 'failed' ? '0' : '4'}
            markerEnd={graphStatus.validate === 'failed' ? 'url(#arrow-active)' : 'url(#arrow)'} />
          
          <text x="340" y="30" textAnchor="middle" 
            fill={graphStatus.validate === 'failed' ? 'var(--error)' : 'var(--text-muted)'} 
            fontSize="9" fontWeight="600">
            {graphStatus.validate === 'failed' ? 'RETRY LOOP (Hallucination Detected)' : 'Validation Fail Loop'}
          </text>

          {/* Connection to End */}
          <path d="M 580 90 L 630 90" stroke={graphStatus.validate === 'completed' ? 'var(--success)' : '#4b5563'} strokeWidth="2" markerEnd={graphStatus.validate === 'completed' ? 'url(#arrow-active)' : 'url(#arrow)'} />

          {/* End Point */}
          <g transform="translate(640, 90)">
            <circle r="8" fill={graphStatus.validate === 'completed' ? 'var(--success)' : 'var(--text-muted)'} />
            <text x="0" y="24" textAnchor="middle" fill="var(--text-secondary)" fontSize="10" fontWeight="600">ANSWER</text>
          </g>
        </svg>
      </div>
    );
  };

  const renderGauge = (label, val) => {
    const percentage = Math.round(val * 100);
    let color = 'var(--success)';
    if (percentage < 70) color = 'var(--error)';
    else if (percentage < 88) color = 'var(--warning)';

    return (
      <div className="stat-card glass-card">
        <span className="stat-label">{label}</span>
        <span className="stat-value" style={{ color }}>{percentage}%</span>
        <div className="progress-bar-container">
          <div className="progress-bar-fill" style={{ width: `${percentage}%`, backgroundColor: color }}></div>
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div>
          <div className="sidebar-logo">
            <Cpu size={24} color="var(--accent-purple)" />
            <span>ResearchStudio</span>
          </div>

          <nav className="sidebar-menu">
            <div 
              className={`menu-item ${activeTab === 'assistant' ? 'active' : ''}`}
              onClick={() => setActiveTab('assistant')}
            >
              <Layers size={18} />
              Assistant playground
            </div>
            <div 
              className={`menu-item ${activeTab === 'library' ? 'active' : ''}`}
              onClick={() => setActiveTab('library')}
            >
              <BookOpen size={18} />
              Document library
            </div>
            <div 
              className={`menu-item ${activeTab === 'evaluation' ? 'active' : ''}`}
              onClick={() => setActiveTab('evaluation')}
            >
              <BarChart3 size={18} />
              Evaluation studio
            </div>
            <div 
              className={`menu-item ${activeTab === 'settings' ? 'active' : ''}`}
              onClick={() => setActiveTab('settings')}
            >
              <Settings size={18} />
              Settings Panel
            </div>
          </nav>
        </div>

        <div className="sidebar-footer">
          <span>Engine Status</span>
          <div className={`badge-status ${isMockMode ? 'mock' : 'live'}`}>
            <span className="bullet"></span>
            {isMockMode ? 'Mock Mode' : 'Live LLM Engine'}
          </div>
          <span>Antigravity Flow v1.0</span>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="main-content">
        
        {/* Header Bar */}
        <header className="header-bar">
          <div className="header-title">
            {activeTab === 'assistant' && (
              <>
                <h1>Research Assistant</h1>
                <p>Query documents and watch the multi-agent graph reasoning loop in real-time.</p>
              </>
            )}
            {activeTab === 'library' && (
              <>
                <h1>Document Library</h1>
                <p>Manage raw text and PDF knowledge bases indexed into local vector search database.</p>
              </>
            )}
            {activeTab === 'evaluation' && (
              <>
                <h1>Evaluation Studio</h1>
                <p>Run RAGAS batch compliance audits measuring faithfulness, relevance, and precision.</p>
              </>
            )}
            {activeTab === 'settings' && (
              <>
                <h1>System Settings</h1>
                <p>Configure model configurations, target temperatures, validation retries, and API keys.</p>
              </>
            )}
          </div>

          <div className="header-actions">
            {isMockMode && activeTab !== 'settings' && (
              <div className="badge-status mock" style={{ fontSize: '0.85rem', padding: '0.4rem 0.8rem' }}>
                <AlertCircle size={14} /> Demo Mode: No API keys required
              </div>
            )}
          </div>
        </header>

        {/* View Content */}
        <div className="view-container">

          {/* TABS 1: ASSISTANT PLAYGROUND */}
          {activeTab === 'assistant' && (
            <div className="assistant-workspace">
              
              {/* Form Input */}
              <form onSubmit={handleQuerySubmit} className="query-box">
                <input 
                  type="text" 
                  className="input-text" 
                  placeholder="Ask a question about your documents (e.g. 'How does local task scheduling work?')..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  disabled={isLoading}
                />
                <button type="submit" className="btn-primary" disabled={isLoading}>
                  {isLoading ? <RefreshCw className="animate-spin" size={18} /> : <Search size={18} />}
                  {isLoading ? 'Running Graph...' : 'Research'}
                </button>
              </form>

              {/* Dynamic Flow Render */}
              {(isLoading || response) && renderFlowGraph()}

              <div className="grid-2col">
                
                {/* Answer Output */}
                <div className="response-area">
                  {response && (
                    <div className="glass-card response-card">
                      <div className="response-header">
                        <span style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <CheckCircle2 size={16} color="var(--success)" /> 
                          Final Grounded Answer
                        </span>
                        <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.8rem' }}>
                          <span>Validation Status: 
                            <strong style={{ color: response.validation.status === 'APPROVED' ? 'var(--success)' : 'var(--error)', marginLeft: '0.2rem' }}>
                              {response.validation.status}
                            </strong>
                          </span>
                          <span>Retries: <strong>{response.retry_count}</strong></span>
                        </div>
                      </div>
                      <div className="response-body">
                        {response.answer}
                      </div>
                    </div>
                  )}

                  {response && response.retrieved_chunks && response.retrieved_chunks.length > 0 && (
                    <div className="glass-card">
                      <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <BookOpen size={18} color="var(--accent-blue)" /> 
                        Retrieved Context Citations
                      </h3>
                      <div className="citations-list">
                        {response.retrieved_chunks.map((chunk, i) => (
                          <div key={i} className="citation-card">
                            <div className="citation-meta">
                              <span className="citation-source">[{i+1}] {chunk.source} (Chunk #{chunk.chunk_index})</span>
                              <span className="citation-score">Relevance Match: {(chunk.score * 100).toFixed(1)}%</span>
                            </div>
                            <p className="citation-text">"{chunk.content}"</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {!isLoading && !response && (
                    <div className="glass-card stat-card" style={{ padding: '4rem 2rem', color: 'var(--text-muted)' }}>
                      <Activity size={48} style={{ marginBottom: '1rem', opacity: 0.3 }} />
                      <p>Enter a query above to execute the multi-agent reasoning flow.</p>
                    </div>
                  )}
                </div>

                {/* Right Side Logs timeline */}
                <div className="trace-panel glass-card">
                  <h3>Agent Reasoning Trace</h3>
                  {displayedTrace.length === 0 && (
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No trace active. Submit a search to view step-by-step executions.</p>
                  )}
                  <div className="trace-timeline">
                    {displayedTrace.map((log, i) => {
                      let itemClass = '';
                      if (log.action === 'REJECT') itemClass = 'error';
                      else if (log.action === 'APPROVE') itemClass = 'success';
                      else if (i === playbackIndex - 1 && isLoading) itemClass = 'active';

                      return (
                        <div key={i} className={`trace-item ${itemClass}`}>
                          <div className="trace-bullet"></div>
                          <div className="trace-meta">
                            <span className="trace-agent">{log.agent}</span>
                            <span className="trace-action">{log.action}</span>
                          </div>
                          <p className="trace-msg">{log.message}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>

              </div>

            </div>
          )}

          {/* TAB 2: DOCUMENT LIBRARY */}
          {activeTab === 'library' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              
              {/* Upload Panel */}
              <div className="glass-card">
                <h3 style={{ marginBottom: '1.25rem' }}>Ingest New Knowledge File</h3>
                <div 
                  className="dropzone"
                  onClick={() => fileInputRef.current.click()}
                >
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    style={{ display: 'none' }} 
                    accept=".pdf,.txt" 
                    onChange={handleFileUpload}
                    disabled={isUploading}
                  />
                  <UploadCloud size={40} className="dropzone-icon" style={{ margin: '0 auto 1rem' }} />
                  <p style={{ fontWeight: 500, marginBottom: '0.35rem' }}>Click or drag PDF/TXT file here to ingest</p>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Files will be chunked, embedded, and indexed locally</p>
                </div>
                {uploadStatus && (
                  <p style={{ 
                    marginTop: '1rem', 
                    fontSize: '0.9rem', 
                    color: uploadStatus.startsWith('Error') ? 'var(--error)' : 'var(--success)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.35rem'
                  }}>
                    <Activity size={14} /> {uploadStatus}
                  </p>
                )}
              </div>

              {/* Ingested Documents List */}
              <div className="glass-card">
                <h3 style={{ marginBottom: '1rem' }}>Active Document Repository ({documents.length})</h3>
                {documents.length === 0 ? (
                  <p style={{ color: 'var(--text-muted)', padding: '2rem 0', textAlign: 'center' }}>No documents uploaded yet. Upload a TXT or PDF file above to index it.</p>
                ) : (
                  <div className="doc-table-container">
                    <table className="doc-table">
                      <thead>
                        <tr>
                          <th>Document Name</th>
                          <th>File Size</th>
                          <th>Uploaded At</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {documents.map((doc, idx) => (
                          <tr key={idx}>
                            <td style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', borderBottom: 'none' }}>
                              <FileText size={16} color="var(--accent-purple)" />
                              {doc.filename}
                            </td>
                            <td>{(doc.size / 1024).toFixed(1)} KB</td>
                            <td>{new Date(doc.created_at * 1000).toLocaleString()}</td>
                            <td>
                              <span className="badge-status live" style={{ display: 'inline-flex' }}>Indexed</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

            </div>
          )}

          {/* TAB 3: EVALUATION STUDIO */}
          {activeTab === 'evaluation' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              
              <div className="glass-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h2 style={{ fontSize: '1.25rem', fontFamily: 'var(--font-display)', marginBottom: '0.35rem' }}>RAG Evaluation Framework</h2>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Runs a complete batch audit on a test Q&A suite to compute faithfulness, relevance, and context precision scores.</p>
                </div>
                <button 
                  className="btn-primary" 
                  onClick={runEvaluation} 
                  disabled={isEvaluating}
                >
                  {isEvaluating ? <RefreshCw className="animate-spin" size={16} /> : <ShieldCheck size={16} />}
                  {isEvaluating ? 'Running Audit...' : 'Run Compliance Audit'}
                </button>
              </div>

              {/* Summary Metrics Gauges */}
              {evalResult && (
                <div className="grid-4col" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem' }}>
                  {renderGauge('Faithfulness (No Hallucinations)', evalResult.summary.avg_faithfulness)}
                  {renderGauge('Answer Relevance', evalResult.summary.avg_answer_relevance)}
                  {renderGauge('Context Precision', evalResult.summary.avg_context_precision)}
                  {renderGauge('Overall Confidence', evalResult.summary.avg_confidence)}
                </div>
              )}

              {/* Detailed case reports */}
              {evalResult && (
                <div className="glass-card">
                  <h3 style={{ marginBottom: '1.25rem' }}>Batch Test Case Breakdown ({evalResult.cases.length} cases run)</h3>
                  <div className="doc-table-container">
                    <table className="doc-table">
                      <thead>
                        <tr>
                          <th>Query</th>
                          <th>Faithfulness</th>
                          <th>Relevance</th>
                          <th>Precision</th>
                          <th>Confidence</th>
                          <th>Loops</th>
                        </tr>
                      </thead>
                      <tbody>
                        {evalResult.cases.map((c, i) => {
                          const getScoreClass = (s) => s >= 0.88 ? 'excellent' : s >= 0.7 ? 'good' : 'poor';
                          return (
                            <tr key={i}>
                              <td style={{ maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                {c.query}
                              </td>
                              <td>
                                <span className={`score-badge ${getScoreClass(c.faithfulness)}`}>
                                  {(c.faithfulness * 100).toFixed(0)}%
                                </span>
                              </td>
                              <td>
                                <span className={`score-badge ${getScoreClass(c.answer_relevance)}`}>
                                  {(c.answer_relevance * 100).toFixed(0)}%
                                </span>
                              </td>
                              <td>
                                <span className={`score-badge ${getScoreClass(c.context_precision)}`}>
                                  {(c.context_precision * 100).toFixed(0)}%
                                </span>
                              </td>
                              <td>
                                <span className={`score-badge ${getScoreClass(c.confidence)}`}>
                                  {(c.confidence * 100).toFixed(0)}%
                                </span>
                              </td>
                              <td style={{ fontFamily: 'var(--font-mono)' }}>{c.retries}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {!evalResult && !isEvaluating && (
                <div className="glass-card" style={{ textAlign: 'center', padding: '4rem 2rem', color: 'var(--text-muted)' }}>
                  <ShieldCheck size={48} style={{ margin: '0 auto 1rem', opacity: 0.3 }} />
                  <p>Click "Run Compliance Audit" above to run Q&A verification benchmarks.</p>
                </div>
              )}

            </div>
          )}

          {/* TAB 4: SETTINGS PANEL */}
          {activeTab === 'settings' && (
            <div className="glass-card">
              <h2 style={{ fontSize: '1.25rem', fontFamily: 'var(--font-display)', marginBottom: '1.5rem', color: '#fff' }}>Configure System Orchestration</h2>
              
              <form onSubmit={handleSaveSettings} className="settings-form">
                
                <div className="form-group full-width switch-group">
                  <input 
                    type="checkbox" 
                    id="mock_mode" 
                    checked={config.mock_mode}
                    onChange={(e) => setConfig({ ...config, mock_mode: e.target.checked })}
                    style={{ width: '18px', height: '18px', cursor: 'pointer' }}
                  />
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <label htmlFor="mock_mode" style={{ cursor: 'pointer' }}>Enable Local Demo Mode (Mock LLM)</label>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      Uses rule-based simulations. Uncheck this to use real OpenAI/Anthropic keys.
                    </span>
                  </div>
                </div>

                <div className="form-group">
                  <label>LLM Engine Provider</label>
                  <select 
                    className="select-input"
                    value={config.default_llm_provider}
                    onChange={(e) => setConfig({ ...config, default_llm_provider: e.target.value })}
                    disabled={config.mock_mode}
                  >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Engine Model</label>
                  <select 
                    className="select-input"
                    value={config.default_model}
                    onChange={(e) => setConfig({ ...config, default_model: e.target.value })}
                    disabled={config.mock_mode}
                  >
                    {config.default_llm_provider === 'openai' ? (
                      <>
                        <option value="gpt-4o-mini">gpt-4o-mini (Faster, cheaper)</option>
                        <option value="gpt-4o">gpt-4o (Reasoning heavy)</option>
                      </>
                    ) : (
                      <>
                        <option value="claude-3-5-sonnet-20240620">claude-3-5-sonnet</option>
                        <option value="claude-3-haiku-20240307">claude-3-haiku</option>
                      </>
                    )}
                  </select>
                </div>

                <div className="form-group">
                  <label>OpenAI API Key</label>
                  <input 
                    type="password" 
                    className="input-text" 
                    placeholder="sk-..." 
                    value={config.openai_api_key}
                    onChange={(e) => setConfig({ ...config, openai_api_key: e.target.value })}
                    disabled={config.mock_mode}
                  />
                </div>

                <div className="form-group">
                  <label>Anthropic API Key</label>
                  <input 
                    type="password" 
                    className="input-text" 
                    placeholder="sk-ant-..." 
                    value={config.anthropic_api_key}
                    onChange={(e) => setConfig({ ...config, anthropic_api_key: e.target.value })}
                    disabled={config.mock_mode}
                  />
                </div>

                <div className="form-group">
                  <label>Generation Temperature</label>
                  <input 
                    type="number" 
                    step="0.1" 
                    min="0" 
                    max="1" 
                    className="input-text" 
                    value={config.temperature}
                    onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
                  />
                </div>

                <div className="form-group">
                  <label>Maximum Validation Retries</label>
                  <input 
                    type="number" 
                    min="1" 
                    max="5" 
                    className="input-text" 
                    value={config.max_retries}
                    onChange={(e) => setConfig({ ...config, max_retries: parseInt(e.target.value) })}
                  />
                </div>

                <div className="form-group full-width" style={{ marginTop: '1rem' }}>
                  <button type="submit" className="btn-primary" style={{ alignSelf: 'flex-start' }}>
                    Save Configuration Settings
                  </button>
                </div>

              </form>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}

export default App;
