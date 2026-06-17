import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from backend.core.config import settings
from backend.core.mock_llm import MockEmbeddings

def get_embeddings_model():
    """Returns the appropriate embeddings model based on Mock Mode settings."""
    if settings.MOCK_MODE:
        return MockEmbeddings()
    else:
        # Fallback to OpenAI if api_key is present, otherwise use mock
        if settings.OPENAI_API_KEY:
            return OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        else:
            print("[!] Warning: OPENAI_API_KEY not found. Defaulting to MockEmbeddings.")
            return MockEmbeddings()

def extract_text_from_file(file_path: str) -> str:
    """Extracts raw text from a text or PDF file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path.lower())
    
    if ext == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def ingest_and_index_file(file_path: str, filename: str) -> dict:
    """
    Reads a file, chunks it, generates embeddings, and saves it into FAISS.
    Returns metadata about the ingested file.
    """
    # 1. Extract text
    text = extract_text_from_file(file_path)
    if not text.strip():
        raise ValueError(f"Extracted text from {filename} is empty.")

    # 2. Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Format chunks as LangChain documents with metadata
    from langchain_core.documents import Document
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": filename, "chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]
    
    # 3. Embed and store in FAISS
    embeddings = get_embeddings_model()
    vector_store_dir = settings.VECTOR_STORE_PATH
    
    if os.path.exists(os.path.join(vector_store_dir, "index.faiss")):
        # Load existing index, merge
        try:
            db = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(documents)
            db.save_local(vector_store_dir)
        except Exception as e:
            # If load fails (e.g. dimensions mismatch), overwrite
            print(f"[!] FAISS load failed, overwriting vector store: {e}")
            db = FAISS.from_documents(documents, embeddings)
            db.save_local(vector_store_dir)
    else:
        # Create new index
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(vector_store_dir)

    return {
        "filename": filename,
        "chunks": len(documents),
        "char_count": len(text)
    }

def retrieve_context(query: str, k: int = 4) -> list[dict]:
    """Retrieves relevant document chunks from the local FAISS database."""
    embeddings = get_embeddings_model()
    vector_store_dir = settings.VECTOR_STORE_PATH
    
    if not os.path.exists(os.path.join(vector_store_dir, "index.faiss")):
        return []

    try:
        db = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)
        docs_and_scores = db.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, distance in docs_and_scores:
            # Convert L2 distance to a pseudo-similarity score in [0, 1] range
            score = 1.0 / (1.0 + float(distance))
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "score": score
            })
        return results
    except Exception as e:
        print(f"[!] Error retrieving context: {e}")
        return []
