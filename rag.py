import os
from document_processor import load_documents
from chunking import chunk_text
from embeddings import get_embedding_model
from vector_store import VectorStore
from retriever import retrieve_context
from llm import get_llm, generate_answer

DB_DIR = "data/vector_store"
DOCS_DIR = "documents"

def setup_rag_pipeline(force_rebuild: bool = False):
    """
    Initialize and populate the RAG pipeline with documents.
    """
    print("Initializing embedding model...")
    embedding_model = get_embedding_model()
    
    print("Connecting to Vector Store...")
    vs = VectorStore(persist_directory=DB_DIR, embedding_model=embedding_model)
    
    # Check if we need to ingest documents
    if force_rebuild or not os.path.exists(DB_DIR) or len(os.listdir(DB_DIR)) == 0:
        print(f"Loading documents from {DOCS_DIR}...")
        docs = load_documents(DOCS_DIR)
        
        if not docs:
            print(f"No documents found in {DOCS_DIR}. Please add PDFs.")
            return vs

        print(f"Loaded {len(docs)} document pages. Chunking text...")
        chunks = chunk_text(docs)
        
        print(f"Generated {len(chunks)} chunks. Adding to Vector Store...")
        vs.add_documents(chunks)
        print("Vector Store populated and persisted.")
    else:
        print("Vector Store already exists. Skipping ingestion.")
        
    return vs

def ask_question(query: str, vector_store, llm):
    """
    Process a user query through the RAG pipeline to generate an answer with evidence.
    """
    print(f"\n[?] Question: {query}")
    
    # 1. Retrieve
    print("[*] Retrieving relevant context...")
    docs = retrieve_context(query, vector_store, top_k=3)
    
    if not docs:
        print("[!] No relevant context found.")
        return
        
    # 2. Generate
    print("[*] Generating answer...")
    answer = generate_answer(query, docs, llm)
    
    # 3. Output Answer and Sources
    print("\n" + "="*50)
    print("🤖 ANSWER:")
    print("="*50)
    print(answer)
    print("\n" + "="*50)
    print("📚 SUPPORTING EVIDENCE:")
    print("="*50)
    
    # Deduplicate sources based on filename and page
    sources = set()
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        sources.add(f"- {source} (Page {page})")
        
    for s in sorted(sources):
        print(s)
    print("="*50 + "\n")

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set or the LLM will fail
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: Please set your OPENAI_API_KEY environment variable.")
        print("Example (Windows): $env:OPENAI_API_KEY='your-key-here'")
    
    vs = setup_rag_pipeline(force_rebuild=True)
    llm = get_llm()
    
    while True:
        try:
            user_query = input("\nEnter your question (or 'quit' to exit): ")
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            if user_query.strip():
                ask_question(user_query, vs, llm)
        except KeyboardInterrupt:
            break
