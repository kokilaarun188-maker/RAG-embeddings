def retrieve_context(query: str, vector_store, top_k: int = 3):
    """
    Retrieve relevant context for a given query from the vector store.
    Returns the raw Document objects so metadata can be extracted for citations.
    """
    return vector_store.similarity_search(query, top_k=top_k)
