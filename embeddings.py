from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Returns a local HuggingFace embedding model.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
