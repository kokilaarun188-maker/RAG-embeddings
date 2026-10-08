from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents(directory_path: str):
    """
    Load PDF documents from the specified directory.
    Returns a list of Document objects with text and metadata.
    """
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    return documents
