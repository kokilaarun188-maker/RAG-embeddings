from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split a list of Document objects into smaller chunks for embedding and retrieval.
    Metadata from the original documents is preserved in the chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
