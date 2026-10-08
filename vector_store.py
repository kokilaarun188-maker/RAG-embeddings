from langchain_community.vectorstores import Chroma

class VectorStore:
    def __init__(self, persist_directory: str, embedding_model):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        # Initialize the Chroma DB
        self.db = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embedding_model
        )

    def add_documents(self, chunks):
        """
        Add document chunks to the vector store and persist them to disk.
        """
        self.db.add_documents(chunks)
        self.db.persist()

    def similarity_search(self, query: str, top_k: int = 5):
        """
        Search for the most similar document chunks to the given query.
        Returns a list of Document objects.
        """
        return self.db.similarity_search(query, k=top_k)
    
    def get_retriever(self, top_k: int = 5):
        """
        Returns a LangChain retriever interface for this vector store.
        """
        return self.db.as_retriever(search_kwargs={"k": top_k})
