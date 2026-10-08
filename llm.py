from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0):
    """
    Initialize the Language Model.
    Requires OPENAI_API_KEY environment variable.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set.")
    return ChatOpenAI(model_name=model_name, temperature=temperature)

def generate_answer(query: str, retrieved_docs, llm):
    """
    Generate an answer using a Large Language Model based strictly on the provided context.
    """
    template = """Answer the question based ONLY on the following context.
If you cannot answer the question based on the context, say "I cannot answer this based on the provided knowledge sources." Do not guess or make up information.

Context:
{context}

Question:
{question}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Format context
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Generate response
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": query})
    
    return response.content
