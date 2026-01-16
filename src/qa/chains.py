from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.config import settings
from src.qa.prompts import get_contextualize_q_prompt, get_qa_prompt
from src.retrieval.vector_store import get_retriever

def create_rag_chain():
    """
    Creates the RAG chain including history awareness and document stuffing.
    """
    llm = ChatOpenAI(model=settings.MODEL_NAME)
    retriever = get_retriever()
    
    # History aware retriever
    contextualize_q_prompt = get_contextualize_q_prompt()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Question Answer chain
    qa_prompt = get_qa_prompt()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Final RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain
