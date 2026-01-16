import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from src.qa.chains import create_rag_chain

def run_streamlit_app():
    st.set_page_config(page_title="RAG Chat", layout="wide")
    st.title("📚 RAG Chat with LangChain")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if "rag_chain" not in st.session_state:
         try:
            st.session_state.rag_chain = create_rag_chain()
         except Exception as e:
            st.error(f"Failed to initialize RAG chain: {e}. Please ensure you have run ingestion.")
            st.session_state.rag_chain = None

    # Display chat history first
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
             with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, SystemMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # Chat input and processing
    user_input = st.chat_input("Ask a question")

    if user_input:
        if not st.session_state.rag_chain:
            st.error("RAG chain is not initialized.")
            return

        # Display user message immediately (optional, or rely on rerun)
        # But to feel responsive we display it, and then the spinner.
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                answer = result["answer"]
                
                # Update history
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(SystemMessage(content=answer))
                
                # Display assistant message
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
            except Exception as e:
                st.error(f"Error generating response: {e}")
            
            # Note: We don't force a rerun here because the next interaction will trigger it.
            # However, if we want the history loop to handle rendering next time, this is fine.
