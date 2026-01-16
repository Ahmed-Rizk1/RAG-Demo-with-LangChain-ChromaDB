import argparse
import sys
import os

# Add src to path if needed (though running as module is better)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import load_documents
from src.data.splitter import split_documents
from src.retrieval.vector_store import create_vector_store
from src.ui.streamlit_app import run_streamlit_app

def ingest():
    """Runs the data ingestion pipeline."""
    print("Starting ingestion...")
    docs = load_documents()
    if not docs:
        print("No documents found in data/raw/")
        return
    
    print(f"Loaded {len(docs)} documents.")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    
    create_vector_store(chunks, reset=True)
    print("Ingestion complete.")

def run_app():
    """Runs the Streamlit app."""
    # This function is called by the streamlit runner, 
    # but if we run this script directly with python, we need to spawn streamlit.
    # However, standard way is `streamlit run main.py`.
    # To support args with streamlit is tricky.
    
    # If this script is run by streamlit, `run_streamlit_app` is called.
    run_streamlit_app()

if __name__ == "__main__":
    # Check if run via streamlit
    # Streamlit sets the script name in sys.argv[0] usually or we check for streamlit env.
    try:
        import streamlit.web.bootstrap
        is_streamlit = True
    except ImportError:
        is_streamlit = False

    # Simple logic: If arguments provided, behave as CLI. If not (or if run by streamlit), run app.
    # But `streamlit run main.py` passes args to the script too.
    
    parser = argparse.ArgumentParser(description="RAG System Entry Point")
    parser.add_argument("--ingest", action="store_true", help="Run data ingestion")
    
    # If running under streamlit, arguments might be weird, so we use parse_known_args
    args, unknown = parser.parse_known_args()
    
    if args.ingest:
        ingest()
    else:
        # If run directly with python (not streamlit), we can tell user to run streamlit
        # OR we can use subprocess to call streamlit.
        # But commonly `streamlit run main.py` is the command.
        
        # We can detect if we are inside streamlit execution loop by checking `st.runtime.exists()`
        try:
             import streamlit as st
             if st.runtime.exists():
                 run_app()
             else:
                 # Not in streamlit, maybe user ran `python main.py`
                 print("To run the app, use: streamlit run main.py")
                 print("To ingest data, use: python main.py --ingest")
        except:
             print("To run the app, use: streamlit run main.py")
