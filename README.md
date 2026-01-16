# Modular RAG Chat with LangChain

A professional, modular implementation of a Retrieval Augmented Generation (RAG) system using LangChain, ChromaDB, and OpenAI.

## Features

-   **Modular Architecture**: Clean separation of data loading, vector storage, and application logic.
-   **Multi-format Support**: Ingests `.txt` and `.pdf` documents.
-   **Persistent Vector Store**: Uses ChromaDB to save embeddings.
-   **Configurable**: Settings management via `config/settings.yaml` and `.env`.
-   **Streamlit UI**: User-friendly chat interface with history.

## Project Structure

```text
rag_project/
├── config/
│   └── settings.yaml       # Configuration
├── data/
│   ├── raw/                # Input documents
│   └── processed/          # Vector DB (Chroma)
├── src/
│   ├── data/               # Loader & Splitter
│   ├── embeddings/         # Embedding logic
│   ├── retrieval/          # Vector Store logic
│   ├── qa/                 # Generic Chains & Prompts
│   └── ui/                 # Streamlit App
├── main.py                 # Entry point
└── requirements.txt
```

## Setup & Installation

1.  **Clone the repository** (if not already local).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment**:
    -   Copy `.env.example` to `.env`.
    -   Add your OpenAI API Key:
        ```text
        OPENAI_API_KEY=sk-...
        ```

## Usage

### 1. Ingest Data
Place your documents (PDFs, text files) in `data/raw/`.
Run the ingestion pipeline:

```bash
python main.py --ingest
```

### 2. Run the App
Launch the Streamlit interface:

```bash
streamlit run main.py
```

## Configuration
You can adjust model parameters (Model Name, Chunk Size, etc.) in `config/settings.yaml`.

## Key Technologies
-   [LangChain](https://langchain.com/)
-   [ChromaDB](https://www.trychroma.com/)
-   [Streamlit](https://streamlit.io/)
-   [OpenAI](https://openai.com/)
