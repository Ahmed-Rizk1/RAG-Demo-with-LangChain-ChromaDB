import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "RAG Demo"
    VERSION: str = "1.0.0"
    
    # OpenAI Config
    OPENAI_API_KEY: str
    MODEL_NAME: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Vector DB Config
    PERSIST_DIRECTORY: str = "data/processed/chroma_db"
    COLLECTION_NAME: str = "rag_collection"
    
    # Data Config
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
