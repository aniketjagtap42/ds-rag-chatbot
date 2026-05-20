from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    openai_api_key: str = 'dummy'
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = 'all-MiniLM-L6-v2'
    llm_model: str = 'llama-3.3-70b-versatile'
    top_k: int = 4
    chroma_dir: str = './chroma_db'
    data_path: str = './data/DS_Complete_Notes.docx'

    class Config:
        env_file = '.env'

settings = Settings()
