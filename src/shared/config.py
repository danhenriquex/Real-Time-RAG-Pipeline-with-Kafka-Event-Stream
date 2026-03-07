from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    kafka_bootstrap_servers: str = "kafka:9092"
    kafka_topic_documents: str = "documents"
    kafka_consumer_group: str = "embedding-workers"
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "documents"
    postgres_user: str = "rag"
    postgres_password: str = "ragpass"
    postgres_db: str = "rag_pipeline"
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 5
    upload_service_url: str = "http://upload-service:8001"
    rag_api_url: str = "http://rag-api:8000"
    log_level: str = "INFO"


settings = Settings()
