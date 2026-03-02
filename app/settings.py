from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    poe_api_key: str = os.getenv("POE_API_KEY", "")
    poe_bot_name: str = os.getenv("POE_BOT_NAME", "telegrambot4taye")

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_dir: str = os.getenv("PERSIST_DIR", "vectordb")

    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "700"))

    firebase_project_id: str = os.getenv("FIREBASE_PROJECT_ID", "")
    firestore_prefix: str = os.getenv("FIRESTORE_PREFIX", "")

    memory_window_turns: int = int(os.getenv("MEMORY_WINDOW_TURNS", "6"))
    memory_recall_k: int = int(os.getenv("MEMORY_RECALL_K", "4"))
    memory_vector_limit: int = int(os.getenv("MEMORY_VECTOR_LIMIT", "200"))

    pathway_url: str = os.getenv("PATHWAY_URL", "")
    pathway_host: str = os.getenv("PATHWAY_HOST", "")
    pathway_port: int = int(os.getenv("PATHWAY_PORT", "8081"))

    tracing_enabled: bool = os.getenv("TRACING_ENABLED", "true").lower() == "true"
    tracing_ttl_days: int = int(os.getenv("TRACING_TTL_DAYS", "30"))

settings = Settings()
