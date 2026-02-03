from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    poe_api_key: str = os.environ["POE_API_KEY"]
    poe_bot_name: str = os.getenv("POE_BOT_NAME", "GPT-3.5-Turbo")

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_dir: str = os.getenv("PERSIST_DIR", "vectordb")

    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "700"))

settings = Settings()
