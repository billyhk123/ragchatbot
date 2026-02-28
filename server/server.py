import os
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.chain import build_chain
from app.memory import ChatMemory
from app.pathway import main as pathway_main
import app.prompts as prompts
import app.telegram_bot as tg

logger = logging.getLogger(__name__)

SERVICE_URL = os.environ.get("SERVICE_URL", "")


@asynccontextmanager
async def lifespan(application: FastAPI):
    threading.Thread(target=pathway_main, name="pathway", daemon=True).start()
    logger.info("[Server] Pathway thread started")

    tg.set_rag_answer(rag_answer)
    await tg.init()
    if SERVICE_URL:
        try:
            await tg.register_webhook(SERVICE_URL)
        except Exception:
            logger.exception("[TG] Failed to register webhook on startup")
    yield
    await tg.shutdown()


app = FastAPI(lifespan=lifespan)

chain = build_chain()
memory = ChatMemory()


class ChatRequest(BaseModel):
    question: str
    user_id: str | None = None


class ChatResponse(BaseModel):
    answer: str


def rag_answer(user_text: str, user_id: str) -> str:
    logger.info("[RAG] user=%s question=%s", user_id, user_text[:200])
    recall = memory.build_context(user_id, user_text)
    memory_context = memory.format_context(recall)
    logger.info("[RAG] memory_context length=%d preview=%s", len(memory_context), memory_context[:300])
    answer = chain.invoke({"question": user_text, "memory": memory_context})
    logger.info("[RAG] answer length=%d preview=%s", len(answer), answer[:300])
    memory.update_after_turn(user_id, user_text, answer)
    return answer


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/privacy", response_class=HTMLResponse)
def privacy_policy():
    policy_path = Path(__file__).with_name("privacy.html")
    html = policy_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/chat-ui", response_class=HTMLResponse)
def chat_ui():
    ui_path = Path(__file__).with_name("chat_ui.html")
    html = ui_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/data-deletion", response_class=HTMLResponse)
def data_deletion():
    policy_path = Path(__file__).with_name("data_deletion.html")
    html = policy_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer = rag_answer(request.question, request.user_id or "web")
    return ChatResponse(answer=answer)


@app.post("/reload-prompts")
def reload_prompts():
    """Hot-reload prompts + params from GCS / local file and rebuild the chain."""
    global chain
    cfg = prompts.reload()
    chain = build_chain()
    return {
        "status": "reloaded",
        "rag_k": cfg["rag"].get("k"),
        "llm_temperature": cfg["llm"].get("temperature"),
        "llm_max_tokens": cfg["llm"].get("max_tokens"),
        "rag_system_preview": cfg["rag"]["system"][:120] + "...",
    }


# ── Telegram ────────────────────────────────────────────────────────

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    payload = await request.json()
    await tg.process_update(payload)
    return {"ok": True}


@app.post("/telegram/set-webhook")
async def telegram_set_webhook(request: Request):
    """Manually register the Telegram webhook. Body: {"url": "https://..."}"""
    body = await request.json()
    base_url = body.get("url", "").rstrip("/")
    if not base_url:
        return {"error": "provide 'url' in the request body"}
    result = await tg.register_webhook(base_url)
    return result


@app.get("/telegram/status")
async def telegram_status():
    try:
        tg_app = tg.get_application()
        info = await tg_app.bot.get_webhook_info()
        return {
            "webhook_url": info.url,
            "pending_update_count": info.pending_update_count,
            "last_error": info.last_error_message or None,
        }
    except Exception as e:
        return {"error": str(e)}
