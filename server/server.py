import os
import requests
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.chain import build_chain
from app.memory import ChatMemory
import app.prompts as prompts

app = FastAPI()

chain = build_chain()
memory = ChatMemory()


WHATSAPP_BRIDGE_URL = os.environ.get("WHATSAPP_BRIDGE_URL", "http://localhost:3001")


class ChatRequest(BaseModel):
    question: str
    user_id: str | None = None


class ChatResponse(BaseModel):
    answer: str


def send_whatsapp_text(to: str, text: str) -> None:
    """Send a message via the whatsapp_bridge service."""
    url = f"{WHATSAPP_BRIDGE_URL}/send"
    r = requests.post(url, json={"to": to, "text": text[:3500]}, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"WhatsApp bridge send failed: {r.status_code} {r.text}")


def rag_answer(user_text: str, user_id: str) -> str:
    # Keep this wrapper: WhatsApp/channel code should never know about Poe/OpenAI/etc.
    recall = memory.build_context(user_id, user_text)
    memory_context = memory.format_context(recall)
    answer = chain.invoke({"question": user_text, "memory": memory_context})
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


@app.get("/whatsapp/status")
def whatsapp_status():
    """Proxy the bridge's connection status."""
    try:
        r = requests.get(f"{WHATSAPP_BRIDGE_URL}/status", timeout=5)
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Bridge unreachable: {e}")
