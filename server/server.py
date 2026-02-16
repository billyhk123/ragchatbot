import os
import requests
from pathlib import Path

from fastapi import FastAPI, Request, Response, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.chain import build_chain
from app.memory import ChatMemory

app = FastAPI()

# Build once at startup (important for performance on Cloud Run)
chain = build_chain()
memory = ChatMemory()

VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
GRAPH_URL = os.environ.get("WHATSAPP_GRAPH_URL", "https://graph.facebook.com/v21.0")


class ChatRequest(BaseModel):
    question: str
    user_id: str | None = None


class ChatResponse(BaseModel):
    answer: str


def send_whatsapp_text(to: str, text: str) -> None:
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        raise RuntimeError("Missing WHATSAPP_ACCESS_TOKEN or WHATSAPP_PHONE_NUMBER_ID")

    url = f"{GRAPH_URL}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text[:3500]},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"WhatsApp send failed: {r.status_code} {r.text}")


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

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer = rag_answer(request.question, request.user_id or "web")
    return ChatResponse(answer=answer)


@app.get("/webhook")
def verify_webhook(
    hub_mode: str = Query("", alias="hub.mode"),
    hub_verify_token: str = Query("", alias="hub.verify_token"),
    hub_challenge: str = Query("", alias="hub.challenge"),
):
    # Meta sends query params: hub.mode, hub.verify_token, hub.challenge
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return Response(content=hub_challenge, media_type="text/plain")
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    data = await request.json()

    try:
        entries = data.get("entry") or []
        if not entries:
            return {"status": "ignored"}

        handled = 0
        for entry in entries:
            for change in entry.get("changes") or []:
                value = change.get("value") or {}
                messages = value.get("messages") or []

                # Often you'll receive status updates (delivered/read) with no "messages"
                if not messages:
                    continue

                for msg in messages:
                    if msg.get("type") != "text":
                        continue

                    from_number = msg.get("from")
                    text = (msg.get("text") or {}).get("body", "")

                    if not from_number or not text:
                        continue

                    answer = rag_answer(text, from_number)
                    send_whatsapp_text(from_number, answer)
                    handled += 1

        if handled == 0:
            return {"status": "ignored"}
        return {"status": "ok"}

    except Exception as e:
        # Cloud Run logs will contain stacktrace if you add logging later
        raise HTTPException(status_code=500, detail=f"Webhook error: {type(e).__name__}")
