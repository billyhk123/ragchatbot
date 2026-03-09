import os
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.chain import build_chain, get_retriever
from app.memory import ChatMemory
from app.pathway import main as pathway_main
from app.tracing import Trace, recent_traces
import app.prompts as prompts
import app.telegram_bot as tg

SERVICE_URL = os.environ.get("SERVICE_URL", "")


@asynccontextmanager
async def lifespan(application: FastAPI):
    threading.Thread(target=pathway_main, name="pathway", daemon=True).start()
    tg.set_rag_answer(rag_answer)
    await tg.init()
    if SERVICE_URL:
        await tg.register_webhook(SERVICE_URL)
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
    """Run the full RAG pipeline: recall memory, retrieve context, invoke LLM, store turn."""
    trace = Trace(user_id, user_text)
    try:
        recall = memory.build_context(user_id, user_text)
        memory_context = memory.format_context(recall)

        trace.record_memory(
            summary=recall.summary,
            recent=recall.recent,
            relevant=recall.relevant,
            formatted=memory_context,
        )

        answer = chain.invoke({
            "question": user_text,
            "memory": memory_context,
            "trace": trace,
        })
        memory.update_after_turn(user_id, user_text, answer)
        trace.finish(answer)
        return answer
    except Exception as exc:
        trace.finish("", error=str(exc))
        raise


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


@app.get("/reload-prompts")
def reload_prompts():
    """Hot-reload prompts + params from GCS / local file and rebuild the chain."""
    global chain
    cfg = prompts.reload()
    chain = build_chain()
    return {
        "status": "reloaded",
        "source": prompts._load_source,
        "error": prompts._load_error,
        "rag": cfg["rag"],
        "summary": cfg["summary"],
        "llm": cfg["llm"],
    }


# ── Retrieval test ────────────────────────────────────────────────────

@app.get("/retrieval-test")
def retrieval_test(q: str, k: int = 4):
    """Return raw retrieval results for a query — no LLM involved."""
    import time
    retriever = get_retriever()
    if retriever is None:
        return {"error": "Retriever not initialised yet"}

    t0 = time.monotonic()
    try:
        docs = retriever.invoke(q)
    except Exception as exc:
        return {"error": str(exc), "documents": []}
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    results = []
    for i, d in enumerate(docs[:k]):
        results.append({
            "rank": i + 1,
            "source": d.metadata.get("source", "unknown"),
            "metadata": {k: v for k, v in d.metadata.items() if k != "source"},
            "content": d.page_content,
            "length": len(d.page_content),
        })
    return {
        "query": q,
        "k": k,
        "returned": len(results),
        "elapsed_ms": elapsed_ms,
        "documents": results,
    }


@app.get("/retrieval-test-ui", response_class=HTMLResponse)
def retrieval_test_ui():
    ui_path = Path(__file__).with_name("retrieval_test.html")
    html = ui_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ── Traces ───────────────────────────────────────────────────────────

@app.get("/traces")
def get_traces(user_id: str | None = None, limit: int = 20):
    """Return recent pipeline traces from Firestore."""
    rows = recent_traces(user_id=user_id, limit=min(limit, 100))
    for r in rows:
        if "timestamp" in r and hasattr(r["timestamp"], "isoformat"):
            r["timestamp"] = r["timestamp"].isoformat()
    return {"count": len(rows), "traces": rows}


# ── Telegram ────────────────────────────────────────────────────────

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    payload = await request.json()
    await tg.process_update(payload)
    return {"ok": True}

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
