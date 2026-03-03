"""Structured request tracing backed by Firestore.

Each RAG pipeline execution is recorded as a single document in the
``rag_traces`` collection.  Writes happen in a background thread so
they never add latency to the user response.

Full payloads (memory text, retrieved docs, LLM messages, token counts)
are captured so every pipeline step can be inspected after the fact.
A per-field truncation limit keeps each Firestore document under 1 MB.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import firebase_admin
from firebase_admin import firestore

from app.settings import settings

logger = logging.getLogger(__name__)

_COLLECTION = f"{settings.firestore_prefix}rag_traces"
_MAX_STR = 3000


def _init_firestore():
    if not firebase_admin._apps:
        if settings.firebase_project_id:
            firebase_admin.initialize_app(
                options={"projectId": settings.firebase_project_id}
            )
        else:
            firebase_admin.initialize_app()
    return firestore.client()


_db = None


def _get_db():
    global _db
    if _db is None:
        _db = _init_firestore()
    return _db


def _now():
    return datetime.now(timezone.utc)


def _trunc(text: str, limit: int = _MAX_STR) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated, total {len(text)} chars]"


def _trunc_messages(messages: list, limit: int = _MAX_STR) -> list[dict]:
    """Serialize a message list, truncating each content field."""
    out: list[dict] = []
    for m in messages:
        if isinstance(m, dict):
            entry = {k: v for k, v in m.items()}
            if "content" in entry and isinstance(entry["content"], str):
                entry["content"] = _trunc(entry["content"], limit)
        else:
            entry: dict[str, Any] = {
                "role": getattr(m, "role", "unknown"),
                "content": _trunc(getattr(m, "content", "") or "", limit),
            }
            if hasattr(m, "tool_calls") and m.tool_calls:
                entry["tool_calls"] = [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in m.tool_calls
                ]
        out.append(entry)
    return out


class Trace:
    """Collects timing and payload data for one RAG pipeline execution."""

    def __init__(self, user_id: str, question: str):
        self.trace_id = uuid.uuid4().hex[:16]
        self.user_id = user_id
        self.question = question
        self.timestamp = _now()
        self._start = time.monotonic()

        self.memory_data: dict[str, Any] = {}
        self.retrieval_data: dict[str, Any] = {}
        self.llm_data: dict[str, Any] = {}
        self.tool_calls: list[dict[str, Any]] = []
        self.answer: str = ""
        self.status: str = "ok"
        self.error: str | None = None

    # ── Memory ──────────────────────────────────────────────────────

    def record_memory(
        self,
        summary: str,
        recent: list[dict],
        relevant: list[dict],
        formatted: str,
    ) -> None:
        self.memory_data = {
            "summary": _trunc(summary),
            "recent": [
                {"role": m.get("role", "?"), "content": _trunc(m.get("content", ""))}
                for m in recent
            ],
            "relevant": [
                {"content": _trunc(m.get("content", ""))} for m in relevant
            ],
            "formatted_length": len(formatted),
            "summary_length": len(summary),
            "recent_count": len(recent),
            "relevant_count": len(relevant),
        }

    # ── Retrieval ───────────────────────────────────────────────────

    def record_retrieval(
        self,
        query: str,
        documents: list[dict],
        context_length: int,
        duration_ms: int,
        error: str | None = None,
    ) -> None:
        self.retrieval_data = {
            "query": query,
            "doc_count": len(documents),
            "documents": [
                {
                    "source": d.get("source", "unknown"),
                    "content": _trunc(d.get("content", "")),
                }
                for d in documents
            ],
            "context_length": context_length,
            "duration_ms": duration_ms,
            "error": error,
        }

    # ── LLM ─────────────────────────────────────────────────────────

    def record_llm(
        self,
        model: str,
        messages: list,
        temperature: float,
        response: str,
        rounds: int,
        duration_ms: int,
        usage: dict | None = None,
    ) -> None:
        self.llm_data = {
            "model": model,
            "temperature": temperature,
            "messages_sent": len(messages),
            "messages": _trunc_messages(messages),
            "response": _trunc(response),
            "answer_length": len(response),
            "word_count": len(response.split()),
            "rounds": rounds,
            "duration_ms": duration_ms,
            "usage": usage,
        }

    # ── Tools ───────────────────────────────────────────────────────

    def record_tool(
        self,
        name: str,
        arguments: dict,
        result: str,
        duration_ms: int,
    ) -> None:
        self.tool_calls.append({
            "name": name,
            "arguments": arguments,
            "result": _trunc(result),
            "duration_ms": duration_ms,
        })

    # ── Finish & write ──────────────────────────────────────────────

    def finish(self, answer: str, error: str | None = None) -> None:
        self.answer = answer
        self.status = "error" if error else "ok"
        self.error = error
        duration_ms = int((time.monotonic() - self._start) * 1000)

        if not settings.tracing_enabled:
            return

        doc = {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "question": self.question,
            "answer": _trunc(self.answer),
            "timestamp": self.timestamp,
            "duration_ms": duration_ms,
            "status": self.status,
            "error": self.error,
            "memory": self.memory_data,
            "retrieval": self.retrieval_data,
            "llm": self.llm_data,
            "tool_calls": self.tool_calls,
        }

        threading.Thread(
            target=self._write, args=(doc,), name="trace-write", daemon=True
        ).start()

    @staticmethod
    def _write(doc: dict) -> None:
        try:
            db = _get_db()
            db.collection(_COLLECTION).document(doc["trace_id"]).set(doc)
            logger.debug("[Trace] written %s", doc["trace_id"])
        except Exception:
            logger.exception("[Trace] failed to write %s", doc.get("trace_id"))


def recent_traces(user_id: str | None = None, limit: int = 20) -> list[dict]:
    """Fetch recent traces, optionally filtered by user_id."""
    db = _get_db()
    ref = db.collection(_COLLECTION)
    if user_id:
        ref = ref.where("user_id", "==", user_id)
    query = ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)
    return [doc.to_dict() for doc in query.stream()]
