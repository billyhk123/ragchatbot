from __future__ import annotations

import os
import yaml
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

GCS_PROMPTS_BUCKET = os.environ.get("GCS_PROMPTS_BUCKET", "")
GCS_PROMPTS_FILE = os.environ.get("GCS_PROMPTS_FILE", "prompts.yaml")
LOCAL_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts.yaml"

_DEFAULTS = {
    "rag": {
        "k": 4,
        "system": (
            "You are a helpful assistant. "
            "Use the provided context to answer the user's question accurately. "
            "If the retrieved context is clearly irrelevant to the question, ignore it "
            "and respond naturally based on general knowledge or conversation memory. "
            "When you do use retrieved context, provide citations like [source]. "
            "For casual greetings or small talk, respond in a friendly, natural way."
        ),
        "human": (
            "Question:\n{question}\n\n"
            "Conversation memory:\n{memory}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Answer:"
        ),
    },
    "summary": {
        "system": (
            "You summarize a conversation for long-term memory. "
            "Produce a concise summary of factual details, preferences, and decisions. "
            "Avoid repeating transient greetings or filler. Keep it short."
        ),
        "human": (
            "Existing summary:\n{summary}\n\n"
            "New conversation lines:\n{lines}\n\n"
            "Updated summary:"
        ),
    },
    "llm": {
        "temperature": 0.2,
        "max_tokens": 700,
    },
}


_load_source: str = "defaults"
_load_error: str | None = None


def _load_from_gcs() -> dict | None:
    global _load_source, _load_error
    if not GCS_PROMPTS_BUCKET:
        return None
    try:
        from google.cloud.storage import Client as GCSClient

        client = GCSClient()
        blob = client.bucket(GCS_PROMPTS_BUCKET).blob(GCS_PROMPTS_FILE)
        if not blob.exists():
            _load_error = f"GCS file gs://{GCS_PROMPTS_BUCKET}/{GCS_PROMPTS_FILE} not found"
            return None
        raw = blob.download_as_text()
        data = yaml.safe_load(raw)
        _load_source = f"gcs: gs://{GCS_PROMPTS_BUCKET}/{GCS_PROMPTS_FILE}"
        _load_error = None
        return data
    except Exception as e:
        _load_error = f"GCS load failed: {e}"
        return None


def _load_from_local() -> dict | None:
    global _load_source, _load_error
    if not LOCAL_PROMPTS_PATH.exists():
        return None
    try:
        data = yaml.safe_load(LOCAL_PROMPTS_PATH.read_text(encoding="utf-8"))
        _load_source = f"local: {LOCAL_PROMPTS_PATH}"
        _load_error = None
        return data
    except Exception as e:
        _load_error = f"Local load failed: {e}"
        return None


def _build(data: dict | None) -> dict:
    global _load_source
    cfg = {k: dict(v) for k, v in _DEFAULTS.items()}
    if data:
        for key in ("rag", "summary", "llm"):
            if key in data and isinstance(data[key], dict):
                cfg[key] = {**cfg[key], **data[key]}
    else:
        _load_source = "defaults"
    return cfg


def load_prompts() -> dict:
    """Try GCS first, then local file, then hardcoded defaults."""
    data = _load_from_gcs()
    if data is None:
        data = _load_from_local()
    return _build(data)


def build_rag_prompt(cfg: dict) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", cfg["rag"]["system"]),
        ("human", cfg["rag"]["human"]),
    ])


def build_summary_prompt(cfg: dict) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", cfg["summary"]["system"]),
        ("human", cfg["summary"]["human"]),
    ])


_cfg = load_prompts()
RAG_PROMPT = build_rag_prompt(_cfg)
SUMMARY_PROMPT = build_summary_prompt(_cfg)


def reload() -> dict:
    """Reload prompts from GCS/local at runtime.

    Returns the loaded config dict.  After calling this, inspect
    ``_load_source`` and ``_load_error`` for diagnostics.
    """
    global _cfg, RAG_PROMPT, SUMMARY_PROMPT
    _cfg = load_prompts()
    RAG_PROMPT = build_rag_prompt(_cfg)
    SUMMARY_PROMPT = build_summary_prompt(_cfg)
    return _cfg
