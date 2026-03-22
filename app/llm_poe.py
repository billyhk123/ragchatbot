"""LLM access via Poe's OpenAI-compatible endpoint using LangChain's ChatOpenAI."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.settings import settings

_POE_BASE_URL = "https://api.poe.com/v1"

_llm: ChatOpenAI | None = None


def get_llm(**overrides) -> ChatOpenAI:
    """Return a shared ChatOpenAI instance pointed at Poe's API.

    Accepts keyword overrides (temperature, max_tokens, etc.) for
    one-off instances; without overrides returns the shared singleton.
    """
    global _llm
    if overrides:
        return ChatOpenAI(
            model=overrides.get("model", settings.poe_bot_name),
            api_key=settings.poe_api_key,
            base_url=_POE_BASE_URL,
            temperature=overrides.get("temperature", settings.temperature),
            max_tokens=overrides.get("max_tokens", settings.max_tokens),
        )
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.poe_bot_name,
            api_key=settings.poe_api_key,
            base_url=_POE_BASE_URL,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    return _llm
