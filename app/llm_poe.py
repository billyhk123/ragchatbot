"""LLM wrappers using Poe's OpenAI-compatible API."""

from __future__ import annotations

from typing import Any, List, Optional

import openai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.settings import settings

_client: openai.OpenAI | None = None


def get_client() -> openai.OpenAI:
    """Return a shared OpenAI client pointed at Poe's API."""
    global _client
    if _client is None:
        _client = openai.OpenAI(
            api_key=settings.poe_api_key,
            base_url="https://api.poe.com/v1",
        )
    return _client


def _to_openai_messages(messages: List[BaseMessage]) -> list[dict]:
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "user"
        else:
            role = "assistant"
        out.append({"role": role, "content": m.content})
    return out


class PoeChatModel(BaseChatModel):
    """LangChain ChatModel wrapper using Poe's OpenAI-compatible API."""

    bot_name: str = settings.poe_bot_name
    api_key: str = settings.poe_api_key

    temperature: float = settings.temperature
    max_tokens: int = settings.max_tokens

    @property
    def _llm_type(self) -> str:
        return "poe-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = get_client()
        resp = client.chat.completions.create(
            model=self.bot_name,
            messages=_to_openai_messages(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = resp.choices[0].message.content or ""

        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]

        gen = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[gen])
