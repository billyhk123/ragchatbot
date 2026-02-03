from __future__ import annotations

from typing import Any, List, Optional

import fastapi_poe as fp
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.settings import settings


def _to_poe_messages(messages: List[BaseMessage]) -> List[fp.ProtocolMessage]:
    poe_messages: List[fp.ProtocolMessage] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "user"
        else:
            # LangChain has AIMessage and others; Poe mainly expects user/system.
            # We can map AI messages to "assistant" if supported by ProtocolMessage;
            # if not, you can skip AI history or include it as user text.
            role = "assistant"
        poe_messages.append(fp.ProtocolMessage(role=role, content=m.content))
    return poe_messages


class PoeChatModel(BaseChatModel):
    """Minimal LangChain ChatModel wrapper around fastapi_poe."""

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
        poe_messages = _to_poe_messages(messages)

        text_parts: List[str] = []
        for partial in fp.get_bot_response_sync(
            messages=poe_messages,
            bot_name=self.bot_name,
            api_key=self.api_key,
        ):
            if partial.text:
                text_parts.append(partial.text)

        text = "".join(text_parts)

        # Apply stop tokens crudely if needed
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]

        gen = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[gen])
