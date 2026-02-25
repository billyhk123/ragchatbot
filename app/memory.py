from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import faiss
import firebase_admin
import numpy as np
from firebase_admin import firestore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from app.llm_poe import PoeChatModel
from app.prompts import SUMMARY_PROMPT
from app.settings import settings


def _now():
    return datetime.now(timezone.utc)


def _init_firestore():
    if not firebase_admin._apps:
        if settings.firebase_project_id:
            firebase_admin.initialize_app(
                options={"projectId": settings.firebase_project_id}
            )
        else:
            firebase_admin.initialize_app()
    return firestore.client()


def _format_messages(messages: Iterable[dict]) -> str:
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {m.get('content', '')}")
    return "\n".join(lines).strip()


@dataclass
class MemoryRecall:
    summary: str
    recent: List[dict]
    relevant: List[dict]


class ChatMemory:
    def __init__(self):
        self.client = _init_firestore()
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.llm = PoeChatModel()
        self.window_turns = settings.memory_window_turns
        self.recall_k = settings.memory_recall_k
        self.vector_limit = settings.memory_vector_limit
        self.prefix = settings.firestore_prefix

    def _sessions(self):
        return self.client.collection(f"{self.prefix}chat_sessions")

    def _messages(self):
        return self.client.collection(f"{self.prefix}chat_messages")

    def _vectors(self):
        return self.client.collection(f"{self.prefix}chat_memory_vectors")

    def get_summary(self, user_id: str) -> str:
        doc = self._sessions().document(user_id).get()
        if not doc.exists:
            return ""
        return (doc.to_dict() or {}).get("summary", "") or ""

    def set_summary(self, user_id: str, summary: str) -> None:
        self._sessions().document(user_id).set(
            {"summary": summary, "updated_at": _now()},
            merge=True,
        )

    def add_message(self, user_id: str, role: str, content: str) -> None:
        self._messages().add(
            {
                "user_id": user_id,
                "role": role,
                "content": content,
                "summarized": False,
                "ts": _now(),
            }
        )

    def _recent_messages(self, user_id: str, limit: int) -> List[dict]:
        query = (
            self._messages()
            .where("user_id", "==", user_id)
            .where("summarized", "==", False)
            .order_by("ts", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        docs = [d.to_dict() for d in query.stream()]
        return list(reversed(docs))

    def _unsummarized_messages(self, user_id: str) -> List[firestore.DocumentSnapshot]:
        query = (
            self._messages()
            .where("user_id", "==", user_id)
            .where("summarized", "==", False)
            .order_by("ts", direction=firestore.Query.ASCENDING)
        )
        return list(query.stream())

    def _add_vectors(self, user_id: str, messages: List[dict]) -> None:
        if not messages:
            return
        texts = [
            f"{m.get('role', 'user')}: {m.get('content', '')}".strip() for m in messages
        ]
        vectors = self.embeddings.embed_documents(texts)
        batch = self.client.batch()
        for text, vec in zip(texts, vectors):
            doc = self._vectors().document()
            batch.set(
                doc,
                {
                    "user_id": user_id,
                    "content": text,
                    "embedding": vec,
                    "ts": _now(),
                },
            )
        batch.commit()

    def _search_vectors(self, user_id: str, query: str) -> List[dict]:
        if self.recall_k <= 0:
            return []
        query_ref = (
            self._vectors()
            .where("user_id", "==", user_id)
            .order_by("ts", direction=firestore.Query.DESCENDING)
            .limit(self.vector_limit)
        )
        docs = [d.to_dict() for d in query_ref.stream()]
        if not docs:
            return []

        embeddings = np.array([d["embedding"] for d in docs], dtype="float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        q = np.array([self.embeddings.embed_query(query)], dtype="float32")
        faiss.normalize_L2(q)
        scores, indices = index.search(q, min(self.recall_k, len(docs)))

        results = []
        for idx in indices[0]:
            if idx < 0:
                continue
            results.append({"content": docs[idx]["content"]})
        return results

    def _summarize(self, summary: str, lines: str) -> str:
        prompt = SUMMARY_PROMPT.format_prompt(summary=summary, lines=lines)
        result = self.llm._generate(prompt.to_messages())
        return result.generations[0].message.content.strip()

    def update_after_turn(self, user_id: str, user_text: str, assistant_text: str) -> None:
        self.add_message(user_id, "user", user_text)
        self.add_message(user_id, "assistant", assistant_text)

        unsummarized = self._unsummarized_messages(user_id)
        window_messages = self.window_turns * 2
        if len(unsummarized) <= window_messages:
            return

        cutoff = len(unsummarized) - window_messages
        to_summarize_docs = unsummarized[:cutoff]
        to_summarize = [d.to_dict() for d in to_summarize_docs]

        current_summary = self.get_summary(user_id)
        lines = _format_messages(to_summarize)
        updated_summary = self._summarize(current_summary, lines)
        self.set_summary(user_id, updated_summary)

        self._add_vectors(user_id, to_summarize)

        batch = self.client.batch()
        for doc in to_summarize_docs:
            batch.update(doc.reference, {"summarized": True})
        batch.commit()

    def build_context(self, user_id: str, query: str) -> MemoryRecall:
        summary = self.get_summary(user_id)
        recent = self._recent_messages(user_id, self.window_turns * 2)
        relevant = self._search_vectors(user_id, query)
        return MemoryRecall(summary=summary, recent=recent, relevant=relevant)

    def format_context(self, memory: MemoryRecall) -> str:
        parts: List[str] = []
        if memory.summary:
            parts.append(f"Summary:\n{memory.summary}")
        if memory.recent:
            parts.append(f"Recent conversation:\n{_format_messages(memory.recent)}")
        if memory.relevant:
            relevant_lines = "\n".join(
                f"- {m.get('content', '')}" for m in memory.relevant
            ).strip()
            if relevant_lines:
                parts.append(f"Relevant past messages:\n{relevant_lines}")
        return "\n\n".join(parts).strip()
