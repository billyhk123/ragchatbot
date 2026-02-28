import logging

import requests as _requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from app.settings import settings
import app.prompts as prompts
from app.llm_poe import PoeChatModel

logger = logging.getLogger(__name__)

_TRUNC = 500


def format_docs(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        lines.append(f"[{src}]\n{d.page_content}")
    return "\n\n---\n\n".join(lines)


def _log_retriever_input(question: str) -> str:
    logger.info("[Chain:1-retriever] query=%s", question[:_TRUNC])
    return question


def _log_retrieved_context(context: str) -> str:
    logger.info("[Chain:2-context] length=%d preview=%s", len(context), context[:_TRUNC])
    return context


def _log_prompt(prompt_value):
    for msg in prompt_value.to_messages():
        logger.info("[Chain:3-prompt] role=%s content=%s", msg.type, msg.content[:_TRUNC])
    return prompt_value


def _log_answer(answer: str) -> str:
    logger.info("[Chain:4-answer] length=%d preview=%s", len(answer), answer[:_TRUNC])
    return answer


def _build_retriever(k: int = 4):
    if settings.pathway_url or settings.pathway_host:
        base = settings.pathway_url or f"http://{settings.pathway_host}:{settings.pathway_port}"
        def _retrieve(query: str) -> list:
            resp = _requests.post(
                f"{base}/v1/retrieve", json={"query": query, "k": k}, timeout=30,
            )
            resp.raise_for_status()
            return [
                Document(page_content=r.get("text", ""), metadata=r.get("metadata", {}))
                for r in sorted(resp.json(), key=lambda x: x["dist"])
            ]
        return RunnableLambda(_retrieve)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    db = FAISS.load_local(
        settings.persist_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return db.as_retriever(search_kwargs={"k": k})


_retriever = None
_llm = None
_last_k = None


def build_chain():
    global _retriever, _llm, _last_k

    cfg = prompts._cfg
    k = int(cfg.get("rag", {}).get("k", 4))
    temperature = float(cfg.get("llm", {}).get("temperature", settings.temperature))
    max_tokens = int(cfg.get("llm", {}).get("max_tokens", settings.max_tokens))

    if _retriever is None or _last_k != k:
        _retriever = _build_retriever(k)
        _last_k = k

    if _llm is None or _llm.temperature != temperature or _llm.max_tokens != max_tokens:
        _llm = PoeChatModel(temperature=temperature, max_tokens=max_tokens)

    chain = (
        {
            "question": RunnableLambda(lambda x: x["question"]),
            "memory": RunnableLambda(lambda x: x.get("memory", "")),
            "context": (
                RunnableLambda(lambda x: x["question"])
                | RunnableLambda(_log_retriever_input)
                | _retriever
                | format_docs
                | RunnableLambda(_log_retrieved_context)
            ),
        }
        | prompts.RAG_PROMPT
        | RunnableLambda(_log_prompt)
        | _llm
        | StrOutputParser()
        | RunnableLambda(_log_answer)
    )
    return chain
