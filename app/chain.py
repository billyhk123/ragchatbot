import json
import time

import requests as _requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.settings import settings
import app.prompts as prompts
from app.llm_poe import get_client
from app.crypto import TOOLS as CRYPTO_TOOLS, execute_tool as execute_crypto_tool

_MAX_TOOL_ROUNDS = 3


def format_docs(docs):
    """Format retrieved Documents into a single string with source citations."""
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        lines.append(f"[{src}]\n{d.page_content}")
    return "\n\n---\n\n".join(lines)


def _build_retriever(k: int = 4):
    """Create a retriever: direct HTTP to Pathway if configured, else local FAISS."""
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
_last_k = None


def build_chain():
    """Build (or rebuild) the RAG chain with tool-calling support.

    Returns a LangChain-compatible Runnable that accepts
    ``{"question": str, "memory": str, "trace": Trace | None}``
    and returns a string answer.
    """
    global _retriever, _last_k

    cfg = prompts._cfg
    k = int(cfg.get("rag", {}).get("k", 4))
    temperature = float(cfg.get("llm", {}).get("temperature", settings.temperature))
    max_tokens = int(cfg.get("llm", {}).get("max_tokens", settings.max_tokens))

    if _retriever is None or _last_k != k:
        _retriever = _build_retriever(k)
        _last_k = k

    def _invoke(inputs: dict) -> str:
        question = inputs["question"]
        memory = inputs.get("memory", "")
        trace = inputs.get("trace")

        # --- 1. Retrieve context (non-fatal) ---
        t0 = time.monotonic()
        ret_error = None
        docs = []
        try:
            docs = _retriever.invoke(question)
            context = format_docs(docs)
        except Exception as exc:
            context = ""
            ret_error = str(exc)
        ret_ms = int((time.monotonic() - t0) * 1000)

        sources = [d.metadata.get("source", "unknown") for d in docs]
        if trace:
            trace.record_retrieval(
                query=question,
                doc_count=len(docs),
                context_length=len(context),
                sources=sources,
                duration_ms=ret_ms,
                error=ret_error,
            )

        # --- 2. Build messages ---
        system_text = cfg["rag"]["system"]
        human_text = cfg["rag"]["human"].format(
            question=question, memory=memory, context=context,
        )
        messages: list[dict] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": human_text},
        ]

        # --- 3. LLM call with tool loop ---
        client = get_client()
        llm_t0 = time.monotonic()
        rounds = 0
        for _round in range(_MAX_TOOL_ROUNDS):
            rounds = _round + 1
            resp = client.chat.completions.create(
                model=settings.poe_bot_name,
                messages=messages,
                tools=CRYPTO_TOOLS,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                answer = msg.content or ""
                llm_ms = int((time.monotonic() - llm_t0) * 1000)
                if trace:
                    trace.record_llm(
                        model=settings.poe_bot_name,
                        messages_sent=len(messages),
                        temperature=temperature,
                        answer_length=len(answer),
                        rounds=rounds,
                        duration_ms=llm_ms,
                    )
                return answer

            messages.append(msg)
            for tc in msg.tool_calls:
                tool_t0 = time.monotonic()
                args = json.loads(tc.function.arguments)
                result = execute_crypto_tool(tc.function.name, args)
                tool_ms = int((time.monotonic() - tool_t0) * 1000)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                if trace:
                    trace.record_tool(
                        name=tc.function.name,
                        arguments=args,
                        result=result,
                        duration_ms=tool_ms,
                    )

        answer = msg.content or ""
        llm_ms = int((time.monotonic() - llm_t0) * 1000)
        if trace:
            trace.record_llm(
                model=settings.poe_bot_name,
                messages_sent=len(messages),
                temperature=temperature,
                answer_length=len(answer),
                rounds=rounds,
                duration_ms=llm_ms,
            )
        return answer

    return RunnableLambda(_invoke)
