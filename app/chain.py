import time

import requests as _requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from app.settings import settings
import app.prompts as prompts
from app.llm_poe import get_llm
from app.crypto import LANGCHAIN_TOOLS, execute_tool as execute_crypto_tool

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


def get_retriever():
    """Return the current retriever instance (built lazily by build_chain)."""
    return _retriever


def build_chain():
    """Build (or rebuild) the RAG chain with tool-calling support.

    Returns a LangChain-compatible Runnable that accepts
    ``{"question": str, "memory": str, "trace": Trace | None}``
    and returns a string answer.
    """
    global _retriever, _last_k

    cfg = prompts._cfg
    k = int(cfg.get("rag", {}).get("k", 4))

    if _retriever is None or _last_k != k:
        _retriever = _build_retriever(k)
        _last_k = k

    def _invoke(inputs: dict) -> str:
        question = inputs["question"]
        memory = inputs.get("memory", "")
        trace = inputs.get("trace")

        live_cfg = prompts._cfg
        temperature = float(live_cfg.get("llm", {}).get("temperature", settings.temperature))
        max_tokens = int(live_cfg.get("llm", {}).get("max_tokens", settings.max_tokens))

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

        if trace:
            trace.record_retrieval(
                query=question,
                documents=[
                    {"source": d.metadata.get("source", "unknown"),
                     "content": d.page_content}
                    for d in docs
                ],
                context_length=len(context),
                duration_ms=ret_ms,
                error=ret_error,
            )

        # --- 2. Build messages ---
        system_text = live_cfg["rag"]["system"]
        human_text = live_cfg["rag"]["human"].format(
            question=question, memory=memory, context=context,
        )
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=human_text),
        ]

        # --- 3. LLM call with tool loop ---
        llm = get_llm(temperature=temperature, max_tokens=max_tokens)
        llm_with_tools = llm.bind_tools(LANGCHAIN_TOOLS, tool_choice="auto")

        llm_t0 = time.monotonic()
        rounds = 0
        for _round in range(_MAX_TOOL_ROUNDS):
            rounds = _round + 1
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                answer = ai_msg.content or ""
                llm_ms = int((time.monotonic() - llm_t0) * 1000)
                usage = _extract_usage(ai_msg)
                if trace:
                    trace.record_llm(
                        model=settings.poe_bot_name,
                        messages=[
                            {"role": "system", "content": system_text},
                            {"role": "user", "content": human_text},
                        ],
                        temperature=temperature,
                        response=answer,
                        rounds=rounds,
                        duration_ms=llm_ms,
                        usage=usage,
                    )
                return answer

            for tc in ai_msg.tool_calls:
                tool_t0 = time.monotonic()
                result = execute_crypto_tool(tc["name"], tc["args"])
                tool_ms = int((time.monotonic() - tool_t0) * 1000)
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                if trace:
                    trace.record_tool(
                        name=tc["name"],
                        arguments=tc["args"],
                        result=result,
                        duration_ms=tool_ms,
                    )

        answer = ai_msg.content or ""
        llm_ms = int((time.monotonic() - llm_t0) * 1000)
        usage = _extract_usage(ai_msg)
        if trace:
            trace.record_llm(
                model=settings.poe_bot_name,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": human_text},
                ],
                temperature=temperature,
                response=answer,
                rounds=rounds,
                duration_ms=llm_ms,
                usage=usage,
            )
        return answer

    return RunnableLambda(_invoke)


def _extract_usage(ai_msg) -> dict | None:
    """Pull token counts from LangChain's AIMessage.usage_metadata."""
    meta = getattr(ai_msg, "usage_metadata", None)
    if not meta:
        return None
    return {
        "prompt_tokens": meta.get("input_tokens"),
        "completion_tokens": meta.get("output_tokens"),
        "total_tokens": meta.get("total_tokens"),
    }
