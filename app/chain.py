from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, PathwayVectorClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from app.settings import settings
import app.prompts as prompts
from app.llm_poe import PoeChatModel


def format_docs(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        lines.append(f"[{src}]\n{d.page_content}")
    return "\n\n---\n\n".join(lines)


def _build_retriever(k: int = 4):
    if settings.pathway_url or settings.pathway_host:
        client = PathwayVectorClient(
            url=settings.pathway_url or None,
            host=settings.pathway_host or None,
            port=settings.pathway_port or None,
        )
        return client.as_retriever(search_kwargs={"k": k})

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
            "context": RunnableLambda(lambda x: x["question"]) | _retriever | format_docs,
        }
        | prompts.RAG_PROMPT
        | _llm
        | StrOutputParser()
    )
    return chain
