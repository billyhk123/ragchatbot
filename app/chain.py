from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, PathwayVectorClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from app.settings import settings
from app.prompts import RAG_PROMPT
from app.llm_poe import PoeChatModel


def format_docs(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        # If PDF loader adds page numbers, you can also include them:
        # page = d.metadata.get("page", None)
        # src = f"{src}#page={page}" if page is not None else src
        lines.append(f"[{src}]\n{d.page_content}")
    return "\n\n---\n\n".join(lines)


def build_chain():
    if settings.pathway_url or settings.pathway_host:
        client = PathwayVectorClient(
            url=settings.pathway_url or None,
            host=settings.pathway_host or None,
            port=settings.pathway_port or None,
        )
        retriever = client.as_retriever(search_kwargs={"k": 4})
    else:
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        db = FAISS.load_local(
            settings.persist_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = PoeChatModel()

    chain = (
        {
            "question": RunnableLambda(lambda x: x["question"]),
            "memory": RunnableLambda(lambda x: x.get("memory", "")),
            "context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
