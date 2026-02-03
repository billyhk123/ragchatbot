from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
            "question": RunnablePassthrough(),
            "context": retriever | format_docs,
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
