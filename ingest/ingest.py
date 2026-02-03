from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.settings import settings
from ingest.loaders import load_documents
from ingest.splitters import make_splitter

def main():
    docs = load_documents("data/raw")
    splitter = make_splitter()
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(settings.persist_dir)

    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks.")
    print(f"Saved vector DB to: {settings.persist_dir}")

if __name__ == "__main__":
    main()
