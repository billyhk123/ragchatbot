from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_documents(data_dir: str):
    data_path = Path(data_dir)
    docs = []

    for p in data_path.rglob("*"):
        if p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower() in [".md", ".txt"]:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())

    return docs
