from langchain_text_splitters import RecursiveCharacterTextSplitter

def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
