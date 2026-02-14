from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer using ONLY the provided context. "
     "If the context does not contain the answer, say you don't know and suggest what to ask next. "
     "Always provide citations like [source]."),
    ("human",
     "Question:\n{question}\n\n"
     "Conversation memory:\n{memory}\n\n"
     "Retrieved context:\n{context}\n\n"
     "Answer with citations:"),
])
