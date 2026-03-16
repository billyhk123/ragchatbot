# RAG Chatbot with Telegram Bot

This project is a retrieval-augmented generation (RAG) chatbot that can answer
questions using a local document knowledge base. It exposes a FastAPI service
with a Telegram bot integration and can also be used as a simple HTTP service.

## Features

- RAG pipeline built with LangChain + FAISS vector store
- Document ingestion from `data/raw/`
- Telegram bot with webhook endpoint (`/telegram/webhook`)
- Web chat UI at `/chat-ui`
- Cloud Run ready via Docker

