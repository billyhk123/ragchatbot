# RAG Chatbot with WhatsApp Webhook

This project is a retrieval-augmented generation (RAG) chatbot that can answer
questions using a local document knowledge base. It exposes a FastAPI webhook
that integrates with the WhatsApp Cloud API and can also be used as a simple
HTTP service.

## Features

- RAG pipeline built with LangChain + FAISS vector store
- Document ingestion from `data/raw/`
- WhatsApp webhook endpoints (`/webhook`) for inbound messages
- Cloud Run ready via Docker

