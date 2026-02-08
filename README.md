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

## Local development

1. Create a `.env` with required secrets:
   - `POE_API_KEY`
   - `WHATSAPP_VERIFY_TOKEN`
   - `WHATSAPP_ACCESS_TOKEN`
   - `WHATSAPP_PHONE_NUMBER_ID`

2. Build the vector DB (if you have data in `data/raw/`):
   ```bash
   python -m ingest.ingest
   ```

3. Run the API:
   ```bash
   uvicorn server.server:app --reload
   ```

## Deploy to Cloud Run

Build and deploy using the provided `Dockerfile`. The container will generate a
vector DB during build if `data/raw/` exists and is non-empty.

Set environment variables in Cloud Run for all required secrets.
