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

## Pathway retriever service (Option A)

You can run Pathway as a separate retriever service backed by GCS. This keeps
your chatbot service lightweight and allows live document updates without
rebuilding the chatbot image.

### 1) Create the GCS bucket

```bash
gsutil mb -l asia-east2 gs://ragchatbot-raw
```

### 2) Upload raw documents

```bash
gsutil -m rsync -r ./data/raw gs://ragchatbot-raw
```

### 3) Build and deploy Pathway service

```bash
docker build -f pathway_service/Dockerfile -t gcr.io/$PROJECT_ID/pathway-retriever .
docker push gcr.io/$PROJECT_ID/pathway-retriever

gcloud run deploy pathway-retriever \
  --image gcr.io/$PROJECT_ID/pathway-retriever \
  --region asia-east2 \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET=ragchatbot-raw,EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Grant the Pathway service account access to the bucket:

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:<PATHWAY_SA>" \
  --role="roles/storage.objectViewer"
```

### 4) Point ragchatbot to Pathway retriever

Set `PATHWAY_URL` (or `PATHWAY_HOST`/`PATHWAY_PORT`) in Cloud Run:

```
PATHWAY_URL=https://pathway-retriever-xxxxx-asia-east2.run.app
```

Then redeploy ragchatbot.

