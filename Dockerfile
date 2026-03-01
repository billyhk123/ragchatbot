FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libmagic1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY app ./app
COPY ingest ./ingest
COPY server ./server
RUN pip install --no-cache-dir .

COPY . .

RUN if [ -d "data/raw" ] && [ "$(ls -A data/raw 2>/dev/null)" ]; then python -m ingest.ingest; else echo "Skipping vector DB build (no data/raw)"; fi

CMD ["sh", "-c", "uvicorn server.server:app --host 0.0.0.0 --port ${PORT:-8080}"]