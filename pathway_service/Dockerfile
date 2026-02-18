FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY pathway_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY pathway_service/app.py ./app.py

ENV PATHWAY_HOST=0.0.0.0
ENV PATHWAY_PORT=8080

CMD ["python", "app.py"]
