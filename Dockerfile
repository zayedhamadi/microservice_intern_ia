FROM python:3.12-slim AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY train_offline.py .

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/model_store \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]