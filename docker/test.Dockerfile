FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pytest-cov httpx

COPY api ./api
COPY pipeline ./pipeline
COPY tests ./tests

ENV PYTHONUNBUFFERED=1
ENV ARTIFACTS_DIR=/artifacts
ENV PYTHONPATH=/app

CMD ["pytest", "-q"]
