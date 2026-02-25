FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline ./pipeline

ENV PYTHONUNBUFFERED=1
ENV ARTIFACTS_DIR=/artifacts
