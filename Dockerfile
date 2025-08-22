FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for BGE-M3
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (BGE-M3 directory is excluded via .dockerignore)
COPY . .

# Create the BGE-M3 directory for mounting or auto-download
RUN mkdir -p BGE-M3

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables for BGE-M3
ENV MODEL_PATH=/app/BGE-M3
ENV MODEL_NAME=BAAI/bge-m3
ENV PYTHONUNBUFFERED=1

# Check if .env exists, and copy from .env.example if not
CMD if [ ! -f .env ]; then echo "No .env file found, copying from .env.example..."; cp -n .env.example .env; fi && \
    uvicorn src.main:app --host 0.0.0.0 --port 8000 