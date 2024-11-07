FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directory for YAML files and ensure it exists
RUN mkdir -p /app/prompts

# Copy prompts.yaml file
COPY prompts/prompts.yaml /app/prompts/

# Environment variables will be provided at runtime
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8001

# Expose the port
EXPOSE 8001

# Run both the API and the Python script
CMD ["python", "sukoon.py"]