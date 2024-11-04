# # Use an official Python runtime as a parent image
# FROM python:3.9

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# # Make the tests directory available
# COPY tests/ /app/tests/

# CMD ["uvicorn", "sukoon_api:app", "--host", "127.0.0.1", "--port", "8001"]

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

# Copy test files
COPY tests/ /app/tests/

# Create directory for YAML files
RUN mkdir -p /app/prompts

# Copy prompts.yaml file
COPY prompts/prompts.yaml /app/prompts/

# Environment variables will be provided at runtime
ENV PYTHONPATH=/app
ENV HOST=127.0.0.1
ENV PORT=8001

# Run the application
CMD ["python", "sukoon.py"]