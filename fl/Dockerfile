# FL Bank Node Dockerfile
# Base: Python 3.11 slim
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY fl/requirements.txt /app/fl/requirements.txt
RUN pip install --no-cache-dir -r fl/requirements.txt

# Copy project source
COPY fl/ /app/fl/
COPY data/vocab/ /app/data/vocab/
COPY experiments/ /app/experiments/

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default: run as a bank FL node
# Override CMD via docker-compose environment for different roles
CMD ["python", "-m", "fl.bank_node.node"]
