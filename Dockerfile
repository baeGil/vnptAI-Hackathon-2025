# BASE IMAGE
# Using Ubuntu 22.04 for better Python 3.11 support while maintaining CUDA 12.2
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# SYSTEM DEPENDENCIES
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python 3.11 from deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Qdrant binary FROM official Docker image (ensures data format compatibility)
COPY --from=qdrant/qdrant:latest /qdrant/qdrant /usr/local/bin/qdrant

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# PROJECT SETUP
WORKDIR /code
COPY . /code

# ENV SETUP
RUN uv venv .venv --python 3.11
ENV PATH="/code/.venv/bin:$PATH"

# Install dependencies using requirements.txt (as requested by BTC)
RUN uv pip install -r requirements.txt

# EXECUTION
# Ensure scripts are executable
RUN chmod +x inference.sh

CMD ["bash", "inference.sh"]