# BASE IMAGE
# Using CUDA 12.2 as required. 
# Note: The base image nvidia/cuda:12.2.0-devel-ubuntu20.04 comes with Python 3.8 default.
# We need to install Python 3.11.

FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# SYSTEM DEPENDENCIES
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install software-properties-common to add deadsnakes PPA for Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# PROJECT SETUP
WORKDIR /code
COPY . /code

# ENV SETUP
# Create a virtual environment using uv directly
RUN uv venv .venv --python 3.11

# Activate venv for subsequent commands
ENV PATH="/code/.venv/bin:$PATH"

# Install dependencies using uv
RUN uv pip install -r pyproject.toml
# Or use uv sync if using the lockfile workflow, but for simplicity with pyproject.toml:
# RUN uv pip install .

# EXECUTION
CMD ["bash", "inference.sh"]
