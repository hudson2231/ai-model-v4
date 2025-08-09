FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System Python 3.10 + tools
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev git curl ca-certificates tini \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy requirements first for layer caching
COPY requirements.txt /src/requirements.txt

# Install deps with resilience; prefer wheels; no cache to keep memory low
ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip && \
    pip install --default-timeout=180 --retries=5 --prefer-binary -r /src/requirements.txt

# Copy the rest of the project
COPY . /src

# Cog runtime (same version your logs showed)
RUN pip install https://github.com/replicate/cog-runtime/releases/download/v0.1.0-beta5/coglet-0.1.0b5-py3-none-any.whl

# Start the predictor
ENTRYPOINT ["tini","-g","--","coglet","predict"]
