ARG PADDLE_PLATFORM=linux/amd64
FROM --platform=$PADDLE_PLATFORM python:3.10-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip==26.1.1 setuptools==82.0.1 wheel==0.47.0 && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.0+cpu && \
    python -m pip install \
        numpy==1.26.4 \
        aistudio-sdk==0.2.6 \
        paddlepaddle==2.6.2 \
        paddlenlp==2.8.1 \
        biopython==1.87 \
        chanfig \
        gdown==6.0.0 \
        safetensors==0.7.0 \
        PyYAML==6.0.3
