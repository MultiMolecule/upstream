ARG TENSORFLOW2_PLATFORM=linux/amd64
FROM --platform=$TENSORFLOW2_PLATFORM tensorflow/tensorflow:2.15.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.12.0+cpu

RUN python -m pip install --no-cache-dir \
    biopython==1.87 \
    chanfig \
    natsort==8.4.0 \
    numpy==1.26.4 \
    pandas==3.0.3 \
    safetensors==0.7.0 \
    scipy==1.17.1 \
    tqdm==4.67.3

ENV TF_CPP_MIN_LOG_LEVEL=2
