ARG PYTORCH2_PLATFORM=linux/amd64
FROM --platform=$PYTORCH2_PLATFORM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.12.0+cpu

RUN python -m pip install --no-cache-dir \
      biopython==1.87 \
      axial_positional_embedding==0.3.12 \
      chanfig \
      einops==0.8.2 \
      gdown==6.0.0 \
      h5py==3.16.0 \
      huggingface_hub==0.36.2 \
      kaggle==2.2.0 \
      matplotlib==3.10.9 \
      ml_collections==1.1.0 \
      numpy==1.26.4 \
      omegaconf==2.3.0 \
      packaging==26.2 \
      pandas==3.0.3 \
      PyYAML==6.0.3 \
      rotary_embedding_torch==0.8.9 \
      safetensors==0.7.0 \
      scikit-learn==1.7.2 \
      sinkhorn-transformer==0.11.4 \
      tokenizers==0.22.2 \
      transformers==4.57.6

WORKDIR /work
