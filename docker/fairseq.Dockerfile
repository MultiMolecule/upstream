# MultiMolecule upstream fixture image for fairseq-based models.
#
# Builds the shared fairseq runtime image. Model code and checkpoints are
# fetched by generate.py into the upstream cache, or supplied through documented
# environment variables. fairseq 0.12.2 is pinned because newer fairseq does not
# support the upstream ERNIE-RNA MaskedLM checkpoint. Python is pinned to 3.10
# because fairseq's dataclass-based config does not import under 3.11. pip is
# pinned below 24.1 because newer resolvers reject fairseq's metadata.
#
# Build:
#   docker build --platform linux/amd64 -t ghcr.io/multimolecule/fairseq:latest \
#     -f docker/fairseq.Dockerfile docker
#
# Regenerate the fixture (mounts the upstream repo so generate.py can read
# _corpus and write _out):
#   docker run --rm --platform linux/amd64 -v "$(pwd):/work" -w /work \
#     ghcr.io/multimolecule/fairseq:latest \
#     python models/ernierna/ernierna/generate.py

ARG FAIRSEQ_PLATFORM=linux/amd64
FROM --platform=$FAIRSEQ_PLATFORM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# pip<24.1 is required because the newer resolver rejects fairseq's metadata.
RUN python -m pip install "pip<24.1"

RUN pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.0.1+cpu" \
      "torchaudio==2.0.2+cpu"

RUN pip install \
      "antlr4-python3-runtime==4.8" \
      "biopython==1.87" \
      "bitarray==3.8.1" \
      "cffi==2.0.0" \
      "chanfig" \
      "cython==3.2.5" \
      "gdown==6.0.0" \
      "hydra-core==1.0.7" \
      "numpy==1.26.4" \
      "omegaconf==2.0.6" \
      "packaging==26.2" \
      "PyYAML==6.0.3" \
      "regex==2026.5.9" \
      "sacrebleu==2.6.0" \
      "safetensors==0.7.0" \
      "scikit-learn==1.7.2" \
      "tqdm==4.67.3"

# fairseq 0.12.2 PyPI sdist is missing version.txt, so install from the v0.12.2
# tag in the upstream git repository instead.
RUN pip install --no-deps "git+https://github.com/facebookresearch/fairseq.git@v0.12.2#egg=fairseq"

WORKDIR /work
