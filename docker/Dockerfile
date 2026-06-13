# iTrialSpace — lightweight image for the core pipeline + web tools.
#
#   docker build -t itrialspace -f docker/Dockerfile .
#   docker run --rm -e ITRIALSPACE_DATA_DIR=/data -v /host/data:/data itrialspace its config
#
# GPU stages (synthesis, VLM) need a CUDA base image + torch; this image covers the
# core/imaging/web extras. See docs/installation.md for the GPU setup.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ITRIALSPACE_DATA_DIR=/data

# System libs needed by nibabel/SimpleITK/matplotlib/faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching), then the package.
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --upgrade pip && pip install ".[imaging,web]"

# Bring in the rest (docs, tests, infra, sample data)
COPY . .

# Non-root user
RUN useradd -m -u 1000 its && mkdir -p /data && chown -R its:its /app /data
USER its

ENTRYPOINT ["its"]
CMD ["config"]
