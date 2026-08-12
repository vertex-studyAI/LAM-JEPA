FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN python -m pip install --upgrade pip \
    && python -m pip install .

# Keep the image useful as a reproducible research environment rather than
# pretending it is a production service. Override CMD to run documented
# training/evaluation scripts when launching the container.
CMD ["lam-jepa", "--help"]
