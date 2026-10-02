# Lightweight CPU-only image (builds in ~1-2 minutes)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# make + git are handy inside Codespaces
RUN apt-get update \
    && apt-get install -y --no-install-recommends make git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

# Run via bash so it works even if the execute bit is lost on a mounted volume
CMD ["bash", "entrypoint.sh"]
