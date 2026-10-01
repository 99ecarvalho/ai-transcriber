# Transcriber: faster-whisper + CUDA + FastAPI.
#
# Runs as a standalone HTTP service — any client can call it via
# POST /transcribe (native API) or POST /v1/audio/transcriptions (OpenAI-compatible).
#
# Base nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04: ships CUDA + cuDNN 9,
# compatible with ctranslate2 4.x (a dependency of faster-whisper 1.x).
# Audio decoding uses PyAV, whose wheels bundle FFmpeg — no system ffmpeg needed.
#
# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
# SPDX-License-Identifier: LGPL-3.0-or-later
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Unprivileged user; /cache is pre-created so a fresh named volume inherits its ownership.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /cache \
    && chown app:app /cache

COPY COPYING COPYING.LESSER ./
COPY transcriber/ ./transcriber/

USER app

# Volume for the model cache (1-3GB per model — avoids re-downloading)
VOLUME ["/cache"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

CMD ["python3", "-m", "transcriber"]
