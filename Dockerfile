# Transcriber: faster-whisper + CUDA + FastAPI.
#
# Runs as a standalone HTTP service — any client can call it via
# POST /transcribe (multipart audio).
#
# Base nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04: ships CUDA + cuDNN 9,
# compatible with ctranslate2 4.x (a dependency of faster-whisper 1.x).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY server.py .

# Volume for the model cache (1-3GB per model — avoids re-downloading)
VOLUME ["/cache"]

EXPOSE 8000

CMD ["python3", "-u", "server.py"]
