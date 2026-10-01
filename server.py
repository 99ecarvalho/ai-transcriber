"""Transcriber HTTP service — faster-whisper + CUDA.

Endpoints:
  GET  /health           -> {"status":"ok","model":..., "device":...}
  POST /transcribe       -> multipart file upload, returns {text, language, elapsed_ms, ...}

Config via env:
  WHISPER_MODEL           default "large-v3"  (tiny|base|small|medium|large-v2|large-v3|distil-large-v3)
  WHISPER_DEVICE          default "cuda"
  WHISPER_COMPUTE_TYPE    default "float16"   (float16|int8_float16|int8)
  WHISPER_LANGUAGE        default None        (auto-detect if empty)
  WHISPER_CACHE_DIR       default "/cache/faster-whisper"
  MAX_UPLOAD_MB           default 200         (requests with a larger body get a 413)
  HOST/PORT               default "0.0.0.0:8000"

The model is loaded lazily (on the first /transcribe) to allow fast startup
and so that GPU failures return a 500 instead of crashing the container. The
model is downloaded on first use and cached in the /cache volume.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse


# ---------- Config ----------

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
DEFAULT_LANGUAGE = os.environ.get("WHISPER_LANGUAGE") or None
CACHE_DIR = os.environ.get("WHISPER_CACHE_DIR", "/cache/faster-whisper")
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024)
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

UPLOAD_CHUNK_BYTES = 1024 * 1024


# ---------- Logging ----------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.PrintLoggerFactory(),
)
structlog.contextvars.bind_contextvars(component="transcriber")
log = structlog.get_logger("transcriber")


# ---------- Model loading (lazy) ----------

_model = None
_model_lock = asyncio.Lock()


def _load_model():
    from faster_whisper import WhisperModel

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root=CACHE_DIR,
    )


async def get_model():
    global _model
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is not None:
            return _model
        log.info(
            "transcriber.loading_model",
            model=MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            cache_dir=CACHE_DIR,
        )
        t0 = time.monotonic()
        # Download + load can take minutes — run it off the event loop so
        # /health and other requests keep responding meanwhile.
        _model = await asyncio.to_thread(_load_model)
        log.info("transcriber.model_loaded", elapsed_sec=round(time.monotonic() - t0, 2))
    return _model


# ---------- Upload size limit ----------

def _too_large_detail() -> str:
    return f"Request body exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."


class MaxBodySizeMiddleware:
    """Rejects request bodies larger than max_bytes with a 413.

    Checks Content-Length up front, and also counts the bytes actually received
    so chunked uploads (or a lying Content-Length) can't bypass the limit.
    The multipart parser spools uploads to disk, so this also bounds disk usage.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break
                if declared > self.max_bytes:
                    response = JSONResponse({"detail": _too_large_detail()}, status_code=413)
                    await response(scope, receive, send)
                    return
                break

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(status_code=413, detail=_too_large_detail())
            return message

        await self.app(scope, limited_receive, send)


# ---------- App ----------

app = FastAPI(title="ai-transcriber", version="0.1.0")
app.add_middleware(MaxBodySizeMiddleware, max_bytes=MAX_UPLOAD_BYTES)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "model_loaded": _model is not None,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (any format supported by ffmpeg)"),
    language: str | None = Form(default=None, description="pt|en|es|... or empty for auto-detect"),
    vad_filter: bool = Form(default=True, description="Voice activity detection (filters silence)"),
    beam_size: int = Form(default=5, ge=1, le=20),
):
    # Write to a temp file because faster-whisper takes a path (better for ffmpeg
    # format auto-detection). Keep the suffix as a format hint. Copy in chunks
    # so the whole upload is never held in memory.
    suffix = Path(file.filename or "audio").suffix or ".bin"
    size_bytes = 0
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                size_bytes += len(chunk)
                tmp.write(chunk)
    except Exception as e:
        log.exception("transcriber.tmp_write_failed")
        if tmp_path:
            _unlink_quietly(tmp_path)
        raise HTTPException(status_code=500, detail=f"Failed to write temp: {e}")

    try:
        if size_bytes == 0:
            raise HTTPException(status_code=400, detail="File is empty.")

        try:
            model = await get_model()
        except Exception as e:
            log.exception("transcriber.model_load_failed")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

        lang = language or DEFAULT_LANGUAGE
        t0 = time.monotonic()

        def _run_transcription():
            segments, info = model.transcribe(
                tmp_path,
                language=lang,
                beam_size=beam_size,
                vad_filter=vad_filter,
            )
            # segments is a generator — force consumption to measure the real duration
            segs = [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in segments
            ]
            return segs, info

        try:
            segs, info = await asyncio.to_thread(_run_transcription)
        except Exception as e:
            log.exception("transcriber.transcribe_failed", filename=file.filename)
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        _unlink_quietly(tmp_path)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    text = "".join(s["text"] for s in segs).strip()

    log.info(
        "transcriber.done",
        filename=file.filename,
        size_kb=round(size_bytes / 1024.0, 1),
        language=info.language,
        language_prob=round(info.language_probability, 3),
        audio_duration_sec=round(info.duration, 2),
        elapsed_ms=elapsed_ms,
        segments=len(segs),
        rtf=round(elapsed_ms / 1000.0 / max(info.duration, 0.001), 3),
    )

    return JSONResponse({
        "text": text,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "audio_duration_sec": round(info.duration, 2),
        "elapsed_ms": elapsed_ms,
        "segments": segs,
    })


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


if __name__ == "__main__":
    import uvicorn

    log.info("transcriber.starting", host=HOST, port=PORT, model=MODEL_NAME, device=DEVICE)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
