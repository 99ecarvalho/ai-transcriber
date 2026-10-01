# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model lifecycle, concurrency control and the transcription itself.

Both APIs (native and OpenAI-compatible) go through Transcriber.transcribe_upload.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from fastapi import UploadFile

from .config import Settings
from .errors import ServiceError

log = structlog.get_logger("transcriber")

UPLOAD_CHUNK_BYTES = 1024 * 1024
GPU_ONLY_COMPUTE_TYPES = {"float16", "int8_float16", "bfloat16", "int8_bfloat16"}


# ---------- Device / compute type ----------

def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import ctranslate2

    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def resolve_compute_type(device: str, requested: str | None) -> str:
    if not requested:
        return "float16" if device == "cuda" else "int8"
    if device == "cpu" and requested in GPU_ONLY_COMPUTE_TYPES:
        log.warning("transcriber.compute_type_fallback", requested=requested, using="int8")
        return "int8"
    return requested


# ---------- Validation ----------

def normalize_language(language: str | None, default: str | None) -> str | None:
    """Returns a valid language code, or None for auto-detect."""
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    value = (language or "").strip().lower() or (default or "").strip().lower()
    if value in ("", "auto"):
        return None
    if value not in _LANGUAGE_CODES:
        raise ServiceError(
            400,
            f"Unsupported language {value!r}. Use an ISO-639-1 code such as en, pt, es, or leave empty "
            "for auto-detect.",
            param="language",
        )
    return value


# ---------- Results ----------

@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float | None
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[Word] | None


@dataclass
class TranscriptionResult:
    task: str
    language: str
    language_probability: float
    duration: float
    elapsed_ms: int
    segments: list[Segment]

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments).strip()


@dataclass
class TranscribeOptions:
    language: str | None = None
    task: str = "transcribe"
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = False
    initial_prompt: str | None = None
    temperature: float | None = None  # None -> faster-whisper's fallback schedule


# ---------- Model manager ----------

class ModelManager:
    """Loads the model once, lazily, off the event loop.

    After a failed load, further attempts are refused for `load_retry_sec`
    so a broken setup (no GPU, bad model name) fails fast instead of every
    request retrying a multi-minute load.
    """

    def __init__(self, settings: Settings, loader: Callable[[ModelManager], Any] | None = None):
        self.settings = settings
        self.device = resolve_device(settings.device)
        self.compute_type = resolve_compute_type(self.device, settings.compute_type)
        self._loader = loader or _load_whisper_model
        self._model: Any = None
        self._lock = asyncio.Lock()
        self.loading = False
        self.last_error: str | None = None
        self._last_error_at: float | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def state(self) -> str:
        if self._model is not None:
            return "ready"
        if self.loading:
            return "loading"
        if self.last_error is not None:
            return "failed"
        return "not_loaded"

    def _raise_if_cooling_down(self) -> None:
        if self._last_error_at is None:
            return
        remaining = self.settings.load_retry_sec - (time.monotonic() - self._last_error_at)
        if remaining > 0:
            raise ServiceError(
                503,
                "Model is unavailable (last load failed).",
                internal=self.last_error,
                retry_after=max(1, int(remaining + 0.999)),
            )

    async def get(self) -> Any:
        if self._model is not None:
            return self._model
        self._raise_if_cooling_down()
        async with self._lock:
            if self._model is not None:
                return self._model
            self._raise_if_cooling_down()

            log.info(
                "transcriber.loading_model",
                model=self.settings.model_name,
                device=self.device,
                compute_type=self.compute_type,
                cache_dir=self.settings.cache_dir,
            )
            t0 = time.monotonic()
            self.loading = True
            try:
                # Download + load can take minutes — run it off the event loop so
                # /health and other requests keep responding meanwhile.
                model = await asyncio.to_thread(self._loader, self)
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                self._last_error_at = time.monotonic()
                log.exception("transcriber.model_load_failed")
                raise ServiceError(
                    503,
                    "Failed to load model.",
                    internal=self.last_error,
                    retry_after=max(1, int(self.settings.load_retry_sec)),
                ) from e
            finally:
                self.loading = False

            self._model = model
            self.last_error = None
            self._last_error_at = None
            log.info("transcriber.model_loaded", elapsed_sec=round(time.monotonic() - t0, 2))
        return self._model


def _load_whisper_model(manager: ModelManager) -> Any:
    from faster_whisper import WhisperModel

    Path(manager.settings.cache_dir).mkdir(parents=True, exist_ok=True)
    return WhisperModel(
        manager.settings.model_name,
        device=manager.device,
        compute_type=manager.compute_type,
        download_root=manager.settings.cache_dir,
    )


# ---------- Concurrency ----------

class Limiter:
    """Bounds concurrent GPU jobs and the number of requests waiting for one.

    A slot is held until the worker thread actually finishes — even if the
    client disconnects and the request is cancelled — so abandoned requests
    can't oversubscribe the GPU.
    """

    def __init__(self, max_concurrent: int, max_queue: int):
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self._sem = asyncio.Semaphore(max_concurrent)
        self.waiting = 0
        self.active = 0

    async def run(self, fn: Callable[[], Any]) -> Any:
        if self.active >= self.max_concurrent and self.waiting >= self.max_queue:
            raise ServiceError(503, "Server is busy, try again later.", retry_after=5)

        self.waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self.waiting -= 1

        self.active += 1
        future = asyncio.ensure_future(asyncio.to_thread(fn))

        def _release(f: asyncio.Future) -> None:
            self.active -= 1
            self._sem.release()
            if not f.cancelled():
                f.exception()  # mark as retrieved if nobody is awaiting anymore

        future.add_done_callback(_release)
        return await asyncio.shield(future)


# ---------- Transcription ----------

class _BadAudio(Exception):
    pass


def _transcribe_file(model: Any, path: str, opts: TranscribeOptions) -> tuple[list[Segment], Any]:
    from faster_whisper import decode_audio

    try:
        audio = decode_audio(path)
    except Exception as e:
        raise _BadAudio(f"{type(e).__name__}: {e}") from e
    if audio.size == 0:
        raise _BadAudio("no audio samples")

    kwargs: dict[str, Any] = dict(
        language=opts.language,
        task=opts.task,
        beam_size=opts.beam_size,
        vad_filter=opts.vad_filter,
        word_timestamps=opts.word_timestamps,
        initial_prompt=opts.initial_prompt,
    )
    if opts.temperature is not None:
        kwargs["temperature"] = opts.temperature

    segments, info = model.transcribe(audio, **kwargs)
    # segments is a generator — force consumption to measure the real duration
    segs = [
        Segment(
            id=s.id,
            seek=s.seek,
            start=s.start,
            end=s.end,
            text=s.text,
            tokens=list(s.tokens),
            temperature=s.temperature,
            avg_logprob=s.avg_logprob,
            compression_ratio=s.compression_ratio,
            no_speech_prob=s.no_speech_prob,
            words=[Word(w.start, w.end, w.word, w.probability) for w in s.words] if s.words else None,
        )
        for s in segments
    ]
    return segs, info


async def _save_upload(file: UploadFile) -> tuple[str, int]:
    # Write to a temp file because faster-whisper's decoder works best with a
    # path (format auto-detection). Keep the suffix as a format hint. Copy in
    # chunks so the whole upload is never held in memory.
    suffix = Path(file.filename or "audio").suffix[:16] or ".bin"
    size = 0
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                tmp.write(chunk)
    except Exception as e:
        log.exception("transcriber.tmp_write_failed")
        if tmp_path:
            _unlink_quietly(tmp_path)
        raise ServiceError(500, "Failed to store upload.", internal=f"{type(e).__name__}: {e}") from e
    return tmp_path, size


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


class Transcriber:
    def __init__(self, settings: Settings, loader: Callable[[ModelManager], Any] | None = None):
        self.settings = settings
        self.models = ModelManager(settings, loader)
        self.limiter = Limiter(settings.max_concurrent, settings.max_queue)

    async def transcribe_upload(self, file: UploadFile, opts: TranscribeOptions) -> TranscriptionResult:
        tmp_path, size = await _save_upload(file)
        try:
            if size == 0:
                raise ServiceError(400, "File is empty.", param="file")

            model = await self.models.get()
            t0 = time.monotonic()
            try:
                segs, info = await self.limiter.run(lambda: _transcribe_file(model, tmp_path, opts))
            except _BadAudio as e:
                log.info("transcriber.bad_audio", reason=str(e), **self._file_fields(file))
                raise ServiceError(
                    400, "Could not decode audio from the uploaded file.", internal=str(e), param="file"
                ) from e
            except ServiceError:
                raise
            except Exception as e:
                log.exception("transcriber.transcribe_failed", **self._file_fields(file))
                raise ServiceError(500, "Transcription failed.", internal=f"{type(e).__name__}: {e}") from e
        finally:
            _unlink_quietly(tmp_path)

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        result = TranscriptionResult(
            task=opts.task,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            elapsed_ms=elapsed_ms,
            segments=segs,
        )
        log.info(
            "transcriber.done",
            **self._file_fields(file),
            size_kb=round(size / 1024.0, 1),
            task=opts.task,
            language=info.language,
            language_prob=round(info.language_probability, 3),
            audio_duration_sec=round(info.duration, 2),
            elapsed_ms=elapsed_ms,
            segments=len(segs),
            rtf=round(elapsed_ms / 1000.0 / max(info.duration, 0.001), 3),
        )
        return result

    def _file_fields(self, file: UploadFile) -> dict[str, Any]:
        return {"filename": file.filename} if self.settings.log_filenames else {}
