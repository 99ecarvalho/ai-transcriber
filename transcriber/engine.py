# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model lifecycle, concurrency control and the transcription itself.

Both APIs (native and OpenAI-compatible) go through Transcriber.transcribe_upload
(or transcribe_source, which also accepts a URL). Live streams reuse the same
models from streaming.py.
"""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import structlog
from fastapi import UploadFile

from .config import Settings
from .errors import ServiceError
from .lazy import LazyLoader

log = structlog.get_logger("transcriber")

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

def normalize_language(
    language: str | None, default: str | None, param: str = "language"
) -> str | None:
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
            param=param,
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
    speaker: str | None = None  # set only when diarization was requested
    translation: str | None = None  # set only when translate_to was requested


@dataclass
class TranscriptionResult:
    task: str
    language: str
    language_probability: float
    duration: float
    elapsed_ms: int
    segments: list[Segment]
    translation_language: str | None = None
    diarized: bool = False

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments).strip()

    @property
    def translation(self) -> str | None:
        if self.translation_language is None:
            return None
        return " ".join(s.translation for s in self.segments if s.translation).strip()


@dataclass
class TranscribeOptions:
    language: str | None = None
    task: str = "transcribe"
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = False
    initial_prompt: str | None = None
    temperature: float | None = None  # None -> faster-whisper's fallback schedule
    diarize: bool = False
    num_speakers: int | None = None
    translate_to: str | None = None


# ---------- Lazily loaded models ----------

class ModelManager(LazyLoader):
    """The Whisper model. See LazyLoader for the loading semantics."""

    def __init__(self, settings: Settings, loader: Callable[[ModelManager], Any] | None = None):
        self.settings = settings
        self.device = resolve_device(settings.device)
        self.compute_type = resolve_compute_type(self.device, settings.compute_type)
        model_loader = loader or _load_whisper_model
        super().__init__("model", "model", settings.load_retry_sec, lambda: model_loader(self))

    def log_fields(self) -> dict[str, Any]:
        return {
            "model": self.settings.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "num_workers": self.settings.max_concurrent,
            "cache_dir": self.settings.cache_dir,
        }


def _load_whisper_model(manager: ModelManager) -> Any:
    from faster_whisper import WhisperModel

    Path(manager.settings.cache_dir).mkdir(parents=True, exist_ok=True)
    return WhisperModel(
        manager.settings.model_name,
        device=manager.device,
        compute_type=manager.compute_type,
        download_root=manager.settings.cache_dir,
        # One worker per allowed concurrent job; with the default of 1,
        # concurrent transcribe() calls would run one at a time.
        num_workers=manager.settings.max_concurrent,
    )


def _load_translator(settings: Settings) -> Any:
    from .translate import TextTranslator, resolve_model_dir

    device = resolve_device(settings.translation_device)
    translator = TextTranslator(
        resolve_model_dir(settings.translation_model),
        device=device,
        compute_type="int8_float16" if device == "cuda" else "int8",
    )
    if translator.family == "nllb":
        log.warning(
            "transcriber.translation_license",
            model=settings.translation_model,
            note="NLLB-200 weights are CC-BY-NC-4.0: non-commercial use only. "
            "Set TRANSLATION_MODEL to an M2M100 model for commercial use.",
        )
    return translator


def _load_speaker_model(settings: Settings) -> Any:
    from .diarize import SpeakerModel

    return SpeakerModel(
        method=settings.diarization_method,
        model=settings.diarization_model,
        cache_dir=settings.diarization_cache_dir,
        threshold=settings.diarization_threshold,
        threads=settings.diarization_threads,
    )


# ---------- Concurrency ----------

class Ticket:
    """One admitted request. Released exactly once: by the request if it never
    reaches a GPU slot, otherwise when its worker thread finishes."""

    def __init__(self, limiter: Limiter):
        self._limiter = limiter
        self._released = False
        self._handed_to_worker = False

    def release(self) -> None:
        if self._handed_to_worker or self._released:
            return
        self._released = True
        self._limiter.in_flight -= 1


class Limiter:
    """Bounds concurrent GPU jobs and the number of requests waiting.

    At most max_concurrent + max_queue requests are admitted at once, counting
    those waiting for the first model load, those waiting for a GPU slot and
    those running. A slot is held until the worker thread actually finishes —
    even if the client disconnects and the request is cancelled — so abandoned
    requests can't oversubscribe the GPU.
    """

    def __init__(self, max_concurrent: int, max_queue: int):
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self._sem = asyncio.Semaphore(max_concurrent)
        self.in_flight = 0
        self.active = 0

    @property
    def waiting(self) -> int:
        return self.in_flight - self.active

    def admit(self) -> Ticket:
        if self.in_flight >= self.max_concurrent + self.max_queue:
            raise ServiceError(503, "Server is busy, try again later.", retry_after=5)
        self.in_flight += 1
        return Ticket(self)

    async def run(self, ticket: Ticket, fn: Callable[[], Any]) -> Any:
        await self._sem.acquire()  # if cancelled here, the caller releases the ticket
        self.active += 1
        ticket._handed_to_worker = True
        future = asyncio.ensure_future(asyncio.to_thread(fn))

        def _done(f: asyncio.Future) -> None:
            self.active -= 1
            self.in_flight -= 1
            self._sem.release()
            if not f.cancelled():
                f.exception()  # mark as retrieved if nobody is awaiting anymore

        future.add_done_callback(_done)
        return await asyncio.shield(future)


# ---------- Transcription ----------

class _BadAudio(Exception):
    pass


class _BadRequest(Exception):
    """Raised in the worker thread for problems only detectable after decoding."""

    def __init__(self, message: str, param: str | None = None):
        super().__init__(message)
        self.param = param


def _decode_upload(audio_file: BinaryIO) -> np.ndarray:
    from faster_whisper import decode_audio

    try:
        audio_file.seek(0)
        return decode_audio(audio_file)
    except Exception as e:
        raise _BadAudio(f"{type(e).__name__}: {e}") from e


def _decode_from_url(url: str, max_seconds: float) -> np.ndarray:
    from .audio import _TooLong, decode_url

    try:
        return decode_url(url, max_seconds)
    except _TooLong as e:
        raise _BadRequest(f"The audio at url is too long: {e}.", param="url") from e
    except Exception as e:
        raise _BadAudio(f"{type(e).__name__}: {e}") from e


def transcribe_audio(model: Any, audio: np.ndarray, opts: TranscribeOptions) -> tuple[list[Segment], Any]:
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


def translate_segments(translator: Any, segs: list[Segment], source: str, target: str) -> None:
    if not translator.supports(source):
        raise _BadRequest(
            f"Translating from {source!r} isn't supported by the translation model.", param="translate_to"
        )
    translations = translator.translate([s.text for s in segs], source, target)
    for seg, text in zip(segs, translations, strict=True):
        seg.translation = text


def _run_pipeline(
    model: Any,
    get_audio: Callable[[], np.ndarray],
    opts: TranscribeOptions,
    speakers: Any = None,
    translator: Any = None,
) -> tuple[list[Segment], Any]:
    audio = get_audio()
    if audio.size == 0:
        raise _BadAudio("no audio samples")
    segs, info = transcribe_audio(model, audio, opts)
    if speakers is not None:
        labels = speakers.label_segments(audio, [(s.start, s.end) for s in segs], opts.num_speakers)
        for seg, label in zip(segs, labels, strict=True):
            seg.speaker = label
    if translator is not None and segs:
        source = "en" if opts.task == "translate" else info.language
        translate_segments(translator, segs, source, opts.translate_to)
    return segs, info


def _upload_size(file: UploadFile) -> int:
    if file.size is not None:
        return file.size
    file.file.seek(0, os.SEEK_END)
    return file.file.tell()


class Transcriber:
    def __init__(
        self,
        settings: Settings,
        loader: Callable[[ModelManager], Any] | None = None,
        translation_loader: Callable[[], Any] | None = None,
        speaker_loader: Callable[[], Any] | None = None,
    ):
        try:
            normalize_language(settings.default_language, None)
        except ServiceError as e:
            raise ValueError(f"WHISPER_LANGUAGE: {e.message}") from None
        self.settings = settings
        self.models = ModelManager(settings, loader)
        self.limiter = Limiter(settings.max_concurrent, settings.max_queue)
        self.translator = LazyLoader(
            "translation model",
            "translator",
            settings.load_retry_sec,
            translation_loader or (lambda: _load_translator(settings)),
        )
        self.speakers = LazyLoader(
            "speaker model",
            "speaker_model",
            settings.load_retry_sec,
            speaker_loader or (lambda: _load_speaker_model(settings)),
        )
        self.active_streams = 0
        self.stream_vad: Callable[[np.ndarray], list[tuple[int, int]]] | None = None  # None -> Silero

    async def transcribe_upload(self, file: UploadFile, opts: TranscribeOptions) -> TranscriptionResult:
        return await self.transcribe_source(opts, file=file)

    async def transcribe_source(
        self, opts: TranscribeOptions, file: UploadFile | None = None, url: str | None = None
    ) -> TranscriptionResult:
        if url is not None:
            await asyncio.to_thread(self.check_url, url)  # DNS lookup — keep it off the event loop
            size = 0
            source_fields: dict[str, Any] = {"source": "url"}
            settings = self.settings

            def get_audio() -> np.ndarray:
                return _decode_from_url(url, settings.max_url_duration_sec)
        else:
            # The multipart parser has already stored the upload (in memory when
            # small, spooled to a temp file otherwise); decode straight from it.
            # FFmpeg detects the format from the content, so no filename is needed.
            size = _upload_size(file)
            if size == 0:
                raise ServiceError(400, "File is empty.", param="file")
            source_fields = self._file_fields(file)

            def get_audio() -> np.ndarray:
                return _decode_upload(file.file)

        ticket = self.limiter.admit()
        try:
            model = await self.models.get()
            speakers = await self.speakers.get() if opts.diarize else None
            translator = await self._translator_for(opts.translate_to) if opts.translate_to else None
            t0 = time.monotonic()
            try:
                segs, info = await self.limiter.run(
                    ticket, lambda: _run_pipeline(model, get_audio, opts, speakers, translator)
                )
            except _BadAudio as e:
                log.info("transcriber.bad_audio", reason=str(e), **source_fields)
                where = "url" if url is not None else "the uploaded file"
                raise ServiceError(
                    400,
                    f"Could not decode audio from {where}.",
                    internal=str(e),
                    param="url" if url is not None else "file",
                ) from e
            except _BadRequest as e:
                raise ServiceError(400, str(e), param=e.param) from e
            except ServiceError:
                raise
            except Exception as e:
                log.exception("transcriber.transcribe_failed", **source_fields)
                raise ServiceError(500, "Transcription failed.", internal=f"{type(e).__name__}: {e}") from e
        finally:
            ticket.release()

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        result = TranscriptionResult(
            task=opts.task,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            elapsed_ms=elapsed_ms,
            segments=segs,
            translation_language=opts.translate_to,
            diarized=opts.diarize,
        )
        extra = {}
        if opts.diarize:
            extra["speakers"] = len({s.speaker for s in segs})
        if opts.translate_to:
            extra["translate_to"] = opts.translate_to
        log.info(
            "transcriber.done",
            **source_fields,
            size_kb=round(size / 1024.0, 1),
            task=opts.task,
            language=info.language,
            language_prob=round(info.language_probability, 3),
            audio_duration_sec=round(info.duration, 2),
            elapsed_ms=elapsed_ms,
            segments=len(segs),
            rtf=round(elapsed_ms / 1000.0 / max(info.duration, 0.001), 3),
            **extra,
        )
        return result

    async def _translator_for(self, target: str) -> Any:
        translator = await self.translator.get()
        if not translator.supports(target):
            raise ServiceError(
                400,
                f"Translating to {target!r} isn't supported by the translation model.",
                param="translate_to",
            )
        return translator

    def check_url(self, url: str) -> None:
        from .audio import check_url

        if not self.settings.allow_urls:
            raise ServiceError(403, "URL input is disabled on this server (set ALLOW_URLS=1).", param="url")
        check_url(url, self.settings.allow_private_urls)

    def _file_fields(self, file: UploadFile) -> dict[str, Any]:
        return {"filename": file.filename} if self.settings.log_filenames else {}
