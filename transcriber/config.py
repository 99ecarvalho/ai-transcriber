# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Configuration, read once from environment variables at startup.

See the README for the full list. Invalid values fail fast with a ValueError.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

MB = 1024 * 1024
LOG_LEVELS = ("debug", "info", "warning", "error", "critical")  # names uvicorn accepts
DIARIZATION_METHODS = ("segments", "pyannote")
# Winners of the comparisons documented in the README.
# NLLB-200 beat M2M100 on quality, speed and language coverage, but its weights are
# CC-BY-NC-4.0 (non-commercial). M2M100 (MIT): "jncraton/m2m100_418M-ct2-int8".
DEFAULT_TRANSLATION_MODEL = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
M2M100_TRANSLATION_MODEL = "jncraton/m2m100_418M-ct2-int8"
DEFAULT_SPEAKER_MODEL = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k"


def _str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _bool(name: str, default: bool) -> bool:
    value = _str(name)
    if value is None:
        return default
    if value.lower() in ("1", "true", "yes", "on"):
        return True
    if value.lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean (1/0, true/false), got {value!r}")


def _int(name: str, default: int, minimum: int) -> int:
    value = _str(name)
    result = default if value is None else int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _float(name: str, default: float, minimum: float) -> float:
    value = _str(name)
    result = default if value is None else float(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


@dataclass(frozen=True)
class Settings:
    model_name: str = "large-v3"
    device: str = "cuda"  # cuda | cpu | auto
    compute_type: str | None = None  # None -> float16 on cuda, int8 on cpu
    default_language: str | None = None  # None -> auto-detect
    cache_dir: str = "/cache/faster-whisper"
    preload: bool = False
    load_retry_sec: float = 60.0

    max_upload_bytes: int = 200 * MB
    max_concurrent: int = 1
    max_queue: int = 8

    api_key: str | None = None
    expose_error_details: bool = False
    log_filenames: bool = False
    log_level: str = "info"

    host: str = "0.0.0.0"
    port: int = 8000

    # URL input (batch `url=` and live streams pulled by the server)
    allow_urls: bool = False
    allow_private_urls: bool = False
    max_url_duration_sec: float = 4 * 3600

    # Live streaming over WebSocket
    max_streams: int = 2  # 0 disables /stream
    stream_min_silence_ms: int = 600
    stream_max_utterance_sec: float = 20.0
    stream_partial_interval_sec: float = 1.0

    # Translation to languages other than English (text step after transcription)
    translation_model: str = DEFAULT_TRANSLATION_MODEL
    translation_device: str = "auto"  # cuda | cpu | auto

    # Speaker identification ("who is speaking")
    diarization_method: str = "segments"  # segments | pyannote
    diarization_model: str = DEFAULT_SPEAKER_MODEL
    diarization_threshold: float = 0.5
    diarization_cache_dir: str = "/cache/speaker"
    diarization_threads: int = 2

    @classmethod
    def from_env(cls) -> Settings:
        device = (_str("WHISPER_DEVICE", "cuda") or "cuda").lower()
        if device not in ("cuda", "cpu", "auto"):
            raise ValueError(f"WHISPER_DEVICE must be cuda, cpu or auto, got {device!r}")
        log_level = (_str("LOG_LEVEL", "info") or "info").lower()
        if log_level not in LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}, got {log_level!r}")
        translation_device = (_str("TRANSLATION_DEVICE", "auto") or "auto").lower()
        if translation_device not in ("cuda", "cpu", "auto"):
            raise ValueError(f"TRANSLATION_DEVICE must be cuda, cpu or auto, got {translation_device!r}")
        diarization_method = (_str("DIARIZATION_METHOD", "segments") or "segments").lower()
        if diarization_method not in DIARIZATION_METHODS:
            choices = ", ".join(DIARIZATION_METHODS)
            raise ValueError(f"DIARIZATION_METHOD must be one of {choices}, got {diarization_method!r}")

        return cls(
            model_name=_str("WHISPER_MODEL", "large-v3"),
            device=device,
            compute_type=_str("WHISPER_COMPUTE_TYPE"),
            default_language=_str("WHISPER_LANGUAGE"),
            cache_dir=_str("WHISPER_CACHE_DIR", "/cache/faster-whisper"),
            preload=_bool("WHISPER_PRELOAD", False),
            load_retry_sec=_float("WHISPER_LOAD_RETRY_SEC", 60.0, 0),
            max_upload_bytes=int(_float("MAX_UPLOAD_MB", 200, 0.001) * MB),
            max_concurrent=_int("MAX_CONCURRENT", 1, 1),
            max_queue=_int("MAX_QUEUE", 8, 0),
            api_key=_str("API_KEY"),
            expose_error_details=_bool("EXPOSE_ERROR_DETAILS", False),
            log_filenames=_bool("LOG_FILENAMES", False),
            log_level=log_level,
            host=_str("HOST", "0.0.0.0"),
            port=_int("PORT", 8000, 1),
            allow_urls=_bool("ALLOW_URLS", False),
            allow_private_urls=_bool("ALLOW_PRIVATE_URLS", False),
            max_url_duration_sec=_float("MAX_URL_DURATION_MIN", 240, 0.1) * 60,
            max_streams=_int("MAX_STREAMS", 2, 0),
            stream_min_silence_ms=_int("STREAM_MIN_SILENCE_MS", 600, 100),
            stream_max_utterance_sec=_float("STREAM_MAX_UTTERANCE_SEC", 20.0, 2.0),
            stream_partial_interval_sec=_float("STREAM_PARTIAL_INTERVAL_SEC", 1.0, 0.2),
            translation_model=_str("TRANSLATION_MODEL", DEFAULT_TRANSLATION_MODEL),
            translation_device=translation_device,
            diarization_method=diarization_method,
            diarization_model=_str("DIARIZATION_MODEL", DEFAULT_SPEAKER_MODEL),
            diarization_threshold=_float("DIARIZATION_THRESHOLD", 0.5, 0.0),
            diarization_cache_dir=_str("DIARIZATION_CACHE_DIR", "/cache/speaker"),
            diarization_threads=_int("DIARIZATION_THREADS", 2, 1),
        )
