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

    @classmethod
    def from_env(cls) -> Settings:
        device = (_str("WHISPER_DEVICE", "cuda") or "cuda").lower()
        if device not in ("cuda", "cpu", "auto"):
            raise ValueError(f"WHISPER_DEVICE must be cuda, cpu or auto, got {device!r}")
        log_level = (_str("LOG_LEVEL", "info") or "info").lower()
        if log_level not in LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}, got {log_level!r}")

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
        )
