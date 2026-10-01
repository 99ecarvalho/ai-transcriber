# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Response bodies for the different output formats."""
from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse, PlainTextResponse, Response

from .engine import Segment, TranscriptionResult

NATIVE_FORMATS = ("json", "text", "srt", "vtt")
OPENAI_FORMATS = ("json", "text", "srt", "verbose_json", "vtt")


def _timestamp(seconds: float, decimal_marker: str) -> str:
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{ms:03d}"


def to_srt(segments: list[Segment]) -> str:
    blocks = [
        f"{i}\n{_timestamp(s.start, ',')} --> {_timestamp(s.end, ',')}\n{s.text.strip()}\n"
        for i, s in enumerate(segments, start=1)
    ]
    return "\n".join(blocks)


def to_vtt(segments: list[Segment]) -> str:
    blocks = [
        f"{_timestamp(s.start, '.')} --> {_timestamp(s.end, '.')}\n{s.text.strip()}\n" for s in segments
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)


def text_response(result: TranscriptionResult, response_format: str) -> Response | None:
    """Shared non-JSON formats; returns None for JSON formats."""
    if response_format == "text":
        return PlainTextResponse(result.text + "\n")
    if response_format == "srt":
        return PlainTextResponse(to_srt(result.segments))
    if response_format == "vtt":
        return PlainTextResponse(to_vtt(result.segments), media_type="text/vtt")
    return None


# ---------- Native API ----------

def native_json(result: TranscriptionResult) -> dict[str, Any]:
    segments = []
    for s in result.segments:
        seg: dict[str, Any] = {"start": s.start, "end": s.end, "text": s.text}
        if s.words is not None:
            seg["words"] = [
                {"start": w.start, "end": w.end, "word": w.word, "probability": round(w.probability, 3)}
                for w in s.words
            ]
        segments.append(seg)
    return {
        "text": result.text,
        "language": result.language,
        "language_probability": round(result.language_probability, 3),
        "audio_duration_sec": round(result.duration, 2),
        "elapsed_ms": result.elapsed_ms,
        "segments": segments,
    }


def native_response(result: TranscriptionResult, response_format: str) -> Response:
    return text_response(result, response_format) or JSONResponse(native_json(result))


# ---------- OpenAI-compatible API ----------

def openai_verbose_json(result: TranscriptionResult, include_segments: bool, include_words: bool) -> dict:
    body: dict[str, Any] = {
        "task": result.task,
        "language": result.language,
        "duration": round(result.duration, 2),
        "text": result.text,
    }
    if include_segments:
        body["segments"] = [
            {
                "id": s.id,
                "seek": s.seek,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "tokens": s.tokens,
                "temperature": s.temperature,
                "avg_logprob": s.avg_logprob,
                "compression_ratio": s.compression_ratio,
                "no_speech_prob": s.no_speech_prob,
            }
            for s in result.segments
        ]
    if include_words:
        body["words"] = [
            {"word": w.word, "start": w.start, "end": w.end} for s in result.segments for w in (s.words or [])
        ]
    return body


def openai_response(
    result: TranscriptionResult, response_format: str, include_segments: bool, include_words: bool
) -> Response:
    text = text_response(result, response_format)
    if text is not None:
        return text
    if response_format == "verbose_json":
        return JSONResponse(openai_verbose_json(result, include_segments, include_words))
    return JSONResponse({"text": result.text})
