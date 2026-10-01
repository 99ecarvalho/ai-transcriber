# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""OpenAI-compatible API: /v1/audio/transcriptions, /v1/audio/translations, /v1/models.

Mirrors the request/response shapes of OpenAI's audio endpoints so existing
OpenAI SDKs and tools work by pointing their base URL here. The `model` field
is accepted for compatibility but ignored — the server's configured model is
always used.
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile

from .engine import TranscribeOptions, Transcriber, normalize_language
from .errors import ServiceError
from .formats import OPENAI_FORMATS, openai_response

router = APIRouter(prefix="/v1")

GRANULARITIES = ("segment", "word")


def _transcriber(request: Request) -> Transcriber:
    return request.app.state.transcriber


def _validate_format(response_format: str) -> str:
    if response_format not in OPENAI_FORMATS:
        raise ServiceError(
            400,
            f"response_format must be one of {', '.join(OPENAI_FORMATS)}.",
            param="response_format",
        )
    return response_format


def _granularities(values: list[str] | None, response_format: str) -> set[str]:
    chosen = set(values or [])
    unknown = chosen - set(GRANULARITIES)
    if unknown:
        raise ServiceError(
            400,
            f"timestamp_granularities must be among {', '.join(GRANULARITIES)}.",
            param="timestamp_granularities",
        )
    if chosen and response_format != "verbose_json":
        raise ServiceError(
            400,
            "timestamp_granularities requires response_format=verbose_json.",
            param="timestamp_granularities",
        )
    return chosen or {"segment"}


@router.post("/audio/transcriptions", summary="OpenAI-compatible transcription")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float | None = Form(default=None, ge=0, le=1),
    timestamp_granularities: list[str] | None = Form(default=None, alias="timestamp_granularities[]"),
):
    t = _transcriber(request)
    response_format = _validate_format(response_format)
    granularities = _granularities(timestamp_granularities, response_format)
    opts = TranscribeOptions(
        language=normalize_language(language, t.settings.default_language),
        word_timestamps="word" in granularities,
        initial_prompt=prompt or None,
        temperature=temperature,
    )
    result = await t.transcribe_upload(file, opts)
    return openai_response(result, response_format, "segment" in granularities, "word" in granularities)


@router.post("/audio/translations", summary="OpenAI-compatible translation to English")
async def create_translation(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float | None = Form(default=None, ge=0, le=1),
):
    t = _transcriber(request)
    response_format = _validate_format(response_format)
    opts = TranscribeOptions(
        language=normalize_language(None, t.settings.default_language),
        task="translate",
        initial_prompt=prompt or None,
        temperature=temperature,
    )
    result = await t.transcribe_upload(file, opts)
    return openai_response(result, response_format, include_segments=True, include_words=False)


@router.get("/models", summary="OpenAI-compatible model list")
async def list_models(request: Request):
    t = _transcriber(request)
    ids = ["whisper-1", t.settings.model_name]
    return {
        "object": "list",
        "data": [{"id": i, "object": "model", "created": 0, "owned_by": "ai-transcriber"} for i in ids],
    }
