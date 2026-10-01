# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native API: /health, /ready, /transcribe.

/transcribe's defaults produce exactly the original response, so existing
clients are unaffected by the optional parameters added later.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .engine import TranscribeOptions, Transcriber, normalize_language
from .errors import ServiceError
from .formats import native_response

router = APIRouter()


def _transcriber(request: Request) -> Transcriber:
    return request.app.state.transcriber


@router.get("/health", summary="Liveness: the process is up (does not load the model)")
async def health(request: Request):
    t = _transcriber(request)
    return {
        "status": "ok",
        "model": t.settings.model_name,
        "device": t.models.device,
        "compute_type": t.models.compute_type,
        "model_loaded": t.models.loaded,
        "model_state": t.models.state,
        "max_upload_bytes": t.settings.max_upload_bytes,
        "version": __version__,
    }


@router.get("/ready", summary="Readiness: 200 once the model is loaded, 503 otherwise")
async def ready(request: Request):
    t = _transcriber(request)
    body = {"status": t.models.state, "model": t.settings.model_name}
    if t.models.last_error and t.settings.expose_error_details:
        body["error"] = t.models.last_error
    return JSONResponse(body, status_code=200 if t.models.loaded else 503)


@router.post("/transcribe", summary="Transcribe an audio file or URL")
async def transcribe(
    request: Request,
    file: UploadFile | None = File(default=None, description="Audio file (any format supported by ffmpeg)"),
    language: str | None = Form(default=None, description="pt|en|es|... or empty for auto-detect"),
    vad_filter: bool = Form(default=True, description="Voice activity detection (filters silence)"),
    beam_size: int = Form(default=5, ge=1, le=20),
    response_format: Literal["json", "text", "srt", "vtt"] = Form(default="json"),
    word_timestamps: bool = Form(default=False, description="Add per-word timings to each segment"),
    initial_prompt: str | None = Form(default=None, description="Text to steer vocabulary, names, style"),
    task: Literal["transcribe", "translate"] = Form(
        default="transcribe", description="translate = any language to English"
    ),
    url: str | None = Form(default=None, description="Audio URL to fetch instead of uploading (ALLOW_URLS)"),
    diarize: bool = Form(default=False, description="Label who is speaking in each segment"),
    num_speakers: int | None = Form(
        default=None, ge=1, le=32, description="Known number of speakers (with diarize)"
    ),
    translate_to: str | None = Form(default=None, description="Also translate the text into this language"),
):
    if file is None and not url:
        # Same 422 body as when `file` was a required field, so clients see no change.
        raise RequestValidationError(
            [{"type": "missing", "loc": ("body", "file"), "msg": "Field required", "input": None}]
        )
    if file is not None and url:
        raise ServiceError(400, "Send either file or url, not both.", param="url")

    t = _transcriber(request)
    opts = TranscribeOptions(
        language=normalize_language(language, t.settings.default_language),
        task=task,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt or None,
        diarize=diarize,
        num_speakers=num_speakers if diarize else None,
        translate_to=normalize_language(translate_to, None, param="translate_to"),
    )
    if url:
        result = await t.transcribe_source(opts, url=url.strip())
    else:
        result = await t.transcribe_upload(file, opts)
    return native_response(result, response_format)
