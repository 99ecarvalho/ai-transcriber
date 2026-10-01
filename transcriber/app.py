# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""FastAPI application factory.

Run with `python -m transcriber`, or `uvicorn transcriber.app:create_app --factory`.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, routes_native, routes_openai, routes_stream
from .config import Settings
from .engine import ModelManager, Transcriber
from .errors import OPENAI_PREFIX, ServiceError, error_response, render_service_error
from .middleware import ApiKeyMiddleware, MaxBodySizeMiddleware

log = structlog.get_logger("transcriber")


def configure_logging(level: str) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    structlog.contextvars.bind_contextvars(component="transcriber")


def create_app(
    settings: Settings | None = None,
    loader: Callable[[ModelManager], Any] | None = None,
    translation_loader: Callable[[], Any] | None = None,
    speaker_loader: Callable[[], Any] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    transcriber = Transcriber(settings, loader, translation_loader, speaker_loader)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        preload_task = None
        if settings.preload:
            # Load in the background so /health answers immediately; /ready flips once loaded.
            async def _preload():
                with contextlib.suppress(ServiceError):
                    await transcriber.models.get()

            preload_task = asyncio.create_task(_preload())
        yield
        if preload_task and not preload_task.done():
            preload_task.cancel()

    app = FastAPI(
        title="ai-transcriber",
        version=__version__,
        description="Speech-to-text with faster-whisper. Copyright (c) 2026 Eduardo Correia.",
        license_info={"name": "LGPL-3.0-or-later", "identifier": "LGPL-3.0-or-later"},
        lifespan=lifespan,
    )
    app.state.transcriber = transcriber
    app.include_router(routes_native.router)
    app.include_router(routes_openai.router)
    app.include_router(routes_stream.router)

    # Order matters: the last one added runs first, so auth rejects before any body is read.
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_upload_bytes)
    if settings.api_key:
        app.add_middleware(ApiKeyMiddleware, api_key=settings.api_key)

    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError):
        return render_service_error(request.url.path, exc, settings.expose_error_details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        return error_response(request.url.path, exc.status_code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        if not request.url.path.startswith(OPENAI_PREFIX):
            return await request_validation_exception_handler(request, exc)
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(p) for p in first.get("loc", ()) if p not in ("body", "query")]
        param = loc[0] if loc else None
        message = f"{param}: {first.get('msg', 'invalid value')}" if param else "Invalid request."
        return error_response(request.url.path, 400, message, param=param)

    log.info(
        "transcriber.configured",
        model=settings.model_name,
        device=transcriber.models.device,
        compute_type=transcriber.models.compute_type,
        max_concurrent=settings.max_concurrent,
        max_queue=settings.max_queue,
        max_upload_bytes=settings.max_upload_bytes,
        auth=bool(settings.api_key),
        preload=settings.preload,
        max_streams=settings.max_streams,
        allow_urls=settings.allow_urls,
        translation_model=settings.translation_model,
        diarization=f"{settings.diarization_method}/{settings.diarization_model}",
    )
    return app
