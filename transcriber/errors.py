# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Error type shared by both APIs, rendered in each API's own format.

Native endpoints answer FastAPI-style: {"detail": "..."}.
OpenAI-compatible endpoints (/v1/...) answer {"error": {"message", "type", "param", "code"}}.
"""
from __future__ import annotations

from fastapi.responses import JSONResponse

OPENAI_PREFIX = "/v1/"


class ServiceError(Exception):
    """An error with a public message and optional internal details.

    `internal` is only sent to clients when EXPOSE_ERROR_DETAILS is on; it is
    always logged.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        internal: str | None = None,
        param: str | None = None,
        retry_after: int | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.internal = internal
        self.param = param
        self.retry_after = retry_after


def _openai_type(status_code: int) -> str:
    if status_code == 401:
        return "authentication_error"
    if status_code < 500:
        return "invalid_request_error"
    return "server_error"


def error_response(
    path: str,
    status_code: int,
    message: str,
    *,
    param: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    if path.startswith(OPENAI_PREFIX):
        body = {
            "error": {
                "message": message,
                "type": _openai_type(status_code),
                "param": param,
                "code": None,
            }
        }
    else:
        body = {"detail": message}
    return JSONResponse(body, status_code=status_code, headers=headers)


def render_service_error(path: str, exc: ServiceError, expose_details: bool) -> JSONResponse:
    message = exc.message
    if expose_details and exc.internal:
        message = f"{message}: {exc.internal}"
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
    return error_response(path, exc.status_code, message, param=exc.param, headers=headers)
