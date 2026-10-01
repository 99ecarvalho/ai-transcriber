# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""ASGI middleware that rejects requests before their body is read."""
from __future__ import annotations

import base64
import binascii
import secrets

from starlette.exceptions import HTTPException

from .errors import error_response

PUBLIC_PATHS = {"/health", "/ready", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json", "/live"}
WS_KEY_PREFIX = "bearer."


class ApiKeyMiddleware:
    """Requires `Authorization: Bearer <API_KEY>` on every path except PUBLIC_PATHS.

    Runs before body parsing, so unauthenticated uploads are refused without
    being received. WebSockets also accept the key as a subprotocol,
    "bearer.<base64url(API_KEY)>", because browsers can't set headers on them
    (and a query string would end up in access logs).
    """

    def __init__(self, app, api_key: str):
        self.app = app
        self.expected = api_key.encode()

    def _authorized(self, header: bytes) -> bool:
        scheme, _, token = header.partition(b" ")
        # The scheme name is case-insensitive (RFC 9110); the key itself is not.
        return scheme.lower() == b"bearer" and secrets.compare_digest(token.strip(), self.expected)

    def _authorized_ws(self, scope) -> bool:
        for name, value in scope["headers"]:
            if name == b"authorization" and self._authorized(value):
                return True
        for protocol in scope.get("subprotocols", []):
            if protocol.startswith(WS_KEY_PREFIX):
                encoded = protocol[len(WS_KEY_PREFIX) :]
                try:
                    key = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
                except (binascii.Error, ValueError):
                    continue
                if secrets.compare_digest(key, self.expected):
                    return True
        return False

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            if self._authorized_ws(scope):
                await self.app(scope, receive, send)
            else:
                await send({"type": "websocket.close", "code": 1008})  # handshake answered with 403
            return
        if scope["type"] != "http" or scope["path"] in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        provided = b""
        for name, value in scope["headers"]:
            if name == b"authorization":
                provided = value
                break
        if not self._authorized(provided):
            response = error_response(
                scope["path"],
                401,
                "Missing or invalid API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


class _BodyTooLarge(HTTPException):
    # An HTTPException so FastAPI's body parser re-raises it instead of turning
    # it into a generic 400; the app's HTTPException handler renders it.
    def __init__(self, message: str):
        super().__init__(status_code=413, detail=message)


class MaxBodySizeMiddleware:
    """Rejects request bodies larger than max_bytes with a 413.

    Checks Content-Length up front, and also counts the bytes actually received
    so chunked uploads (or a lying Content-Length) can't bypass the limit.
    The multipart parser spools uploads to disk, so this also bounds disk usage.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    def _message(self) -> str:
        return f"Request body exceeds the {self.max_bytes / (1024 * 1024):g} MB limit."

    def _too_large(self, path: str):
        return error_response(path, 413, self._message())

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
                    await self._too_large(scope["path"])(scope, receive, send)
                    return
                break

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge(self._message())
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if response_started:
                raise
            await self._too_large(scope["path"])(scope, receive, send)
