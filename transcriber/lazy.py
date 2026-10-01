# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Load-once resources (models) that are loaded lazily, off the event loop."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import structlog

from .errors import ServiceError

log = structlog.get_logger("transcriber")


class LazyLoader:
    """Loads a resource once, on first use, in a worker thread.

    The load runs in its own task, so a request that is cancelled while
    waiting doesn't abandon a half-finished load; whoever asks next gets the
    same load (or its result).

    After a failed load, further attempts are refused for `retry_sec` so a
    broken setup (no GPU, bad model name) fails fast instead of every request
    retrying a multi-minute load.
    """

    def __init__(self, what: str, key: str, retry_sec: float, load: Callable[[], Any]):
        self.what = what  # human-readable, e.g. "model", "translation model"
        self.key = key  # for log event names, e.g. "model" -> transcriber.loading_model
        self.retry_sec = retry_sec
        self._load_fn = load
        self._value: Any = None
        self._load_task: asyncio.Task | None = None
        self.last_error: str | None = None
        self._last_error_at: float | None = None

    def log_fields(self) -> dict[str, Any]:
        return {}

    @property
    def loaded(self) -> bool:
        return self._value is not None

    @property
    def loading(self) -> bool:
        return self._load_task is not None and not self._load_task.done()

    @property
    def state(self) -> str:
        if self._value is not None:
            return "ready"
        if self.loading:
            return "loading"
        if self.last_error is not None:
            return "failed"
        return "not_loaded"

    def _raise_if_cooling_down(self) -> None:
        if self._last_error_at is None:
            return
        remaining = self.retry_sec - (time.monotonic() - self._last_error_at)
        if remaining > 0:
            raise ServiceError(
                503,
                f"{self.what[0].upper()}{self.what[1:]} is unavailable (last load failed).",
                internal=self.last_error,
                retry_after=max(1, int(remaining + 0.999)),
            )

    async def get(self) -> Any:
        if self._value is not None:
            return self._value
        if self._load_task is None:
            self._raise_if_cooling_down()
            self._load_task = asyncio.ensure_future(self._load())
            # Retrieve the outcome even if every waiter was cancelled, so a
            # failure is never reported as "exception was never retrieved".
            self._load_task.add_done_callback(lambda t: t.cancelled() or t.exception())
        return await asyncio.shield(self._load_task)

    async def _load(self) -> Any:
        log.info(f"transcriber.loading_{self.key}", **self.log_fields())
        t0 = time.monotonic()
        try:
            # Downloads + loads can take minutes — run them off the event loop
            # so /health and other requests keep responding meanwhile.
            value = await asyncio.to_thread(self._load_fn)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self._last_error_at = time.monotonic()
            self._load_task = None
            log.exception(f"transcriber.{self.key}_load_failed")
            raise ServiceError(
                503,
                f"Failed to load {self.what}.",
                internal=self.last_error,
                retry_after=max(1, int(self.retry_sec)),
            ) from e

        self._value = value
        self.last_error = None
        self._last_error_at = None
        log.info(f"transcriber.{self.key}_loaded", elapsed_sec=round(time.monotonic() - t0, 2))
        return value
