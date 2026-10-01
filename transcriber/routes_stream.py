# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Live transcription: WebSocket /stream and the browser page /live.

Protocol (JSON text messages, plus binary audio frames):

1. Client connects (optionally offering subprotocol "transcriber.v1") and sends
   {"type": "start", "language": "en", "task": "transcribe", "translate_to": "pt",
    "diarize": true, "sample_rate": 48000, "url": null}
   All fields are optional. With "url", the server pulls the audio itself;
   otherwise the client sends binary frames of signed 16-bit little-endian
   mono PCM at `sample_rate` (default 16000).
2. Server answers {"type": "ready", ...} once the models are loaded, then
   streams "partial", "final", "language" and "error" events.
3. Client sends {"type": "stop"} (or the pulled stream ends): the server
   finalises what's left, sends {"type": "end", "duration": ...} and closes.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from importlib import resources

import structlog
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .audio import PcmConverter
from .engine import Transcriber, normalize_language
from .errors import ServiceError
from .streaming import QUEUE_CHUNKS, StreamConfig, StreamSession, pump_url

log = structlog.get_logger("transcriber")
router = APIRouter()

SUBPROTOCOL = "transcriber.v1"
START_TIMEOUT_SEC = 30


@router.get("/live", include_in_schema=False)
async def live_page(request: Request):
    page = resources.files("transcriber").joinpath("static/live.html").read_text(encoding="utf-8")
    return HTMLResponse(page)


def _parse_start(message: dict, transcriber: Transcriber) -> tuple[StreamConfig, str | None, int]:
    if message.get("type") != "start":
        raise ServiceError(400, 'The first message must be {"type": "start", ...}.')
    task = message.get("task") or "transcribe"
    if task not in ("transcribe", "translate"):
        raise ServiceError(400, "task must be transcribe or translate.", param="task")
    sample_rate = int(message.get("sample_rate") or 16000)
    if not 8000 <= sample_rate <= 192000:
        raise ServiceError(400, "sample_rate must be between 8000 and 192000.", param="sample_rate")
    if message.get("encoding", "pcm_s16le") != "pcm_s16le":
        raise ServiceError(400, "encoding must be pcm_s16le.", param="encoding")
    config = StreamConfig(
        language=normalize_language(message.get("language"), transcriber.settings.default_language),
        task=task,
        translate_to=normalize_language(message.get("translate_to"), None, param="translate_to"),
        diarize=bool(message.get("diarize")),
        initial_prompt=(message.get("initial_prompt") or None),
    )
    url = (message.get("url") or "").strip() or None
    return config, url, sample_rate


@router.websocket("/stream")
async def stream(ws: WebSocket):
    t: Transcriber = ws.app.state.transcriber
    subprotocol = SUBPROTOCOL if SUBPROTOCOL in ws.scope.get("subprotocols", []) else None
    await ws.accept(subprotocol=subprotocol)

    async def fail(message: str, code: int = 1008) -> None:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": message})
            await ws.close(code=code)

    if t.settings.max_streams == 0:
        await fail("Live streaming is disabled on this server (MAX_STREAMS=0).")
        return
    if t.active_streams >= t.settings.max_streams:
        await fail("Too many live streams, try again later.", code=1013)
        return

    t.active_streams += 1
    stop = threading.Event()
    tasks: list[asyncio.Task] = []
    try:
        try:
            first = await asyncio.wait_for(ws.receive_json(), START_TIMEOUT_SEC)
            config, url, sample_rate = _parse_start(first, t)
            if url:
                await asyncio.to_thread(t.check_url, url)
        except asyncio.TimeoutError:
            await fail("No start message received.")
            return
        except ServiceError as e:
            await fail(e.message)
            return
        except (ValueError, TypeError):
            await fail("Invalid start message.")
            return

        async def send(event: dict) -> None:
            await ws.send_json(event)

        session = StreamSession(t, config, send, vad=t.stream_vad)
        if not t.models.loaded or (config.diarize and not t.speakers.loaded) or (
            config.translate_to and not t.translator.loaded
        ):
            await send({"type": "loading", "message": "Loading models; the first start can take minutes."})
        try:
            await session.prepare()
        except ServiceError as e:
            await fail(e.message, code=1011)
            return

        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_CHUNKS)
        session_task = asyncio.create_task(session.run(queue))
        tasks.append(session_task)
        if url:
            tasks.append(asyncio.create_task(pump_url(url, queue, stop)))
        receiver = asyncio.create_task(_receive(ws, queue, None if url else PcmConverter(sample_rate)))
        tasks.append(receiver)
        log.info("transcriber.stream_started", source="url" if url else "push", diarize=config.diarize,
                 translate_to=config.translate_to, language=config.language)

        done, _ = await asyncio.wait({session_task, receiver, *tasks}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task is not session_task and task.exception() is not None:
                if isinstance(task.exception(), WebSocketDisconnect):
                    return
                await send({"type": "error", "message": f"Audio source failed: {task.exception()}"})
                await queue.put(None)
        if receiver in done and receiver.result() == "disconnect":
            return  # client went away; nothing left to send
        await session_task  # flush what's left, then "end"
        with contextlib.suppress(Exception):
            await ws.close()
    except WebSocketDisconnect:
        pass
    except ServiceError as e:
        await fail(e.message, code=1011)
    except Exception:
        log.exception("transcriber.stream_failed")
        await fail("Stream failed.", code=1011)
    finally:
        stop.set()
        for task in tasks:
            task.cancel()
        t.active_streams -= 1
        log.info("transcriber.stream_ended")


async def _receive(ws: WebSocket, queue: asyncio.Queue, converter: PcmConverter | None) -> str:
    """Moves client messages into the session queue. Returns why it stopped."""
    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            return "disconnect"
        if message.get("bytes") is not None:
            if converter is None:
                continue  # url mode: the server is the audio source
            audio = converter.convert(message["bytes"])
            if len(audio):
                await queue.put(audio)
        elif message.get("text") is not None:
            try:
                kind = json.loads(message["text"]).get("type")
            except (ValueError, AttributeError):
                kind = None
            if kind == "stop":
                await queue.put(None)
                return "stop"
