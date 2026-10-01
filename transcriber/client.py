# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Command-line client for live transcription (WebSocket /stream).

  python -m transcriber.client --url https://example.com/live.m3u8
  python -m transcriber.client --file meeting.mp3            # pushed at real-time speed
  ffmpeg -f pulse -i default -ac 1 -ar 16000 -f s16le - | python -m transcriber.client --stdin

Finished lines go to stdout; the in-progress line is shown on stderr.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import statistics
import sys
import time

CHUNK_SEC = 0.1


def _ws_url(server: str) -> str:
    server = server.rstrip("/")
    if server.startswith("https://"):
        server = "wss://" + server[len("https://") :]
    elif server.startswith("http://"):
        server = "ws://" + server[len("http://") :]
    elif not server.startswith(("ws://", "wss://")):
        server = "ws://" + server
    return server + "/stream"


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


async def _send_file(ws, path: str, speed: float, clock: dict) -> None:
    import av
    import numpy as np

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    pending = b""
    chunk_bytes = int(16000 * CHUNK_SEC) * 2
    sent = 0
    clock["t0"] = time.monotonic()
    with av.open(path) as container:
        for frame in container.decode(audio=0):
            frame.pts = None
            for out in resampler.resample(frame):
                pending += np.ascontiguousarray(out.to_ndarray().reshape(-1)).tobytes()
                while len(pending) >= chunk_bytes:
                    await ws.send(pending[:chunk_bytes])
                    pending = pending[chunk_bytes:]
                    sent += chunk_bytes
                    if speed > 0:  # pace like a live source
                        due = clock["t0"] + (sent / 2 / 16000) / speed
                        await asyncio.sleep(max(0.0, due - time.monotonic()))
    if pending:
        await ws.send(pending)
    await ws.send(json.dumps({"type": "stop"}))


async def _send_stdin(ws, rate: int) -> None:
    loop = asyncio.get_running_loop()
    chunk_bytes = int(rate * CHUNK_SEC) * 2
    reader = sys.stdin.buffer
    while True:
        data = await loop.run_in_executor(None, reader.read, chunk_bytes)
        if not data:
            break
        await ws.send(data)
    await ws.send(json.dumps({"type": "stop"}))


async def run(args: argparse.Namespace) -> int:
    from websockets.asyncio.client import connect

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    start = {
        "type": "start",
        "language": args.language,
        "task": args.task,
        "translate_to": args.translate_to,
        "diarize": args.diarize,
        "url": args.url,
        "sample_rate": args.rate if args.stdin else 16000,
    }
    clock: dict = {}
    latencies: list[float] = []
    partial_shown = False
    tty = sys.stderr.isatty()

    async with connect(_ws_url(args.server), additional_headers=headers, subprotocols=["transcriber.v1"],
                       max_size=None, open_timeout=30) as ws:
        await ws.send(json.dumps(start))
        sender: asyncio.Task | None = None
        loop = asyncio.get_running_loop()

        def on_interrupt() -> None:
            # First Ctrl+C: stop sending and let the server finish the last line.
            # A second one falls through to the default handler and quits.
            loop.remove_signal_handler(signal.SIGINT)
            if sender is not None:
                sender.cancel()
            print("\n· finishing (Ctrl+C again to quit)", file=sys.stderr)
            asyncio.ensure_future(ws.send(json.dumps({"type": "stop"})))

        with contextlib.suppress(NotImplementedError):  # not available on Windows
            loop.add_signal_handler(signal.SIGINT, on_interrupt)
        async for raw in ws:
            event = json.loads(raw)
            kind = event.get("type")
            if args.json:
                print(json.dumps(event, ensure_ascii=False), flush=True)
            if kind == "ready" and sender is None:
                if args.file:
                    sender = asyncio.create_task(_send_file(ws, args.file, args.speed, clock))
                elif args.stdin:
                    sender = asyncio.create_task(_send_stdin(ws, args.rate))
                if not args.json:
                    print("· listening (Ctrl+C to stop)", file=sys.stderr)
            elif kind == "loading" and not args.json:
                print(f"· {event['message']}", file=sys.stderr)
            elif kind == "partial" and not args.json and tty:
                print(f"\r\033[K\033[2m{event['text'][-150:]}\033[0m", end="", file=sys.stderr, flush=True)
                partial_shown = bool(event["text"])
            elif kind == "final":
                if "t0" in clock and args.speed > 0:
                    latencies.append(time.monotonic() - (clock["t0"] + event["end"] / args.speed))
                if not args.json:
                    if partial_shown:
                        print("\r\033[K", end="", file=sys.stderr, flush=True)
                        partial_shown = False
                    speaker = f"{event['speaker']}: " if event.get("speaker") else ""
                    print(f"[{_fmt(event['start'])}] {speaker}{event['text']}", flush=True)
                    if event.get("translation"):
                        print(f"        → {event['translation']}", flush=True)
            elif kind == "language" and not args.json:
                print(f"· language: {event['language']} ({event['probability']:.0%})", file=sys.stderr)
            elif kind == "error":
                print(f"error: {event['message']}", file=sys.stderr)
            elif kind == "end":
                break
        if sender is not None and not sender.done():
            sender.cancel()

    if latencies and not args.json:
        latencies.sort()
        p90 = latencies[min(len(latencies) - 1, int(0.9 * len(latencies)))]
        print(
            f"· latency from end of speech to final text: median {statistics.median(latencies):.2f}s, "
            f"p90 {p90:.2f}s, max {latencies[-1]:.2f}s over {len(latencies)} lines",
            file=sys.stderr,
        )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m transcriber.client", description=__doc__.split("\n\n")[0])
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="audio URL for the server to pull (podcast, Icecast, HLS)")
    source.add_argument("--file", help="local audio file, sent like a live source")
    source.add_argument("--stdin", action="store_true", help="raw s16le mono PCM on stdin")
    p.add_argument("--server", default=os.environ.get("TRANSCRIBER_URL", "http://localhost:8000"))
    p.add_argument("--api-key", default=os.environ.get("API_KEY") or None)
    p.add_argument("--language", default=None, help="spoken language (default: detect)")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p.add_argument("--translate-to", default=None, help="also translate each line into this language")
    p.add_argument("--diarize", action="store_true", help="label who is speaking")
    p.add_argument("--rate", type=int, default=16000, help="sample rate of --stdin audio")
    p.add_argument("--speed", type=float, default=1.0,
                   help="--file pace: 1 = real time, 0 = as fast as possible")
    p.add_argument("--json", action="store_true", help="print raw events as JSON lines")
    args = p.parse_args()
    try:
        sys.exit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        sys.exit(130)
    except OSError as e:
        sys.exit(f"error: cannot connect to {args.server}: {e}")
    except Exception as e:  # websockets' handshake errors, e.g. a 403 for a wrong API key
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 403:
            sys.exit("error: the server refused the connection (HTTP 403) — wrong or missing API key?")
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
