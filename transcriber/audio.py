# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Audio input helpers: URLs (files and live streams) and raw PCM from clients.

Everything is converted to 16 kHz mono float32, which is what Whisper expects.
"""
from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterator
from urllib.parse import urlsplit

import numpy as np

from .errors import ServiceError

SAMPLE_RATE = 16000
# FFmpeg protocols allowed when opening a URL: plain HTTP(S) and HLS playlists.
# Anything else (file:, concat:, subfile:, ...) is refused by FFmpeg itself.
URL_PROTOCOLS = "http,https,tcp,tls,hls,crypto,httpproxy"


def check_url(url: str, allow_private: bool) -> None:
    """Accepts only http(s) URLs whose host resolves to public addresses.

    Redirects and HLS segment URLs are followed by FFmpeg and not re-checked;
    keep ALLOW_URLS off on servers where that matters.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ServiceError(400, "url must be an http:// or https:// URL.", param="url")
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ServiceError(400, f"Could not resolve host {parts.hostname!r}.", param="url") from e
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ServiceError(
                400,
                "url points to a private or local network address, which this server doesn't allow "
                "(see ALLOW_PRIVATE_URLS).",
                param="url",
            )


def _open_options() -> dict[str, str]:
    return {
        "protocol_whitelist": URL_PROTOCOLS,
        "reconnect": "1",
        "reconnect_streamed": "1",
        "reconnect_delay_max": "5",
        "user_agent": "ai-transcriber",
    }


def iter_url_audio(
    url: str, stop: Callable[[], bool] | None = None, chunk_sec: float = 0.5
) -> Iterator[np.ndarray]:
    """Decodes a URL (file, Icecast/HTTP stream or HLS) into 16 kHz mono chunks.

    Blocking — run it in a worker thread. `stop()` is polled between frames.
    """
    import av

    chunk = int(chunk_sec * SAMPLE_RATE)
    pending: list[np.ndarray] = []
    pending_len = 0
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    with av.open(url, mode="r", options=_open_options(), timeout=(15.0, 30.0)) as container:
        if not container.streams.audio:
            raise ValueError("no audio stream found")
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            if stop is not None and stop():
                return
            frame.pts = None
            for out in resampler.resample(frame):
                pcm = out.to_ndarray().reshape(-1)
                pending.append(pcm)
                pending_len += len(pcm)
                if pending_len >= chunk:
                    yield np.concatenate(pending).astype(np.float32) / 32768.0
                    pending, pending_len = [], 0
        for out in resampler.resample(None):
            pending.append(out.to_ndarray().reshape(-1))
    if pending:
        yield np.concatenate(pending).astype(np.float32) / 32768.0


class _TooLong(Exception):
    def __init__(self, max_seconds: float):
        super().__init__(f"audio is longer than {max_seconds / 60:g} minutes")


def decode_url(url: str, max_seconds: float) -> np.ndarray:
    """Whole-file decode of a URL, refusing audio longer than max_seconds."""
    chunks, total = [], 0
    for chunk in iter_url_audio(url, chunk_sec=5.0):
        chunks.append(chunk)
        total += len(chunk)
        if total > max_seconds * SAMPLE_RATE:
            raise _TooLong(max_seconds)
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


class PcmConverter:
    """Raw signed 16-bit little-endian mono PCM at any rate -> 16 kHz float32."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._leftover = b""
        self._resampler = None
        if sample_rate != SAMPLE_RATE:
            import av

            self._resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

    def convert(self, data: bytes) -> np.ndarray:
        data = self._leftover + data
        usable = len(data) - (len(data) % 2)
        self._leftover = data[usable:]
        samples = np.frombuffer(data[:usable], dtype="<i2")
        if not len(samples):
            return np.zeros(0, np.float32)
        if self._resampler is None:
            return samples.astype(np.float32) / 32768.0

        import av

        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        out = [f.to_ndarray().reshape(-1) for f in self._resampler.resample(frame)]
        return (np.concatenate(out).astype(np.float32) / 32768.0) if out else np.zeros(0, np.float32)
