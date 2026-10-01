# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

import io
import math
import struct
import threading
import wave
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from transcriber.app import create_app
from transcriber.config import Settings


class FakeModel:
    """Stands in for faster_whisper.WhisperModel; real audio decoding still runs."""

    def __init__(self):
        self.calls: list[dict] = []
        self.gate: threading.Event | None = None  # set to block transcribe() until released

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        if self.gate is not None:
            self.gate.wait(timeout=10)
        words = None
        if kwargs.get("word_timestamps"):
            words = [
                SimpleNamespace(start=0.0, end=0.4, word=" Hello", probability=0.98),
                SimpleNamespace(start=0.5, end=0.9, word=" world.", probability=0.91),
            ]
        segments = [
            SimpleNamespace(
                id=1, seek=0, start=0.0, end=1.0, text=" Hello world.", tokens=[50364, 2425, 1002],
                temperature=0.0, avg_logprob=-0.2, compression_ratio=1.1, no_speech_prob=0.01, words=words,
            )
        ]
        info = SimpleNamespace(language=kwargs.get("language") or "en", language_probability=0.987654,
                               duration=1.0)
        return iter(segments), info


def make_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = int(seconds * rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 10))) for i in range(frames)))
    return buf.getvalue()


@pytest.fixture
def wav() -> bytes:
    return make_wav()


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def base_settings(tmp_path) -> Settings:
    return Settings(device="cpu", cache_dir=str(tmp_path / "cache"))


@pytest.fixture
def make_client(base_settings, fake_model):
    """make_client(**settings_overrides, loader=...) -> TestClient"""

    def _make(loader=None, **overrides) -> TestClient:
        settings = replace(base_settings, **overrides)
        app = create_app(settings, loader or (lambda _manager: fake_model))
        return TestClient(app)

    return _make


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


@pytest.fixture
def anyio_backend():
    return "asyncio"
