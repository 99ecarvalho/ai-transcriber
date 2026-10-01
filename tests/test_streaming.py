# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Live transcription over WebSocket, with a fake model and an amplitude-based VAD."""
from __future__ import annotations

import base64
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from transcriber.app import create_app

RATE = 16000


class StreamFakeModel:
    """One segment per started second of audio, text naming the clip length."""

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        seconds = len(audio) / RATE
        n = max(1, int(np.ceil(seconds)))
        segments = [
            SimpleNamespace(id=i + 1, seek=0, start=float(i), end=float(min(i + 1, seconds)),
                            text=f" part{i + 1} of {seconds:.1f}s.", tokens=[], temperature=0.0,
                            avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=0.0, words=None)
            for i in range(n)
        ]
        language = kwargs.get("language") or "en"
        info = SimpleNamespace(language=language, language_probability=0.9, duration=seconds)
        return iter(segments), info


def amplitude_vad(audio):
    """Speech wherever 32 ms windows are louder than a threshold."""
    win = 512
    loud = [np.abs(audio[i:i + win]).mean() > 0.02 for i in range(0, len(audio) - win + 1, win)]
    spans, start = [], None
    for k, is_loud in enumerate(loud):
        if is_loud and start is None:
            start = k * win
        if not is_loud and start is not None:
            spans.append((start, k * win))
            start = None
    if start is not None:
        spans.append((start, len(loud) * win))
    return spans


class FakeSpeakers:
    """The 'voice' is the tone's loudness: 0.3 -> one speaker, 0.6 -> another."""

    def embed(self, audio):
        level = np.abs(audio).mean()
        return np.array([1.0, 0.0]) if level < 0.3 else np.array([0.0, 1.0])


class FakeTranslator:
    def supports(self, code):
        return True

    def translate(self, texts, source, target):
        return [f"[{target}]{t.strip()}" for t in texts]


def pcm(seconds, amplitude):
    t = np.arange(int(seconds * RATE))
    return (np.sin(t / 3) * amplitude * 32767).astype("<i2").tobytes()


def speech_then_silence(*bursts):
    """bytes for: burst, 1 s of silence, burst, 1 s of silence, ..."""
    return b"".join(pcm(seconds, amp) + pcm(1.0, 0.0) for seconds, amp in bursts)


@pytest.fixture
def stream_app(base_settings):
    def make(**overrides):
        model = StreamFakeModel()
        settings = replace(base_settings, stream_partial_interval_sec=0.5, **overrides)
        app = create_app(settings, lambda _m: model, lambda: FakeTranslator(), lambda: FakeSpeakers())
        app.state.transcriber.stream_vad = amplitude_vad
        app.state.model = model
        return app

    return make


def run_stream(client, start, audio, chunk=3200, **connect):
    events = []
    with client.websocket_connect("/stream", **connect) as ws:
        ws.send_json({"type": "start", **start})
        first = ws.receive_json()
        if first["type"] == "loading":
            first = ws.receive_json()
        events.append(first)
        assert first["type"] == "ready", first
        for i in range(0, len(audio), chunk):
            ws.send_bytes(audio[i:i + chunk])
        ws.send_json({"type": "stop"})
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in ("end", "error"):
                break
    return events


def finals(events):
    return [e for e in events if e["type"] == "final"]


def test_stream_two_utterances(stream_app):
    client = TestClient(stream_app())
    events = run_stream(client, {"language": "en"}, speech_then_silence((1.5, 0.3), (2.0, 0.3)))
    lines = finals(events)
    assert len(lines) >= 2
    assert lines[0]["start"] < 1.0 and lines[-1]["start"] >= 2.0  # second utterance starts after the gap
    assert [e["id"] for e in lines] == list(range(1, len(lines) + 1))
    assert events[-1] == {"type": "end", "duration": 5.5}
    assert all("speaker" not in e and "translation" not in e for e in lines)


def test_stream_detects_and_keeps_language(stream_app):
    app = stream_app()
    events = run_stream(TestClient(app), {}, speech_then_silence((1.5, 0.3), (1.5, 0.3)))
    detected = [e for e in events if e["type"] == "language"]
    assert detected == [{"type": "language", "language": "en", "probability": 0.9}]
    final_calls = [c for c in app.state.model.calls if c["beam_size"] == 5]
    assert final_calls[0]["language"] is None and final_calls[-1]["language"] == "en"


def test_stream_speakers_and_translation(stream_app):
    client = TestClient(stream_app())
    events = run_stream(client, {"language": "en", "diarize": True, "translate_to": "pt"},
                        speech_then_silence((3.5, 0.1), (3.5, 0.6), (3.5, 0.1)))
    lines = finals(events)

    def speaker_at(t):  # speaker of the last line that started within the burst around t
        return [e["speaker"] for e in lines if e["start"] < t][-1]

    # Bursts: 0-3.5 s (voice A), 4.5-8 s (voice B), 9-12.5 s (voice A). A new voice is
    # recognised once enough of its turn has been heard, so check each turn's last line.
    assert speaker_at(3.5) == "SPEAKER_1"
    assert speaker_at(8.0) == "SPEAKER_2"
    assert speaker_at(13.0) == "SPEAKER_1"
    assert all(e["translation"] == f"[pt]{e['text']}" for e in lines)


def test_stream_partials_while_speaking(stream_app):
    client = TestClient(stream_app())
    events = run_stream(client, {"language": "en"}, speech_then_silence((3.0, 0.3)), chunk=1600)
    kinds = [e["type"] for e in events]
    assert "partial" in kinds and kinds.index("partial") < kinds.index("final")


def test_long_monologue_is_cut(stream_app):
    client = TestClient(stream_app(stream_max_utterance_sec=3.0))
    events = run_stream(client, {"language": "en"}, pcm(8.0, 0.3), chunk=1600)
    lines = finals(events)
    assert len(lines) >= 3
    assert lines[1]["start"] < 6.0  # finals were produced during the monologue, not only at the end


def test_stream_rejects_bad_start(stream_app):
    with TestClient(stream_app()).websocket_connect("/stream") as ws:
        ws.send_json({"type": "start", "language": "xx"})
        assert ws.receive_json()["type"] == "error"


def test_stream_url_needs_allow_urls(stream_app):
    with TestClient(stream_app()).websocket_connect("/stream") as ws:
        ws.send_json({"type": "start", "url": "https://example.com/live.m3u8"})
        assert "disabled" in ws.receive_json()["message"]


def test_streaming_can_be_disabled(stream_app):
    with TestClient(stream_app(max_streams=0)).websocket_connect("/stream") as ws:
        assert "disabled" in ws.receive_json()["message"]


def test_stream_auth(stream_app):
    client = TestClient(stream_app(api_key="s3cret/+="))
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/stream") as ws:
            ws.receive_json()
    audio = speech_then_silence((1.5, 0.3))
    by_header = run_stream(client, {"language": "en"}, audio, headers={"Authorization": "Bearer s3cret/+="})
    assert by_header[-1]["type"] == "end"
    token = base64.urlsafe_b64encode(b"s3cret/+=").decode().rstrip("=")
    protocols = ["transcriber.v1", f"bearer.{token}"]
    by_protocol = run_stream(client, {"language": "en"}, audio, subprotocols=protocols)
    assert by_protocol[-1]["type"] == "end"


def test_live_page_is_public(stream_app):
    r = TestClient(stream_app(api_key="k")).get("/live")
    assert r.status_code == 200 and "<title>Live Transcription</title>" in r.text


def test_repeated_short_utterances_are_kept(stream_app):
    """Saying the same short thing twice is speech, not a hallucination."""
    app = stream_app()

    class SaysYes(StreamFakeModel):
        def transcribe(self, audio, **kwargs):
            segs, info = super().transcribe(audio, **kwargs)
            segs = list(segs)[:1]
            segs[0].text, segs[0].end = " Yes.", len(audio) / RATE
            return iter(segs), info

    app.state.transcriber.models._value = SaysYes()
    audio = speech_then_silence((1.0, 0.3), (1.0, 0.3), (1.0, 0.3))
    events = run_stream(TestClient(app), {"language": "en"}, audio)
    assert [e["text"] for e in finals(events)] == ["Yes.", "Yes.", "Yes."]


def test_implausibly_wordy_clip_is_dropped(stream_app):
    app = stream_app()

    class Chatty(StreamFakeModel):
        def transcribe(self, audio, **kwargs):
            segs, info = super().transcribe(audio, **kwargs)
            segs = list(segs)
            if kwargs["beam_size"] == 5 and len(audio) < RATE:  # a sliver "says" a whole sentence
                segs[0].text = " Ask not what your country can do for you, ask what you can do."
            return iter(segs), info

    app.state.transcriber.models._value = Chatty()
    events = run_stream(TestClient(app), {"language": "en"}, speech_then_silence((0.3, 0.3), (1.5, 0.3)))
    assert all("country" not in e["text"] for e in finals(events))
    assert len(finals(events)) >= 1
