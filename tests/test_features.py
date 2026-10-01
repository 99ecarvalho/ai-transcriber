# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""URL input, speaker labels and translation on the batch endpoint, plus their building blocks."""
from __future__ import annotations

import numpy as np
import pytest

from transcriber import audio as audio_mod
from transcriber.audio import PcmConverter, check_url
from transcriber.diarize import OnlineSpeakerTracker, SpeakerModel, cluster
from transcriber.errors import ServiceError
from transcriber.translate import TextTranslator, split_sentences


class FakeTranslator:
    supported = {"en", "pt", "es"}

    def __init__(self):
        self.calls = []

    def supports(self, code):
        return code in self.supported

    def translate(self, texts, source, target):
        self.calls.append((list(texts), source, target))
        return [f"<{target}>{t.strip()}" for t in texts]


class FakeSpeakers:
    def label_segments(self, audio, spans, num_speakers=None):
        return [f"SPEAKER_{i % 2 + 1}" for i in range(len(spans))]

    def embed(self, audio):
        v = np.zeros(4, np.float32)
        v[0] = 1.0
        return v


@pytest.fixture
def feature_client(make_client, fake_model):
    translator, speakers = FakeTranslator(), FakeSpeakers()

    def make(**overrides):
        from fastapi.testclient import TestClient

        from transcriber.app import create_app
        from transcriber.config import Settings

        settings = Settings(device="cpu", cache_dir="/tmp/unused", **overrides)
        app = create_app(settings, lambda _m: fake_model, lambda: translator, lambda: speakers)
        return TestClient(app)

    make.translator = translator
    return make


# ---------- Batch: diarization and translation ----------

def test_diarize_adds_speakers_only_when_asked(feature_client, wav):
    client = feature_client()
    plain = client.post("/transcribe", files={"file": ("a.wav", wav)}).json()
    assert "speakers" not in plain and "speaker" not in plain["segments"][0]

    body = client.post("/transcribe", files={"file": ("a.wav", wav)}, data={"diarize": "true"}).json()
    assert body["speakers"] == ["SPEAKER_1"]
    assert body["segments"][0]["speaker"] == "SPEAKER_1"


def test_translate_to(feature_client, wav):
    client = feature_client()
    body = client.post("/transcribe", files={"file": ("a.wav", wav)}, data={"translate_to": "pt"}).json()
    assert body["translation_language"] == "pt"
    assert body["translation"] == "<pt>Hello world."
    assert body["segments"][0]["translation"] == "<pt>Hello world."
    assert body["text"] == "Hello world."  # original text is kept


def test_translation_source_is_english_after_whisper_translate(feature_client, wav):
    client = feature_client()
    client.post("/transcribe", files={"file": ("a.wav", wav)},
                data={"translate_to": "pt", "task": "translate", "language": "es"})
    assert feature_client.translator.calls[-1][1:] == ("en", "pt")


def test_text_formats_with_speakers_and_translation(feature_client, wav):
    client = feature_client()
    data = {"diarize": "true", "translate_to": "pt"}
    files = {"file": ("a.wav", wav)}
    text = client.post("/transcribe", files=files, data={**data, "response_format": "text"})
    assert text.text == "SPEAKER_1: <pt>Hello world.\n"
    srt = client.post("/transcribe", files={"file": ("a.wav", wav)}, data={**data, "response_format": "srt"})
    assert srt.text == "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_1] <pt>Hello world.\n"


def test_unsupported_translation_target(feature_client, wav):
    client = feature_client()
    r = client.post("/transcribe", files={"file": ("a.wav", wav)}, data={"translate_to": "de"})
    assert r.status_code == 400 and r.json()["detail"].startswith("Translating to 'de'")
    r = client.post("/transcribe", files={"file": ("a.wav", wav)}, data={"translate_to": "xx"})
    assert r.status_code == 400 and "Unsupported language" in r.json()["detail"]


def test_openai_endpoints_are_untouched(feature_client, wav):
    r = feature_client().post("/v1/audio/transcriptions", files={"file": ("a.wav", wav)},
                              data={"model": "whisper-1", "diarize": "true", "translate_to": "pt"})
    assert r.json() == {"text": "Hello world."}


# ---------- Batch: URL input ----------

def test_url_input_is_off_by_default(feature_client):
    r = feature_client().post("/transcribe", data={"url": "https://example.com/a.mp3"})
    assert r.status_code == 403
    assert r.json()["detail"] == "URL input is disabled on this server (set ALLOW_URLS=1)."


def test_missing_file_and_url_keeps_the_old_422(feature_client):
    r = feature_client().post("/transcribe", data={"language": "en"})
    assert r.status_code == 422
    missing = {"type": "missing", "loc": ["body", "file"], "msg": "Field required", "input": None}
    assert r.json() == {"detail": [missing]}


def test_file_and_url_together(feature_client, wav):
    r = feature_client(allow_urls=True).post(
        "/transcribe", files={"file": ("a.wav", wav)}, data={"url": "https://example.com/a.mp3"}
    )
    assert r.status_code == 400


def test_url_transcription(feature_client, monkeypatch):
    seen = {}

    def fake_decode(url, max_seconds):
        seen["url"] = url
        return np.zeros(16000, np.float32) + 0.1

    monkeypatch.setattr(audio_mod, "decode_url", fake_decode)
    client = feature_client(allow_urls=True, allow_private_urls=True)
    r = client.post("/transcribe", data={"url": " http://127.0.0.1:9/a.mp3 "})
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "Hello world."
    assert seen["url"] == "http://127.0.0.1:9/a.mp3"


def test_url_too_long(feature_client, monkeypatch):
    def too_long(url, max_seconds):
        raise audio_mod._TooLong(max_seconds)

    monkeypatch.setattr(audio_mod, "decode_url", too_long)
    r = feature_client(allow_urls=True, allow_private_urls=True).post(
        "/transcribe", data={"url": "http://127.0.0.1:9/a.mp3"}
    )
    assert r.status_code == 400 and "too long" in r.json()["detail"]


def test_url_private_address_refused(feature_client):
    r = feature_client(allow_urls=True).post("/transcribe", data={"url": "http://127.0.0.1/a.mp3"})
    assert r.status_code == 400 and "private or local" in r.json()["detail"]


@pytest.mark.parametrize("url", ["ftp://example.com/a.mp3", "file:///etc/passwd", "http://10.0.0.1/x",
                                 "http://[::1]/x", "http://localhost/x", "not a url"])
def test_check_url_rejects(url):
    with pytest.raises(ServiceError):
        check_url(url, allow_private=False)


def test_check_url_allows_private_when_configured():
    check_url("http://127.0.0.1/x", allow_private=True)


# ---------- Building blocks ----------

def test_cluster_separates_two_voices():
    rng = np.random.default_rng(0)
    a, b = np.eye(8)[0], np.eye(8)[1]
    vectors = [a + 0.05 * rng.standard_normal(8) for _ in range(4)] + [b + 0.05 * rng.standard_normal(8)
                                                                     for _ in range(3)]
    emb = np.stack([v / np.linalg.norm(v) for v in vectors])
    labels = cluster(emb, threshold=0.5)
    assert len(set(labels[:4])) == 1 and len(set(labels[4:])) == 1 and labels[0] != labels[4]
    assert len(set(cluster(emb, threshold=0.5, num_speakers=1))) == 1
    assert len(set(cluster(emb, threshold=0.99, num_speakers=2))) == 2


def test_label_segments_with_short_segments():
    model = SpeakerModel.__new__(SpeakerModel)
    model.method, model.threshold = "segments", 0.5

    def embed(audio):  # voice = which half of the test signal the clip comes from
        v = np.array([1.0, 0.0]) if audio.mean() > 0 else np.array([0.0, 1.0])
        return v

    model.embed = embed
    audio = np.concatenate([np.full(16000 * 10, 0.1), np.full(16000 * 10, -0.1)]).astype(np.float32)
    spans = [(0, 4), (4, 4.5), (4.5, 9), (10, 14), (14, 15.5), (15.5, 19)]
    assert model.label_segments(audio, spans) == [
        "SPEAKER_1", "SPEAKER_1", "SPEAKER_1", "SPEAKER_2", "SPEAKER_2", "SPEAKER_2",
    ]


def test_online_tracker():
    tracker = OnlineSpeakerTracker(threshold=0.5)
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert tracker.assign(a, 4.0) == "SPEAKER_1"
    assert tracker.assign(b, 4.0) == "SPEAKER_2"
    assert tracker.assign(a, 4.0) == "SPEAKER_1"
    assert tracker.assign(None, 0.5) == "SPEAKER_1"  # too short: keeps the previous speaker
    assert tracker.assign(np.array([0.6, 0.8]), 1.5) == "SPEAKER_2"  # short: joins the closest, no new one


def test_split_sentences():
    assert split_sentences(" Hello there.  How are you? Fine! ") == ["Hello there.", "How are you?", "Fine!"]
    assert split_sentences("") == []


def test_translator_language_tokens():
    tr = TextTranslator.__new__(TextTranslator)
    tr.family, tr.vocabulary = "nllb", {"eng_Latn", "por_Latn", "jav_Latn"}
    assert tr.lang_token("pt") == "por_Latn" and tr.lang_token("jw") == "jav_Latn"
    assert tr.lang_token("de") is None
    tr.family, tr.vocabulary = "m2m100", {"__en__", "__jv__"}
    assert tr.lang_token("en") == "__en__" and tr.lang_token("jw") == "__jv__"
    assert not tr.supports("pt")


def test_pcm_converter():
    tone = (np.sin(np.arange(48000) / 5) * 10000).astype("<i2").tobytes()
    same = PcmConverter(16000).convert(tone[:32000])
    assert same.dtype == np.float32 and len(same) == 16000
    resampled = PcmConverter(48000)
    out = np.concatenate([resampled.convert(tone[:30001]), resampled.convert(tone[30001:])])
    assert abs(len(out) - 16000) < 200  # 1 s of 48 kHz audio -> ~1 s at 16 kHz
