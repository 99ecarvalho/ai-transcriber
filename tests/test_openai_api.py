# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations


def post(client, path="/v1/audio/transcriptions", wav=b"", data=None, files=None):
    files = files or {"file": ("a.wav", wav)}
    return client.post(path, files=files, data={"model": "whisper-1", **(data or {})})


def test_json_default(client, wav):
    r = post(client, wav=wav)
    assert r.status_code == 200
    assert r.json() == {"text": "Hello world."}


def test_text_format(client, wav):
    assert post(client, wav=wav, data={"response_format": "text"}).text == "Hello world.\n"


def test_verbose_json_segments(client, wav, fake_model):
    body = post(client, wav=wav, data={"response_format": "verbose_json"}).json()
    assert body["task"] == "transcribe"
    assert body["language"] == "en"
    assert body["duration"] == 1.0
    seg = body["segments"][0]
    assert set(seg) == {
        "id", "seek", "start", "end", "text", "tokens", "temperature", "avg_logprob", "compression_ratio",
        "no_speech_prob",
    }
    assert "words" not in body
    assert fake_model.calls[0]["word_timestamps"] is False


def test_verbose_json_words(client, wav, fake_model):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav)},
        data={"model": "whisper-1", "response_format": "verbose_json", "timestamp_granularities[]": ["word"]},
    )
    body = r.json()
    assert body["words"] == [
        {"word": " Hello", "start": 0.0, "end": 0.4},
        {"word": " world.", "start": 0.5, "end": 0.9},
    ]
    assert "segments" not in body
    assert fake_model.calls[0]["word_timestamps"] is True


def test_prompt_temperature_language(client, wav, fake_model):
    post(client, wav=wav, data={"prompt": "Grafana", "temperature": "0.2", "language": "es"})
    call = fake_model.calls[0]
    assert call["initial_prompt"] == "Grafana"
    assert call["temperature"] == 0.2
    assert call["language"] == "es"


def test_translation(client, wav, fake_model):
    body = post(client, "/v1/audio/translations", wav=wav, data={"response_format": "verbose_json"}).json()
    assert body["task"] == "translate"
    assert fake_model.calls[0]["task"] == "translate"


def test_errors_use_openai_shape(client, wav):
    r = post(client, wav=wav, data={"response_format": "docx"})
    assert r.status_code == 400
    assert r.json() == {
        "error": {
            "message": "response_format must be one of json, text, srt, verbose_json, vtt.",
            "type": "invalid_request_error",
            "param": "response_format",
            "code": None,
        }
    }
    r = post(client, wav=wav, data={"language": "xx"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "language"


def test_validation_errors_become_400(client, wav):
    r = post(client, wav=wav, data={"temperature": "5"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "temperature"


def test_missing_file(client):
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper-1"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "file"


def test_models(client):
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["whisper-1", "large-v3"]
