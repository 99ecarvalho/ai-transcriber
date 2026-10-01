# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations


def test_health_keeps_original_fields(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    for key in ("model", "device", "compute_type", "model_loaded", "max_upload_bytes"):
        assert key in body
    assert body["device"] == "cpu"
    assert body["compute_type"] == "int8"  # auto-selected for CPU
    assert body["model_loaded"] is False


def test_default_response_is_unchanged(client, wav, fake_model):
    r = client.post("/transcribe", files={"file": ("a.wav", wav, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert list(body) == [
        "text", "language", "language_probability", "audio_duration_sec", "elapsed_ms", "segments",
    ]
    assert body["text"] == "Hello world."
    assert body["language_probability"] == 0.988
    assert body["segments"] == [{"start": 0.0, "end": 1.0, "text": " Hello world."}]
    call = fake_model.calls[0]
    assert call["beam_size"] == 5 and call["vad_filter"] is True
    assert call["task"] == "transcribe" and call["word_timestamps"] is False
    assert "temperature" not in call  # faster-whisper's own fallback schedule


def test_word_timestamps(client, wav):
    r = client.post("/transcribe", files={"file": ("a.wav", wav)}, data={"word_timestamps": "true"})
    words = r.json()["segments"][0]["words"]
    assert words[0] == {"start": 0.0, "end": 0.4, "word": " Hello", "probability": 0.98}


def test_text_srt_vtt(client, wav):
    def get(fmt):
        return client.post("/transcribe", files={"file": ("a.wav", wav)}, data={"response_format": fmt})

    assert get("text").text == "Hello world.\n"
    assert get("srt").text == "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n"
    vtt = get("vtt")
    assert vtt.headers["content-type"].startswith("text/vtt")
    assert vtt.text == "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello world.\n"


def test_prompt_task_and_language_are_passed(client, wav, fake_model):
    client.post(
        "/transcribe",
        files={"file": ("a.wav", wav)},
        data={"initial_prompt": "Kubernetes, Grafana", "task": "translate", "language": "PT"},
    )
    call = fake_model.calls[0]
    assert call["initial_prompt"] == "Kubernetes, Grafana"
    assert call["task"] == "translate"
    assert call["language"] == "pt"


def test_auto_language(client, wav, fake_model):
    client.post("/transcribe", files={"file": ("a.wav", wav)}, data={"language": "auto"})
    assert fake_model.calls[0]["language"] is None


def test_invalid_language_is_400(client, wav):
    r = client.post("/transcribe", files={"file": ("a.wav", wav)}, data={"language": "xx"})
    assert r.status_code == 400
    assert "Unsupported language" in r.json()["detail"]


def test_undecodable_audio_is_400(client):
    r = client.post("/transcribe", files={"file": ("a.mp3", b"\x00\x01garbage" * 200)})
    assert r.status_code == 400
    assert r.json()["detail"] == "Could not decode audio from the uploaded file."


def test_error_details_can_be_exposed(make_client):
    client = make_client(expose_error_details=True)
    r = client.post("/transcribe", files={"file": ("a.mp3", b"\x00\x01garbage" * 200)})
    assert r.json()["detail"].startswith("Could not decode audio from the uploaded file.: InvalidDataError")


def test_empty_file_is_400(client):
    r = client.post("/transcribe", files={"file": ("a.wav", b"")})
    assert r.status_code == 400
    assert r.json()["detail"] == "File is empty."


def test_invalid_params_are_422(client, wav):
    r = client.post("/transcribe", files={"file": ("a.wav", wav)}, data={"beam_size": "0"})
    assert r.status_code == 422
    r = client.post("/transcribe", files={"file": ("a.wav", wav)}, data={"response_format": "docx"})
    assert r.status_code == 422


def test_no_temp_files_left(client, wav, tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    client.post("/transcribe", files={"file": ("a.wav", wav)})
    client.post("/transcribe", files={"file": ("a.mp3", b"garbage" * 100)})
    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == []
