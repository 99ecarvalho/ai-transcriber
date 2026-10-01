# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Limits, auth, model lifecycle and concurrency."""
from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import httpx
import pytest

from transcriber.app import create_app
from transcriber.config import Settings
from transcriber.engine import resolve_compute_type

MB = 1024 * 1024


# ---------- Upload limit ----------

def test_upload_limit_with_content_length(make_client):
    client = make_client(max_upload_bytes=MB)
    r = client.post("/transcribe", files={"file": ("a.wav", b"\0" * (2 * MB))})
    assert r.status_code == 413
    assert r.json() == {"detail": "Request body exceeds the 1 MB limit."}


def test_upload_limit_chunked(make_client):
    client = make_client(max_upload_bytes=MB)

    def body():
        yield b"--x\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n"
        for _ in range(4):
            yield b"\0" * (MB // 2)
        yield b"\r\n--x--\r\n"

    r = client.post(
        "/transcribe", content=body(), headers={"Content-Type": "multipart/form-data; boundary=x"}
    )
    assert "content-length" not in r.request.headers  # really chunked, no declared size
    assert r.status_code == 413


def test_upload_limit_openai_shape(make_client):
    client = make_client(max_upload_bytes=MB)
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"\0" * (2 * MB))})
    assert r.status_code == 413
    assert r.json()["error"]["type"] == "invalid_request_error"


# ---------- Auth ----------

def test_api_key(make_client, wav):
    client = make_client(api_key="s3cret")
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code in (200, 503)

    r = client.post("/transcribe", files={"file": ("a.wav", wav)})
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"

    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav)},
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"

    r = client.post("/transcribe", files={"file": ("a.wav", wav)}, headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200


# ---------- Model lifecycle ----------

def test_ready_flips_after_first_load(client, wav):
    assert client.get("/ready").status_code == 503
    assert client.get("/ready").json()["status"] == "not_loaded"
    client.post("/transcribe", files={"file": ("a.wav", wav)})
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_preload(make_client, fake_model):
    with make_client(preload=True) as client:  # context manager runs the lifespan
        for _ in range(100):
            if client.get("/ready").status_code == 200:
                break
            threading.Event().wait(0.02)
        assert client.get("/ready").status_code == 200


def test_load_failure_cools_down(make_client, wav):
    attempts = []

    def failing_loader(_manager):
        attempts.append(1)
        raise RuntimeError("CUDA driver not found")

    client = make_client(loader=failing_loader, load_retry_sec=60)
    r = client.post("/transcribe", files={"file": ("a.wav", wav)})
    assert r.status_code == 503
    assert r.json()["detail"] == "Failed to load model."
    assert "retry-after" in r.headers

    r = client.post("/transcribe", files={"file": ("a.wav", wav)})
    assert r.status_code == 503
    assert r.json()["detail"] == "Model is unavailable (last load failed)."
    assert len(attempts) == 1  # second request did not retry the load

    assert client.get("/ready").json()["status"] == "failed"


def test_load_retries_after_cooldown(make_client, wav, fake_model):
    attempts = []

    def flaky_loader(_manager):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("transient")
        return fake_model

    client = make_client(loader=flaky_loader, load_retry_sec=0)
    assert client.post("/transcribe", files={"file": ("a.wav", wav)}).status_code == 503
    assert client.post("/transcribe", files={"file": ("a.wav", wav)}).status_code == 200


# ---------- Device / compute type ----------

def test_compute_type_resolution():
    assert resolve_compute_type("cuda", None) == "float16"
    assert resolve_compute_type("cpu", None) == "int8"
    assert resolve_compute_type("cpu", "float16") == "int8"
    assert resolve_compute_type("cpu", "float32") == "float32"
    assert resolve_compute_type("cuda", "int8_float16") == "int8_float16"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "CPU")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1.5")
    monkeypatch.setenv("API_KEY", "  k  ")
    s = Settings.from_env()
    assert s.device == "cpu"
    assert s.max_upload_bytes == int(1.5 * MB)
    assert s.api_key == "k"
    monkeypatch.setenv("MAX_CONCURRENT", "0")
    with pytest.raises(ValueError):
        Settings.from_env()


# ---------- Concurrency ----------

@pytest.mark.anyio
async def test_queue_full_returns_503(base_settings, fake_model, wav):
    fake_model.gate = threading.Event()
    app = create_app(replace(base_settings, max_concurrent=1, max_queue=0), lambda _m: fake_model)
    limiter = app.state.transcriber.limiter

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        first = asyncio.create_task(client.post("/transcribe", files={"file": ("a.wav", wav)}))
        for _ in range(200):
            if limiter.active == 1:
                break
            await asyncio.sleep(0.01)
        assert limiter.active == 1

        busy = await client.post("/transcribe", files={"file": ("a.wav", wav)})
        assert busy.status_code == 503
        assert busy.headers["retry-after"] == "5"

        fake_model.gate.set()
        assert (await first).status_code == 200
    assert limiter.active == 0 and limiter.in_flight == 0


@pytest.mark.anyio
async def test_requests_waiting_for_model_load_are_bounded(base_settings, fake_model, wav):
    loaded = threading.Event()

    def slow_loader(_manager):
        loaded.wait(timeout=10)
        return fake_model

    app = create_app(replace(base_settings, max_concurrent=1, max_queue=0), slow_loader)
    t = app.state.transcriber

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        first = asyncio.create_task(client.post("/transcribe", files={"file": ("a.wav", wav)}))
        for _ in range(200):
            if t.models.loading:
                break
            await asyncio.sleep(0.01)
        assert t.models.state == "loading"

        busy = await client.post("/transcribe", files={"file": ("a.wav", wav)})
        assert busy.status_code == 503

        loaded.set()
        assert (await first).status_code == 200
    assert t.limiter.in_flight == 0


@pytest.mark.anyio
async def test_cancelled_waiter_does_not_abandon_the_load(base_settings, fake_model):
    from transcriber.engine import ModelManager

    release = threading.Event()
    calls = []

    def loader(_manager):
        calls.append(1)
        release.wait(timeout=10)
        return fake_model

    manager = ModelManager(base_settings, loader)
    waiter = asyncio.create_task(manager.get())
    for _ in range(200):
        if calls:
            break
        await asyncio.sleep(0.01)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release.set()
    assert await manager.get() is fake_model
    assert len(calls) == 1


def test_model_gets_one_worker_per_concurrent_job(base_settings, monkeypatch):
    import faster_whisper

    from transcriber.engine import ModelManager, _load_whisper_model

    seen = {}

    class RecordingModel:
        def __init__(self, name, **kwargs):
            seen.update(kwargs, name=name)

    monkeypatch.setattr(faster_whisper, "WhisperModel", RecordingModel)
    _load_whisper_model(ModelManager(replace(base_settings, max_concurrent=3)))
    assert seen["num_workers"] == 3
    assert seen["device"] == "cpu" and seen["compute_type"] == "int8"


def test_invalid_default_language_fails_at_startup(base_settings):
    with pytest.raises(ValueError, match="WHISPER_LANGUAGE"):
        create_app(replace(base_settings, default_language="xx"))


def test_invalid_log_level_fails_at_startup(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "warn")
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        Settings.from_env()
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    assert Settings.from_env().log_level == "warning"


def test_bearer_scheme_is_case_insensitive(make_client, wav):
    client = make_client(api_key="s3cret")
    for header in ("bearer s3cret", "BEARER s3cret"):
        r = client.post("/transcribe", files={"file": ("a.wav", wav)}, headers={"Authorization": header})
        assert r.status_code == 200, header
    r = client.post("/transcribe", files={"file": ("a.wav", wav)}, headers={"Authorization": "Bearer S3CRET"})
    assert r.status_code == 401
