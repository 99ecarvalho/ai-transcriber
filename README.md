# ai-transcriber

[![CI](https://github.com/99ecarvalho/ai-transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/99ecarvalho/ai-transcriber/actions/workflows/ci.yml)
[![License: LGPL v3+](https://img.shields.io/badge/license-LGPL--3.0--or--later-blue.svg)](COPYING.LESSER)

A self-hosted speech-to-text HTTP service built on
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Send an audio file
and get back text, timestamps or subtitles. It runs on an NVIDIA GPU or on the
CPU, ships as a Docker image, and also speaks the **OpenAI audio API**, so
existing OpenAI SDKs and tools work by changing their base URL.

Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>

## Quickstart

You need Docker and, for GPU mode, the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
git clone https://github.com/99ecarvalho/ai-transcriber.git
cd ai-transcriber

./run.sh up                          # build the image and start it on the GPU
./run.sh wait                        # first start downloads the model (~3 GB for large-v3)
./run.sh transcribe recording.mp3    # → JSON with text, language and segments
./run.sh transcribe recording.mp3 srt
```

No GPU? Use a smaller model on the CPU instead of `./run.sh up`:

```bash
WHISPER_MODEL=small ./run.sh start-cpu
./run.sh wait
```

Or call it directly:

```bash
curl -F file=@recording.mp3 http://localhost:8000/transcribe
```

`./run.sh help` lists every command. Stop the service with `./run.sh down`;
the downloaded model stays in a Docker volume for the next start.

---

- [Features](#features)
- [Running](#running)
- [Configuration](#configuration)
- [Native API](#native-api)
- [OpenAI-compatible API](#openai-compatible-api)
- [Health and readiness](#health-and-readiness)
- [Errors](#errors)
- [Security](#security)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Any audio format FFmpeg can read:** mp3, wav, m4a, ogg, flac, webm, and
  audio tracks from video files.
- **Output formats:** JSON, plain text, SRT and WebVTT subtitles, and optional
  per-word timestamps.
- **About 100 languages:** detected automatically or set per request, plus
  translation of any of them to English.
- **OpenAI-compatible endpoints:** `/v1/audio/transcriptions`,
  `/v1/audio/translations` and `/v1/models`.
- **Fit for production:**
  - an optional API key
  - an upload size limit
  - a cap on concurrent GPU jobs, with a bounded queue
  - separate liveness and readiness checks
  - structured JSON logs
- **GPU or CPU:** `WHISPER_DEVICE=auto` picks CUDA when it's available, and
  the compute type adjusts to the device.

## Running

### Docker Compose (GPU)

[docker-compose.yml](docker-compose.yml) builds the image, reserves the GPU,
loads the model at startup and keeps the model cache in the named volume
`whisper-cache`:

```bash
./run.sh up      # same as: docker compose up -d --build
./run.sh logs
./run.sh down
```

Settings are read from the environment or from a `.env` file next to
`docker-compose.yml`, for example:

```bash
WHISPER_MODEL=medium API_KEY=change-me ./run.sh up
```

### Plain `docker run`

```bash
docker build -t ai-transcriber .

# GPU
docker run -d --name ai-transcriber --gpus all -p 8000:8000 \
  -v ai-transcriber-cache:/cache \
  -e WHISPER_PRELOAD=1 \
  ai-transcriber

# CPU
docker run -d --name ai-transcriber -p 8000:8000 \
  -v ai-transcriber-cache:/cache \
  -e WHISPER_DEVICE=cpu -e WHISPER_MODEL=small -e WHISPER_PRELOAD=1 \
  ai-transcriber
```

`./run.sh start` and `./run.sh start-cpu` run these same commands for you.

The container runs as an unprivileged user (UID 1000). A named volume works
as is. If you bind-mount a host directory on `/cache` instead, make it
writable by UID 1000 (`sudo chown 1000:1000 /path/to/cache`).

### From source (no Docker)

You need Python 3.10 or newer. For the GPU you also need CUDA 12 and cuDNN 9
libraries (see the
[faster-whisper requirements](https://github.com/SYSTRAN/faster-whisper#gpu)).

```bash
./run.sh setup   # creates .venv/ with runtime and dev dependencies
./run.sh dev     # serves on :8000, device auto-detected, models cached in ./cache
```

### Choosing a model

| Model | Notes |
| ----- | ----- |
| `tiny`, `base` | Very fast; fine for clear speech and for testing |
| `small` | A good CPU default |
| `medium` | Better accuracy; GPU recommended |
| `large-v3` (default) | The most accurate; GPU recommended |
| `distil-large-v3` | Close to large-v3 for English, much faster |

English-only variants (`tiny.en`, `base.en`, `small.en`, `medium.en`) are also
available. Models are downloaded from Hugging Face on first use and cached in
`/cache`.

## Configuration

All settings are environment variables, and all are optional.

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `WHISPER_MODEL` | `large-v3` | Model name (see [Choosing a model](#choosing-a-model)) or a path to a converted model |
| `WHISPER_DEVICE` | `cuda` | `cuda`, `cpu` or `auto` |
| `WHISPER_COMPUTE_TYPE` | per device | `float16` on CUDA and `int8` on CPU. Also accepts `int8_float16`, `float32`, and others. GPU-only types fall back to `int8` on the CPU |
| `WHISPER_LANGUAGE` | auto-detect | Default language code when a request doesn't set one |
| `WHISPER_CACHE_DIR` | `/cache/faster-whisper` | Where models are stored |
| `WHISPER_PRELOAD` | `0` | `1` loads the model at startup rather than on the first request. Compose and `run.sh` set it to `1` |
| `WHISPER_LOAD_RETRY_SEC` | `60` | After a failed model load, requests fail fast with a 503 for this long before another load is attempted |
| `MAX_UPLOAD_MB` | `200` | Largest accepted request body. Bigger requests get a 413 |
| `MAX_CONCURRENT` | `1` | Transcriptions that run at the same time. Raise it only if your GPU has spare memory |
| `MAX_QUEUE` | `8` | Requests allowed to wait for a free slot. Beyond that the server answers 503 with `Retry-After` |
| `API_KEY` | none | When set, every endpoint except `/health`, `/ready` and the docs requires `Authorization: Bearer <API_KEY>` |
| `EXPOSE_ERROR_DETAILS` | `0` | `1` adds internal exception details to error messages. Useful for debugging; keep it off in public deployments |
| `LOG_FILENAMES` | `0` | `1` logs uploaded file names, which can contain personal data |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warning` or `error` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Address the server listens on inside the container |

## Native API

Interactive documentation is served at `/docs` (Swagger UI) and `/redoc`.

### `POST /transcribe`

Multipart form fields:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `file` | required | The audio file |
| `language` | auto | Language code such as `en`, `pt` or `es`. Empty or `auto` detects it |
| `task` | `transcribe` | `translate` translates the speech into English |
| `response_format` | `json` | `json`, `text`, `srt` or `vtt` |
| `word_timestamps` | `false` | Adds a `words` list to each segment |
| `initial_prompt` | none | Text that steers vocabulary, spelling and style, such as names or jargon |
| `vad_filter` | `true` | Skip silence using voice activity detection |
| `beam_size` | `5` | Beam search width, 1–20. Lower is faster |

```bash
curl -F file=@meeting.m4a -F language=en http://localhost:8000/transcribe
```

```json
{
  "text": "And so my fellow Americans, ask not what your country can do for you...",
  "language": "en",
  "language_probability": 0.974,
  "audio_duration_sec": 11.0,
  "elapsed_ms": 1187,
  "segments": [
    {"start": 0.0, "end": 11.0, "text": " And so my fellow Americans, ask not what your country can do for you..."}
  ]
}
```

Subtitles and other options:

```bash
curl -F file=@talk.mp4 -F response_format=srt http://localhost:8000/transcribe > talk.srt
curl -F file=@talk.mp4 -F response_format=vtt http://localhost:8000/transcribe > talk.vtt
curl -F file=@talk.mp4 -F word_timestamps=true http://localhost:8000/transcribe
curl -F file=@entrevista.mp3 -F task=translate http://localhost:8000/transcribe
curl -F file=@standup.mp3 -F initial_prompt="Kubernetes, Grafana, Prometheus" http://localhost:8000/transcribe
```

Without the optional fields, the response is the same as in earlier versions,
so existing clients keep working unchanged.

## OpenAI-compatible API

These endpoints follow
[OpenAI's audio API](https://platform.openai.com/docs/api-reference/audio):

| Endpoint | Purpose |
| -------- | ------- |
| `POST /v1/audio/transcriptions` | Transcribe. Formats: `json`, `text`, `srt`, `vtt` and `verbose_json`, with `timestamp_granularities[]` of `segment` and/or `word` |
| `POST /v1/audio/translations` | Translate to English |
| `GET /v1/models` | Lists `whisper-1` and the served model |

With the official Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-API_KEY-or-anything")

with open("meeting.mp3", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper-1", file=f).text)

with open("meeting.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
    )
print(result.words[0])  # word='And', start=0.0, end=0.74
```

With curl:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F model=whisper-1 -F file=@meeting.mp3 -F response_format=verbose_json
```

How it differs from OpenAI:

- `model` is accepted but ignored. The server always uses `WHISPER_MODEL`.
- `language` in `verbose_json` is a code such as `en`, not a name such as
  `english`.
- Setting `temperature` uses that single value. Leaving it out keeps Whisper's
  usual fallback, which retries at higher temperatures when decoding fails.
- Streaming responses aren't supported yet.

## Health and readiness

| Endpoint | Meaning |
| -------- | ------- |
| `GET /health` | **Liveness:** 200 whenever the process is up. Also reports the model, device, compute type, whether the model is loaded (`model_state`) and the upload limit. The Docker `HEALTHCHECK` uses it |
| `GET /ready` | **Readiness:** 200 once the model is loaded. 503 while it isn't, with `status` set to `not_loaded`, `loading` or `failed` |

Without `WHISPER_PRELOAD=1`, the model loads on the first transcription
request, and `/ready` returns 503 until then.

## Errors

Native endpoints return `{"detail": "..."}`. The `/v1/` endpoints return
OpenAI-style `{"error": {"message", "type", "param", "code"}}`.

| Status | When |
| ------ | ---- |
| 400 | Empty file, audio that can't be decoded, unsupported language, or invalid parameters on `/v1/` endpoints |
| 401 | Missing or wrong API key |
| 413 | Request body larger than `MAX_UPLOAD_MB` |
| 422 | Invalid parameters on native endpoints |
| 503 | Model is loading or failed to load, or the queue is full. Check `Retry-After` |
| 500 | Unexpected failure during transcription |

## Security

- Set `API_KEY` whenever the service is reachable by anyone but you. Without
  it, anyone who can reach the port can use your GPU.
- The service speaks plain HTTP, so the API key and audio travel
  unencrypted. If traffic leaves your machine or private network, put the
  service behind a reverse proxy that handles TLS.
- Unauthenticated requests are refused before their body is read. Uploads
  over `MAX_UPLOAD_MB` are refused up front when they declare their size, and
  cut off once they pass the limit when they don't.
- Error messages and logs leave out internal details and file names unless
  `EXPOSE_ERROR_DETAILS=1` or `LOG_FILENAMES=1` is set.

## Development

```bash
./run.sh setup     # .venv/ with runtime + dev dependencies
./run.sh check     # ruff + pytest
./run.sh dev       # run from source
```

The tests use a fake model, so they need no GPU and download nothing. They
still decode real audio and exercise the full HTTP stack: both APIs, every
output format, the upload limit, authentication, model-load failures and the
concurrency queue.

```
transcriber/
  app.py            FastAPI app factory, exception handlers, middleware wiring
  config.py         Environment variables → Settings
  engine.py         Model loading, concurrency limiter, transcription
  middleware.py     API key and upload-size checks (before the body is read)
  routes_native.py  /health, /ready, /transcribe
  routes_openai.py  /v1/audio/transcriptions, /v1/audio/translations, /v1/models
  formats.py        JSON / text / SRT / VTT rendering
  errors.py         Error type and per-API error formats
tests/              pytest suite
run.sh              Every common command
```

## Contributing

Bug reports and pull requests are welcome on
[GitHub](https://github.com/99ecarvalho/ai-transcriber/issues). Please run `./run.sh check` before
opening a pull request, and add tests for any behavior you change. Keep the
native `/transcribe` defaults backward compatible.

## License

Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>

ai-transcriber is free software: you can redistribute it and/or modify it
under the terms of the **GNU Lesser General Public License, version 3 or (at
your option) any later version**. The license text is in
[COPYING.LESSER](COPYING.LESSER); it supplements the GNU General Public
License v3, included as [COPYING](COPYING).

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.

The Whisper models and faster-whisper are distributed under their own
licenses (MIT).
