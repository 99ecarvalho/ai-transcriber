# ai-transcriber

[![CI](https://github.com/99ecarvalho/ai-transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/99ecarvalho/ai-transcriber/actions/workflows/ci.yml)
[![License: LGPL v3+](https://img.shields.io/badge/license-LGPL--3.0--or--later-blue.svg)](COPYING.LESSER)

A self-hosted speech-to-text service built on
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Send an audio file
or a URL and get back text, timestamps or subtitles, optionally with who said
what and a translation. It can also transcribe **live**, as people speak: a
meeting, your microphone, or a stream. It runs on an NVIDIA GPU or on the CPU,
ships as a Docker image, and speaks the **OpenAI audio API**, so existing
OpenAI SDKs and tools work by changing their base URL.

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

DIARIZE=1 TRANSLATE_TO=pt ./run.sh transcribe interview.mp3 text   # who said what, in Portuguese
./run.sh transcribe-url https://example.com/episode.mp3            # a podcast episode by URL
./run.sh live                        # browser page: live captions of your mic or a meeting tab
DIARIZE=1 ./run.sh system-audio      # live captions of whatever your computer is playing
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
downloaded models stay in a Docker volume for the next start.

---

- [Features](#features)
- [Running](#running)
- [Configuration](#configuration)
- [Native API](#native-api)
- [Live transcription](#live-transcription)
- [Who is speaking](#who-is-speaking)
- [Translation](#translation)
- [OpenAI-compatible API](#openai-compatible-api)
- [Health and readiness](#health-and-readiness)
- [Errors](#errors)
- [Security](#security)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Any audio format FFmpeg can read:** mp3, wav, m4a, ogg, flac, webm, and
  audio tracks from video files. Upload a file, or pass a URL.
- **Output formats:** JSON, plain text, SRT and WebVTT subtitles, and optional
  per-word timestamps.
- **About 100 languages:** detected automatically or set per request.
- **Live transcription** over WebSocket: text appears while people speak,
  from a microphone, a meeting in a browser tab, your computer's audio, or a
  live stream URL. Includes a browser page and a command-line client.
- **Who is speaking:** speaker labels (`SPEAKER_1`, `SPEAKER_2`, …) for files
  and live audio.
- **Translation:** into English by Whisper itself, or between any of ~100
  languages with a dedicated translation model.
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

Speaker labels, translation, URL input and live streaming are all opt-in per
request. A request that doesn't ask for them gets exactly the same response as
before, and their models are only downloaded on first use.

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

`./run.sh start` and `./run.sh start-cpu` run these same commands for you
(and also enable URL input, `ALLOW_URLS=1`).

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
| `small` | A good CPU default, and a good choice for live transcription |
| `medium` | Better accuracy; GPU recommended |
| `large-v3` (default) | The most accurate; GPU recommended |
| `distil-large-v3` | Close to large-v3 for English, much faster |

English-only variants (`tiny.en`, `base.en`, `small.en`, `medium.en`) are also
available. Models are downloaded from Hugging Face on first use and cached in
`/cache`.

## Configuration

All settings are environment variables, and all are optional.

### Transcription

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `WHISPER_MODEL` | `large-v3` | Model name (see [Choosing a model](#choosing-a-model)) or a path to a converted model |
| `WHISPER_DEVICE` | `cuda` | `cuda`, `cpu` or `auto` |
| `WHISPER_COMPUTE_TYPE` | per device | `float16` on CUDA and `int8` on CPU. Also accepts `int8_float16`, `float32`, and others. GPU-only types fall back to `int8` on the CPU |
| `WHISPER_LANGUAGE` | auto-detect | Default language code when a request doesn't set one. An unsupported code stops the server at startup |
| `WHISPER_CACHE_DIR` | `/cache/faster-whisper` | Where models are stored |
| `WHISPER_PRELOAD` | `0` | `1` loads the model at startup rather than on the first request. Compose and `run.sh` set it to `1` |
| `WHISPER_LOAD_RETRY_SEC` | `60` | After a failed model load, requests fail fast with a 503 for this long before another load is attempted |

### Limits and security

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `MAX_UPLOAD_MB` | `200` | Largest accepted request body. Bigger requests get a 413 |
| `MAX_CONCURRENT` | `1` | Transcriptions that run in parallel, each with its own model worker and extra memory. On a single GPU, `1` is usually fastest. On many-core CPUs, `2` or more can raise throughput. Measure on your hardware |
| `MAX_QUEUE` | `8` | Requests allowed to wait, for a free slot or for the model to finish loading, on top of the `MAX_CONCURRENT` running ones. Beyond that the server answers 503 with `Retry-After` |
| `API_KEY` | none | When set, every endpoint except `/health`, `/ready`, `/live` and the docs requires `Authorization: Bearer <API_KEY>` |
| `ALLOW_URLS` | `0` | `1` lets requests pass a `url` for the server to fetch. Compose and `run.sh` set it to `1`. See [Security](#security) |
| `ALLOW_PRIVATE_URLS` | `0` | `1` also allows URLs on private and local networks |
| `MAX_URL_DURATION_MIN` | `240` | Longest audio accepted from a URL by `/transcribe` |
| `EXPOSE_ERROR_DETAILS` | `0` | `1` adds internal exception details to error messages. Useful for debugging; keep it off in public deployments |
| `LOG_FILENAMES` | `0` | `1` logs uploaded file names, which can contain personal data |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warning`, `error` or `critical` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Address the server listens on inside the container |

### Live transcription settings

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `MAX_STREAMS` | `2` | Live sessions at the same time; each keeps the GPU busy while it runs. `0` disables `/stream` |
| `STREAM_MIN_SILENCE_MS` | `600` | Pause that ends an utterance. Lower gives final text sooner but splits sentences more |
| `STREAM_MAX_UTTERANCE_SEC` | `20` | Longest stretch without a pause before the text is finalised anyway |
| `STREAM_PARTIAL_INTERVAL_SEC` | `1.0` | How often in-progress text is refreshed |

### Speakers and translation

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `DIARIZATION_METHOD` | `segments` | `segments` or `pyannote`; see [Who is speaking](#who-is-speaking) |
| `DIARIZATION_MODEL` | `3dspeaker_speech_eres2net_sv_en_voxceleb_16k` | Speaker-embedding model: a [sherpa-onnx release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) name, a URL or a local `.onnx` path |
| `DIARIZATION_THRESHOLD` | `0.5` | Voice similarity needed to count as the same speaker. Higher finds more speakers |
| `DIARIZATION_CACHE_DIR` | `/cache/speaker` | Where speaker models are stored |
| `DIARIZATION_THREADS` | `2` | CPU threads for speaker models |
| `TRANSLATION_MODEL` | `JustFrederik/nllb-200-distilled-600M-ct2-int8` | A CTranslate2 NLLB-200 or M2M100 model: a Hugging Face repo or local path. See [Translation](#translation) for the license |
| `TRANSLATION_DEVICE` | `auto` | `cuda`, `cpu` or `auto` |

## Native API

Interactive documentation is served at `/docs` (Swagger UI) and `/redoc`.

### `POST /transcribe`

Multipart form fields:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `file` | — | The audio file. Send either `file` or `url` |
| `url` | — | An audio URL for the server to download instead, such as a podcast episode. Needs `ALLOW_URLS=1` |
| `language` | auto | Language code such as `en`, `pt` or `es`. Empty or `auto` detects it |
| `task` | `transcribe` | `translate` translates the speech into English |
| `response_format` | `json` | `json`, `text`, `srt` or `vtt` |
| `word_timestamps` | `false` | Adds a `words` list to each segment |
| `initial_prompt` | none | Text that steers vocabulary, spelling and style, such as names or jargon |
| `vad_filter` | `true` | Skip silence using voice activity detection |
| `beam_size` | `5` | Beam search width, 1–20. Lower is faster |
| `diarize` | `false` | Label who is speaking in each segment |
| `num_speakers` | auto | With `diarize`: the number of speakers, if you know it |
| `translate_to` | none | Also translate the text into this language, e.g. `pt` |

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

With `diarize` and `translate_to`, each segment gains `speaker` and
`translation`, and the response gains `speakers`, `translation` and
`translation_language`:

```bash
curl -F file=@interview.mp3 -F diarize=true -F translate_to=pt http://localhost:8000/transcribe
```

```json
{
  "text": "...",
  "segments": [
    {"start": 0.0, "end": 6.0, "text": " Good morning, everyone.", "speaker": "SPEAKER_1", "translation": "Bom dia a todos."},
    {"start": 6.0, "end": 9.5, "text": " Thanks for having me.", "speaker": "SPEAKER_2", "translation": "Obrigado por me receber."}
  ],
  "speakers": ["SPEAKER_1", "SPEAKER_2"],
  "translation": "Bom dia a todos. Obrigado por me receber.",
  "translation_language": "pt"
}
```

In `text`, `srt` and `vtt`, the lines show the translation when one was
requested, and are prefixed with the speaker when `diarize` is on:

```bash
curl -F file=@interview.mp3 -F diarize=true -F response_format=text http://localhost:8000/transcribe
# SPEAKER_1: Good morning, everyone.
# SPEAKER_2: Thanks for having me.
```

Other examples:

```bash
curl -F file=@talk.mp4 -F response_format=srt http://localhost:8000/transcribe > talk.srt
curl -F file=@talk.mp4 -F response_format=vtt -F translate_to=es http://localhost:8000/transcribe > talk.es.vtt
curl -F file=@talk.mp4 -F word_timestamps=true http://localhost:8000/transcribe
curl -F file=@entrevista.mp3 -F task=translate http://localhost:8000/transcribe
curl -F file=@standup.mp3 -F initial_prompt="Kubernetes, Grafana, Prometheus" http://localhost:8000/transcribe
curl -F url=https://example.com/episode.mp3 -F response_format=text http://localhost:8000/transcribe
```

Without the optional fields, the response is the same as in earlier versions,
so existing clients keep working unchanged.

## Live transcription

Whisper transcribes whole chunks of audio, not a stream, so live mode works
like this:

1. Incoming audio collects in a buffer. About once a second, the current
   utterance is re-transcribed quickly and sent as a **partial**: text that
   may still change.
2. When voice activity detection hears a pause (`STREAM_MIN_SILENCE_MS`), or
   when two partials in a row agree that a sentence has ended, that part is
   transcribed properly and sent as **final**, with its speaker and
   translation if requested.
3. When processing falls behind (or a URL is read faster than real time),
   partials are skipped until it catches up.

Measured on an RTX 4060 laptop GPU with the `small` model, playing a
two-speaker recording at real-time speed with speaker labels and translation
on: final lines arrived a median of **1.0–1.3 s** after the speech ended (90th
percentile 2.9–4.0 s, across runs from source and in Docker), and every line
had the right speaker.

### Ways to use it

| Source | Command |
| ------ | ------- |
| Browser: microphone, a meeting tab (Meet, Teams, Zoom web), or both | `./run.sh live`, or open `http://localhost:8000/live` |
| Whatever your computer is playing (Linux) | `./run.sh system-audio` |
| Microphone (Linux) | `./run.sh mic` |
| A live stream or podcast URL (HTTP, Icecast, HLS) | `./run.sh listen https://example.com/live.m3u8` |
| A local file, sent at real-time speed (for testing) | `python -m transcriber.client --file meeting.mp3` |

`LANGUAGE`, `TRANSLATE_TO` and `DIARIZE=1` work with all the `run.sh`
commands, for example `DIARIZE=1 TRANSLATE_TO=en ./run.sh listen URL`.

- **Browser page:** for a meeting in another tab, tick both sources and share
  that tab with "Share tab audio" on. The tab gives you the other
  participants, the microphone gives you yourself. Browsers only allow audio
  capture on `https://` pages or `http://localhost`, so for a remote server,
  put it behind HTTPS.
- **`mic` and `system-audio`** capture with ffmpeg from PulseAudio or PipeWire.
  On other systems, set `AUDIO_INPUT` to any ffmpeg input, for example
  `AUDIO_INPUT="-f avfoundation -i :0" ./run.sh mic` on macOS.
- **The client** runs from `.venv/` if you ran `./run.sh setup`, otherwise
  inside the running container. Press Ctrl+C once to finish the last line,
  twice to quit.

### WebSocket protocol

Connect to `ws://HOST:8000/stream`, optionally offering the subprotocol
`transcriber.v1`. Send a start message first; every field is optional:

```json
{"type": "start", "language": "en", "task": "transcribe", "translate_to": "pt",
 "diarize": true, "sample_rate": 48000, "url": null}
```

Then send binary frames of signed 16-bit little-endian mono PCM at
`sample_rate` (default 16000), or set `url` and the server pulls the audio
itself. Send `{"type": "stop"}` when done. The server sends JSON events:

| Event | Meaning |
| ----- | ------- |
| `loading` | Models are loading (the first start can take minutes) |
| `ready` | Send audio now |
| `partial` | `text` of the utterance in progress; replaces the previous partial |
| `final` | Finished line: `id`, `start`, `end`, `text`, and `speaker` / `translation` when requested |
| `language` | The detected language, now fixed for the rest of the session |
| `error` | `message`; the connection closes if it can't continue |
| `end` | All audio processed (`duration` in seconds); the server then closes |

Times are seconds since the start of the stream.

## Who is speaking

Speaker labels come from speaker-embedding models run on ONNX (via
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)), so no PyTorch or
Hugging Face token is needed. The models (25–40 MB) download on first use.

Two methods are available for files (`DIARIZATION_METHOD`). For live audio,
each finished utterance is compared with the voices heard so far.

| Method | How it works |
| ------ | ------------ |
| `segments` (default) | One voice embedding per transcript segment, then clustering. Labels line up exactly with the transcript |
| `pyannote` | pyannote's segmentation model finds speaker turns on its own; each transcript segment takes the speaker it overlaps most |

How the default was chosen: on a two-speaker test recording (33 s, alternating
turns, one voice from a noisy 1961 recording), against the known answer. Each
row uses the best clustering threshold tried for that combination.

| Approach | Speakers found | Time labelled correctly |
| -------- | -------------- | ----------------------- |
| `segments` + ERes2Net (default) | 2 | 100% |
| `segments` + CAM++ | 3 | 100% |
| `pyannote` + ERes2Net | 2 | 97% |
| `pyannote` + WeSpeaker ResNet34 | 2 | 97% |
| `pyannote` + CAM++ | 5 | 90% |

This is one recording, so treat it as a sanity check, not a benchmark.
`pyannote` was also about 6 times slower here.

Limits:

- **Short phrases are hard to attribute.** Measured with the default model,
  1-second clips were attributed correctly 57% of the time, 3-second clips
  92%. So a new speaker is only recognised from 3 seconds of their speech;
  shorter lines join the most similar known voice.
- **Overlapping speech gets one label.**
- **Labels are per file or per session.** `SPEAKER_1` in one recording isn't
  the same person as `SPEAKER_1` in another.

## Translation

- **Into English:** use `task=translate`. Whisper translates directly from the
  audio, with no extra model.
- **Into any other language** (and English, too): use `translate_to`. A text
  translation model runs on the transcript, on
  [CTranslate2](https://github.com/OpenNMT/CTranslate2), the engine
  faster-whisper uses. Text is translated one sentence at a time, and in live
  mode only final lines are translated.

Two model families are supported. They were compared on meeting- and
podcast-style sentences in 6 language pairs (en→pt/es/de/fr, pt→en, es→en),
against reference translations, on an RTX 4060:

| Model | Quality (chrF, higher is better) | Time per sentence | Whisper languages covered | License |
| ----- | -------------------------------- | ----------------- | ------------------------- | ------- |
| NLLB-200 distilled 600M (default) | **82.9** | **189 ms** | **97** of 100 | CC-BY-NC-4.0 |
| M2M100 418M | 78.6 | 226 ms | 84 of 100 | MIT |

> [!WARNING]
> The default translation model's weights, NLLB-200, are licensed for
> **non-commercial use only** (CC-BY-NC-4.0). The server logs a warning when
> it loads. For commercial use, switch to M2M100:
>
> ```bash
> TRANSLATION_MODEL=jncraton/m2m100_418M-ct2-int8
> ```

Notes from the comparison:

- NLLB translated into European Portuguese ("podes", "a tua"), while M2M100
  produced Brazilian Portuguese.
- M2M100 scored higher on English→German.
- Both can translate very short interjections oddly. "What?" became
  "- O quê? - Não." with NLLB.

Other CTranslate2 conversions of these two families, such as the larger
NLLB and M2M100 models, should work too, as long as the repository includes
`sentencepiece.bpe.model` and a `shared_vocabulary` file. Only the two models
above were tested. The family is detected from the model's vocabulary.

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
- Speaker labels, `translate_to`, URL input and streaming are native-API
  features. The `/v1/` endpoints stay as close to OpenAI's as possible.

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
| 400 | Empty file, audio that can't be decoded, unsupported language, a URL that's invalid or private, or invalid parameters on `/v1/` endpoints |
| 401 | Missing or wrong API key |
| 403 | `url` sent while URL input is off (`ALLOW_URLS=0`) |
| 413 | Request body larger than `MAX_UPLOAD_MB` |
| 422 | Invalid parameters on native endpoints |
| 503 | A model is loading or failed to load, or the queue is full. Check `Retry-After` |
| 500 | Unexpected failure during transcription |

Live sessions report problems as `error` events. A wrong or missing API key
refuses the WebSocket handshake (HTTP 403).

## Security

- Set `API_KEY` whenever the service is reachable by anyone but you. Without
  it, anyone who can reach the port can use your GPU. WebSocket clients send
  the key as an `Authorization` header or, from browsers, which can't set
  headers, as the subprotocol `bearer.<base64url(API_KEY)>`. The `/live` page
  does this for you. It isn't sent in the URL, because URLs end up in access
  logs.
- The service speaks plain HTTP, so the API key and audio travel
  unencrypted. If traffic leaves your machine or private network, put the
  service behind a reverse proxy that handles TLS. That's also required for
  the `/live` page to use a microphone on a remote host.
- **URL input is off by default** (`ALLOW_URLS=0`), because it makes the
  server fetch addresses chosen by its callers. When on, only `http(s)` URLs
  that resolve to public addresses are accepted, and FFmpeg is limited to
  HTTP(S) and HLS. Redirects and HLS segment addresses aren't re-checked, so
  on servers that can reach sensitive internal systems, keep it off or keep
  `API_KEY` set.
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

The tests use fake models, so they need no GPU and download nothing. They
still decode real audio and exercise the full HTTP and WebSocket stack:

- both APIs and every output format
- the upload limit, authentication, model-load failures and the queue
- URL input
- speaker labels and translation
- live sessions: partials, finals, pauses, long monologues, speaker changes

```
transcriber/
  app.py            FastAPI app factory, exception handlers, middleware wiring
  config.py         Environment variables → Settings
  engine.py         Model loading, concurrency limiter, the transcription pipeline
  lazy.py           Load-once models with failure cool-down
  audio.py          URL checks and decoding, PCM conversion
  streaming.py      Live sessions: buffering, partials, finals
  diarize.py        Speaker labels (clustering, pyannote, live tracking)
  translate.py      Text translation (NLLB-200 / M2M100 on CTranslate2)
  middleware.py     API key and upload-size checks (before the body is read)
  routes_native.py  /health, /ready, /transcribe
  routes_openai.py  /v1/audio/transcriptions, /v1/audio/translations, /v1/models
  routes_stream.py  WebSocket /stream and the /live page
  client.py         Command-line live client (python -m transcriber.client)
  formats.py        JSON / text / SRT / VTT rendering
  errors.py         Error type and per-API error formats
  static/live.html  The browser page
tests/              pytest suite
run.sh              Every common command
```

## Contributing

Bug reports and pull requests are welcome on
[GitHub](https://github.com/99ecarvalho/ai-transcriber/issues). Please run
`./run.sh check` before opening a pull request, and add tests for any
behavior you change. Keep the native `/transcribe` defaults backward
compatible.

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

The models this service downloads at runtime have their own licenses:

| Model | License |
| ----- | ------- |
| Whisper (and faster-whisper itself) | MIT |
| NLLB-200, the default translation model | CC-BY-NC-4.0: non-commercial use only |
| M2M100, the alternative translation model | MIT |
| pyannote segmentation 3.0 | MIT |
| 3D-Speaker ERes2Net and CAM++ speaker models | Apache-2.0 |
