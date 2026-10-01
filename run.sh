#!/usr/bin/env bash
# One entry point for building, running, testing and calling ai-transcriber.
#
# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
set -euo pipefail

cd "$(dirname "$0")"

IMAGE="${IMAGE:-ai-transcriber:latest}"
CONTAINER="${CONTAINER:-ai-transcriber}"
PORT="${PORT:-8000}"
URL="${URL:-http://localhost:${PORT}}"
VENV="${VENV:-.venv}"

usage() {
    cat <<EOF
Usage: ./run.sh <command> [args]

ai-transcriber: speech-to-text HTTP service (faster-whisper), with an
OpenAI-compatible API, live transcription, speaker labels and translation.

Docker (recommended):
  up                  Build and start with Docker Compose (GPU), in the background
  down                Stop and remove the Compose service (keeps the model cache)
  logs                Follow the service logs
  build               Build the Docker image (${IMAGE})
  start               Run the image with plain 'docker run' on the GPU
  start-cpu           Run the image with plain 'docker run' on the CPU (slower)
  stop                Stop the container started by 'start' / 'start-cpu'

Local development (no Docker):
  setup               Create ${VENV}/ and install runtime + dev dependencies
  dev                 Run the server from source (uses ${VENV}/)
  test                Run the test suite
  lint                Run the linter (ruff)
  check               lint + test

Talking to a running server:
  health              Show /health
  ready               Show /ready (200 once the model is loaded)
  wait                Block until /ready answers 200
  transcribe FILE [FORMAT]
                      Transcribe FILE via the native API;
                      FORMAT: json (default) | text | srt | vtt
  transcribe-url URL [FORMAT]
                      Same, for audio the server downloads (e.g. a podcast episode)
  openai FILE         Transcribe FILE via the OpenAI-compatible API

Live transcription (lines appear as people speak):
  listen URL          A live stream or podcast URL, pulled by the server
  mic                 Your microphone (Linux: PulseAudio/PipeWire via ffmpeg)
  system-audio        Whatever your computer is playing: a meeting, a video, a podcast
  live                Open the browser page (microphone, tab audio or a URL)

  help                Show this help

Environment (all optional):
  PORT=${PORT}          Host port to publish / call
  URL=${URL}
  API_KEY             Sent as 'Authorization: Bearer ...' when set; also
                      passed to the server by 'up', 'start', 'start-cpu', 'dev'
  WHISPER_MODEL       Model to serve (default large-v3; try 'small' on CPU)
  LANGUAGE            Spoken language for transcribe/listen/mic (default: detect)
  TRANSLATE_TO        Also translate into this language, e.g. TRANSLATE_TO=pt
  DIARIZE=1           Label who is speaking
  AUDIO_INPUT         ffmpeg input for 'mic' / 'system-audio', e.g. "-f alsa -i hw:0"
  IMAGE=${IMAGE}
  CONTAINER=${CONTAINER}
  VENV=${VENV}

Examples:
  ./run.sh up && ./run.sh wait && ./run.sh transcribe meeting.mp3 srt
  DIARIZE=1 TRANSLATE_TO=pt ./run.sh transcribe interview.mp3 text
  ./run.sh transcribe-url https://example.com/episode.mp3 srt
  DIARIZE=1 ./run.sh system-audio
  WHISPER_MODEL=small ./run.sh start-cpu
  API_KEY=secret ./run.sh dev
EOF
}

die() { echo "error: $*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not installed"; }

auth_args() {
    if [[ -n "${API_KEY:-}" ]]; then
        printf '%s\n' -H "Authorization: Bearer ${API_KEY}"
    fi
}

# Optional form fields for /transcribe, from LANGUAGE / TRANSLATE_TO / DIARIZE.
option_args() {
    if [[ -n "${LANGUAGE:-}" ]]; then printf '%s\n' -F "language=${LANGUAGE}"; fi
    if [[ -n "${TRANSLATE_TO:-}" ]]; then printf '%s\n' -F "translate_to=${TRANSLATE_TO}"; fi
    if [[ "${DIARIZE:-0}" == "1" ]]; then printf '%s\n' -F "diarize=true"; fi
}

# Live client: from the local virtualenv if there is one, otherwise inside the
# running container (which has everything installed). Extra args pass through.
client() {
    local opts=()
    if [[ -n "${LANGUAGE:-}" ]]; then opts+=(--language "$LANGUAGE"); fi
    if [[ -n "${TRANSLATE_TO:-}" ]]; then opts+=(--translate-to "$TRANSLATE_TO"); fi
    if [[ "${DIARIZE:-0}" == "1" ]]; then opts+=(--diarize); fi
    if [[ -x "${VENV}/bin/python" ]]; then
        API_KEY="${API_KEY:-}" "${VENV}/bin/python" -m transcriber.client --server "$URL" "${opts[@]}" "$@"
    elif docker compose ps -q transcriber 2>/dev/null | grep -q .; then
        docker compose exec -T -e API_KEY="${API_KEY:-}" transcriber \
            python3 -m transcriber.client --server http://localhost:8000 "${opts[@]}" "$@"
    elif docker container inspect "$CONTAINER" >/dev/null 2>&1; then
        docker exec -i -e API_KEY="${API_KEY:-}" "$CONTAINER" \
            python3 -m transcriber.client --server http://localhost:8000 "${opts[@]}" "$@"
    else
        die "no client available: run './run.sh setup', or start the server with './run.sh up'"
    fi
}

# Captures audio with ffmpeg as 16 kHz mono PCM and streams it to the server.
capture() {
    local input=("$@")
    need ffmpeg
    if [[ -n "${AUDIO_INPUT:-}" ]]; then
        read -r -a input <<< "$AUDIO_INPUT"
    fi
    echo "· capturing: ffmpeg ${input[*]}" >&2
    ffmpeg -hide_banner -loglevel error "${input[@]}" -ac 1 -ar 16000 -f s16le - | client --stdin
}

docker_run() {
    local gpu_args=("$@")
    need docker
    if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
        if [[ "$(docker container inspect -f '{{.State.Running}}' "$CONTAINER")" == "true" ]]; then
            die "container '${CONTAINER}' is already running — './run.sh stop' first"
        fi
        docker rm "$CONTAINER" >/dev/null  # leftover from an earlier run
    fi
    docker build -q -t "$IMAGE" . >/dev/null  # cached layers make this quick when nothing changed
    docker run -d --name "$CONTAINER" "${gpu_args[@]}" \
        -p "${PORT}:8000" \
        -v ai-transcriber-cache:/cache \
        -e WHISPER_MODEL="${WHISPER_MODEL:-large-v3}" \
        -e WHISPER_PRELOAD="${WHISPER_PRELOAD:-1}" \
        -e API_KEY="${API_KEY:-}" \
        -e ALLOW_URLS="${ALLOW_URLS:-1}" \
        ${WHISPER_DEVICE:+-e WHISPER_DEVICE="$WHISPER_DEVICE"} \
        ${TRANSLATION_MODEL:+-e TRANSLATION_MODEL="$TRANSLATION_MODEL"} \
        ${DIARIZATION_METHOD:+-e DIARIZATION_METHOD="$DIARIZATION_METHOD"} \
        "$IMAGE" >/dev/null
    echo "Started ${CONTAINER} on ${URL} — './run.sh wait' blocks until the model is loaded."
}

# Must be called directly, not inside $(...): die has to exit the main shell.
require_venv() {
    [[ -x "${VENV}/bin/python" ]] || die "no virtualenv at ${VENV}/ — run './run.sh setup' first"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
    up)
        need docker
        docker compose up -d --build
        echo "Started on ${URL} — './run.sh wait' blocks until the model is loaded."
        ;;
    down)
        need docker
        docker compose down
        ;;
    logs)
        need docker
        docker compose logs -f
        ;;
    build)
        need docker
        docker build -t "$IMAGE" .
        ;;
    start)
        docker_run --gpus all
        ;;
    start-cpu)
        WHISPER_DEVICE=cpu docker_run
        ;;
    stop)
        need docker
        if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
            docker rm -f "$CONTAINER" >/dev/null
            echo "Stopped ${CONTAINER}."
        else
            echo "No container named ${CONTAINER}."
        fi
        ;;
    setup)
        need python3
        python3 -m venv "$VENV"
        "${VENV}/bin/pip" install -q --upgrade pip
        "${VENV}/bin/pip" install -q -r requirements-dev.txt
        echo "Ready: ${VENV}/"
        ;;
    dev)
        require_venv
        py="${VENV}/bin/python"
        PORT="$PORT" WHISPER_DEVICE="${WHISPER_DEVICE:-auto}" WHISPER_CACHE_DIR="${WHISPER_CACHE_DIR:-./cache}" \
            DIARIZATION_CACHE_DIR="${DIARIZATION_CACHE_DIR:-./cache/speaker}" ALLOW_URLS="${ALLOW_URLS:-1}" \
            exec "$py" -m transcriber
        ;;
    test)
        require_venv
        exec "${VENV}/bin/python" -m pytest "$@"
        ;;
    lint)
        require_venv
        exec "${VENV}/bin/python" -m ruff check .
        ;;
    check)
        require_venv
        "${VENV}/bin/python" -m ruff check .
        exec "${VENV}/bin/python" -m pytest "$@"
        ;;
    health)
        need curl
        curl --fail-with-body -sS "${URL}/health"; echo
        ;;
    ready)
        need curl
        curl -sS "${URL}/ready"; echo
        ;;
    wait)
        need curl
        echo -n "Waiting for ${URL}/ready "
        for _ in $(seq 1 600); do
            if [[ "$(curl -s -o /dev/null -w '%{http_code}' "${URL}/ready")" == "200" ]]; then
                echo " ready."
                exit 0
            fi
            echo -n "."
            sleep 1
        done
        echo
        die "not ready after 10 minutes (see './run.sh logs')"
        ;;
    transcribe)
        need curl
        [[ $# -ge 1 ]] || die "usage: ./run.sh transcribe FILE [json|text|srt|vtt]"
        [[ -f "$1" ]] || die "no such file: $1"
        mapfile -t auth < <(auth_args)
        mapfile -t opts < <(option_args)
        curl --fail-with-body -sS "${auth[@]}" "${opts[@]}" -F "file=@$1" -F "response_format=${2:-json}" \
            "${URL}/transcribe"
        if [[ "${2:-json}" == "json" ]]; then echo; fi  # text formats already end with a newline
        ;;
    transcribe-url)
        need curl
        [[ $# -ge 1 ]] || die "usage: ./run.sh transcribe-url URL [json|text|srt|vtt]"
        mapfile -t auth < <(auth_args)
        mapfile -t opts < <(option_args)
        curl --fail-with-body -sS "${auth[@]}" "${opts[@]}" -F "url=$1" -F "response_format=${2:-json}" \
            "${URL}/transcribe"
        if [[ "${2:-json}" == "json" ]]; then echo; fi
        ;;
    listen)
        [[ $# -ge 1 ]] || die "usage: ./run.sh listen URL"
        url="$1"; shift
        client --url "$url" "$@"
        ;;
    mic)
        capture -f pulse -i default
        ;;
    system-audio)
        need pactl
        capture -f pulse -i "$(pactl get-default-sink).monitor"
        ;;
    live)
        echo "Open ${URL}/live"
        if command -v xdg-open >/dev/null 2>&1; then xdg-open "${URL}/live" >/dev/null 2>&1 || true; fi
        ;;
    openai)
        need curl
        [[ $# -ge 1 ]] || die "usage: ./run.sh openai FILE"
        [[ -f "$1" ]] || die "no such file: $1"
        mapfile -t auth < <(auth_args)
        curl --fail-with-body -sS "${auth[@]}" -F "file=@$1" -F model=whisper-1 "${URL}/v1/audio/transcriptions"
        echo
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        die "unknown command: $cmd"
        ;;
esac
