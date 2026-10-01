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
OpenAI-compatible API.

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
  openai FILE         Transcribe FILE via the OpenAI-compatible API

  help                Show this help

Environment (all optional):
  PORT=${PORT}          Host port to publish / call
  URL=${URL}
  API_KEY             Sent as 'Authorization: Bearer ...' when set; also
                      passed to the server by 'up', 'start', 'start-cpu', 'dev'
  WHISPER_MODEL       Model to serve (default large-v3; try 'small' on CPU)
  IMAGE=${IMAGE}
  CONTAINER=${CONTAINER}
  VENV=${VENV}

Examples:
  ./run.sh up && ./run.sh wait && ./run.sh transcribe meeting.mp3 srt
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

docker_run() {
    local gpu_args=("$@")
    need docker
    docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" .
    docker run -d --name "$CONTAINER" "${gpu_args[@]}" \
        -p "${PORT}:8000" \
        -v ai-transcriber-cache:/cache \
        -e WHISPER_MODEL="${WHISPER_MODEL:-large-v3}" \
        -e WHISPER_PRELOAD="${WHISPER_PRELOAD:-1}" \
        -e API_KEY="${API_KEY:-}" \
        ${WHISPER_DEVICE:+-e WHISPER_DEVICE="$WHISPER_DEVICE"} \
        "$IMAGE" >/dev/null
    echo "Started ${CONTAINER} on ${URL} — './run.sh wait' blocks until the model is loaded."
}

venv_python() {
    [[ -x "${VENV}/bin/python" ]] || die "no virtualenv at ${VENV}/ — run './run.sh setup' first"
    echo "${VENV}/bin/python"
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
        docker rm -f "$CONTAINER" >/dev/null && echo "Stopped ${CONTAINER}."
        ;;
    setup)
        need python3
        python3 -m venv "$VENV"
        "${VENV}/bin/pip" install -q --upgrade pip
        "${VENV}/bin/pip" install -q -r requirements-dev.txt
        echo "Ready: ${VENV}/"
        ;;
    dev)
        py="$(venv_python)"
        PORT="$PORT" WHISPER_DEVICE="${WHISPER_DEVICE:-auto}" WHISPER_CACHE_DIR="${WHISPER_CACHE_DIR:-./cache}" \
            exec "$py" -m transcriber
        ;;
    test)
        exec "$(venv_python)" -m pytest "$@"
        ;;
    lint)
        exec "$(venv_python)" -m ruff check .
        ;;
    check)
        "$(venv_python)" -m ruff check .
        exec "$(venv_python)" -m pytest "$@"
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
        curl --fail-with-body -sS "${auth[@]}" -F "file=@$1" -F "response_format=${2:-json}" "${URL}/transcribe"
        if [[ "${2:-json}" == "json" ]]; then echo; fi  # text formats already end with a newline
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
