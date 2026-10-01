# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Live (near real-time) transcription of an audio stream.

Whisper works on whole windows of audio, not on a stream, so a session keeps a
rolling buffer of the current utterance:

- About once a second (STREAM_PARTIAL_INTERVAL_SEC of new audio) the buffer is
  re-transcribed quickly and sent as a `partial` event — text that may change.
- When voice activity detection sees a pause of STREAM_MIN_SILENCE_MS after
  speech, the utterance is transcribed properly and sent as `final` events
  (optionally with a speaker label and a translation), then dropped from the
  buffer.
- An utterance longer than STREAM_MAX_UTTERANCE_SEC (someone who never pauses)
  is cut: everything but its last segment is finalised.

Processing is driven by audio arriving, not by a clock: a pulled podcast file
is processed as fast as the GPU allows, and when processing falls behind,
partials are skipped until it catches up.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from .audio import SAMPLE_RATE, iter_url_audio
from .diarize import MIN_MATCH_SEC, OnlineSpeakerTracker
from .engine import TranscribeOptions, Transcriber, transcribe_audio
from .errors import ServiceError

log = structlog.get_logger("transcriber")

Send = Callable[[dict[str, Any]], Awaitable[None]]
Vad = Callable[[np.ndarray], list[tuple[int, int]]]

SPEECH_PAD_SEC = 0.2
MIN_COMMIT_SEC = 1.0  # don't early-commit slivers of audio
MIN_PROMPT_AUDIO_SEC = 2.0
MAX_WORDS_PER_SEC = 6.0  # fast speech is ~4-5 words/s
MAX_TURN_SAMPLES = 15 * SAMPLE_RATE  # judge the speaker on at most the last 15 s of a turn
FILLERS = {"um", "uh", "uhm", "umm", "hmm", "hm", "mm", "mhm", "ah", "er", "erm"}  # not worth a line
BOUNDARY_AGREEMENT = 0.4  # seconds
CONTEXT_CHARS = 200  # recent final text passed to Whisper as a prompt, for continuity
QUEUE_CHUNKS = 256


def silero_vad(min_silence_ms: int) -> Vad:
    """faster-whisper's bundled Silero VAD, as [(start_sample, end_sample), ...]."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_silence_duration_ms=max(100, min_silence_ms // 2), speech_pad_ms=0)

    def vad(audio: np.ndarray) -> list[tuple[int, int]]:
        return [(t["start"], t["end"]) for t in get_speech_timestamps(audio, options)]

    return vad


@dataclass
class StreamConfig:
    language: str | None = None  # None -> detected from the first utterance, then kept
    task: str = "transcribe"
    translate_to: str | None = None
    diarize: bool = False
    initial_prompt: str | None = None


class StreamSession:
    def __init__(self, transcriber: Transcriber, config: StreamConfig, send: Send, vad: Vad | None = None):
        self.t = transcriber
        self.settings = transcriber.settings
        self.config = config
        self.send = send
        self.vad = vad or silero_vad(self.settings.stream_min_silence_ms)

        self.buffer = np.zeros(0, np.float32)
        self.buffer_start = 0.0  # stream time (seconds) of buffer[0]
        self.received = 0  # samples received so far
        self._last_partial_at = 0
        self._partial_shown = False
        self._final_id = 0
        self._boundary: float | None = None  # segment boundary seen by the previous partial
        self._turn_audio = np.zeros(0, np.float32)  # already-final audio of the current speaker turn
        self._context = config.initial_prompt or ""
        self.language = config.language
        self.tracker = OnlineSpeakerTracker(self.settings.diarization_threshold) if config.diarize else None
        self.model: Any = None
        self.speakers: Any = None
        self.translator: Any = None

    # ---------- lifecycle ----------

    async def prepare(self) -> None:
        """Loads every model the session needs (may take minutes the first time)."""
        self.model = await self.t.models.get()
        if self.config.diarize:
            self.speakers = await self.t.speakers.get()
        if self.config.translate_to:
            self.translator = await self.t._translator_for(self.config.translate_to)

    async def run(self, queue: asyncio.Queue) -> None:
        """Consumes float32 16 kHz chunks from `queue` until it yields None."""
        await self.send({
            "type": "ready",
            "model": self.settings.model_name,
            "sample_rate": SAMPLE_RATE,
            "language": self.language,
            "translate_to": self.config.translate_to,
            "diarize": self.config.diarize,
        })
        ended = False
        while not ended:
            chunks = [await queue.get()]
            while not queue.empty():
                chunks.append(queue.get_nowait())
            if any(c is None for c in chunks):
                ended = True
                chunks = [c for c in chunks if c is not None]
            if chunks:
                audio = np.concatenate(chunks)
                self.buffer = np.concatenate([self.buffer, audio])
                self.received += len(audio)
            try:
                await self._step(flush=ended, allow_partial=queue.empty())
            except ServiceError:
                raise
            except Exception as e:
                log.exception("transcriber.stream_step_failed")
                await self.send({"type": "error", "message": f"Processing failed: {type(e).__name__}"})
                self._drop(len(self.buffer))
        await self.send({"type": "end", "duration": round(self.received / SAMPLE_RATE, 2)})

    # ---------- per step ----------

    async def _step(self, flush: bool, allow_partial: bool) -> None:
        # When processing has fallen behind, the buffer can hold several
        # utterances: finalise them one at a time, so each gets its own speaker.
        while await self._step_once(flush, allow_partial):
            pass

    async def _step_once(self, flush: bool, allow_partial: bool) -> bool:
        """Processes the buffer once. True if it finalised something (call again)."""
        buf = self.buffer
        if len(buf) < int(0.25 * SAMPLE_RATE) and not flush:
            return False
        speech = await asyncio.to_thread(self.vad, buf) if len(buf) else []
        if not speech:
            if self._partial_shown:
                at = round(self.buffer_start, 2)
                await self.send({"type": "partial", "text": "", "start": at, "end": at})
                self._partial_shown = False
            self._drop(max(0, len(buf) - int(0.3 * SAMPLE_RATE)))  # keep a little lead-in
            return False

        pad = int(SPEECH_PAD_SEC * SAMPLE_RATE)
        min_gap = self.settings.stream_min_silence_ms * SAMPLE_RATE // 1000
        begin = max(0, speech[0][0] - pad)
        for (_, end), (next_start, _) in zip(speech, speech[1:]):  # noqa: B905 - pairs of neighbours
            if next_start - end >= min_gap:  # a complete utterance followed by more speech
                await self._finalize(begin, min(end + pad, next_start), ends_utterance=True)
                return True

        last_end = speech[-1][1]
        if flush:
            await self._finalize(begin, len(buf), ends_utterance=True)
            return False
        if len(buf) - last_end >= min_gap:
            await self._finalize(begin, min(len(buf), last_end + pad), ends_utterance=True)
            return True
        if len(buf) - begin >= self.settings.stream_max_utterance_sec * SAMPLE_RATE:
            await self._finalize(begin, len(buf), ends_utterance=False, keep_last_segment=True)
            return True
        if allow_partial and self.received - self._last_partial_at >= (
            self.settings.stream_partial_interval_sec * SAMPLE_RATE
        ):
            await self._partial(begin)
        return False

    def _drop(self, samples: int) -> None:
        self.buffer = self.buffer[samples:]
        self.buffer_start += samples / SAMPLE_RATE

    def _options(self, final: bool, audio_sec: float) -> TranscribeOptions:
        # Given only a sliver of audio, Whisper tends to repeat its prompt back,
        # so short clips get no context prompt.
        prompt = self._context[-CONTEXT_CHARS:] if audio_sec >= MIN_PROMPT_AUDIO_SEC else ""
        return TranscribeOptions(
            language=self.language,
            task=self.config.task,
            beam_size=5 if final else 1,
            vad_filter=False,  # the session already cut the audio at speech boundaries
            initial_prompt=prompt or None,
        )

    async def _partial(self, begin: int) -> None:
        self._last_partial_at = self.received
        audio = self.buffer[begin:]
        opts = self._options(final=False, audio_sec=len(audio) / SAMPLE_RATE)
        segs, _ = await asyncio.to_thread(transcribe_audio, self.model, audio, opts)
        text = "".join(s.text for s in segs).strip()
        start = self.buffer_start + begin / SAMPLE_RATE

        # Early commit: when two partials in a row agree that a segment ended at
        # the same point, everything before it is settled — finalise it now
        # instead of waiting for the speaker to pause.
        boundary = None
        if len(segs) >= 2 and segs[-1].start >= MIN_COMMIT_SEC:
            boundary = start + segs[-1].start
        previous, self._boundary = self._boundary, boundary
        if boundary is not None and previous is not None and abs(boundary - previous) <= BOUNDARY_AGREEMENT:
            self._boundary = None
            await self._finalize(begin, begin + int(segs[-1].start * SAMPLE_RATE), ends_utterance=False)
            return
        await self.send({
            "type": "partial",
            "text": text,
            "start": round(start, 2),
            "end": round(start + len(audio) / SAMPLE_RATE, 2),
        })
        self._partial_shown = bool(text)

    async def _finalize(
        self, begin: int, cut: int, ends_utterance: bool, keep_last_segment: bool = False
    ) -> None:
        """Transcribes buffer[begin:cut] properly and sends it as final lines.

        ends_utterance=False means the speaker is still talking (early commit or
        a forced cut): the audio is kept for judging the speaker, so a long turn
        is recognised by its whole voice, not by sentence-sized pieces.
        """
        audio = self.buffer[begin:cut]
        offset = self.buffer_start + begin / SAMPLE_RATE
        opts = self._options(final=True, audio_sec=len(audio) / SAMPLE_RATE)
        turn_audio = np.concatenate([self._turn_audio, audio]) if len(self._turn_audio) else audio
        recent_turn = turn_audio[-MAX_TURN_SAMPLES:]
        segs, info, embedding = await asyncio.to_thread(self._final_pass, audio, recent_turn, opts)
        # More words than anyone can say in that much audio is Whisper repeating
        # its prompt or inventing text, not speech; fillers aren't worth a line.
        max_words = MAX_WORDS_PER_SEC * len(audio) / SAMPLE_RATE + 1
        segs = [s for s in segs if len(s.text.split()) <= max_words and _norm(s.text) not in FILLERS]

        consumed = cut
        if keep_last_segment and len(segs) > 1 and segs[-1].start > 0:
            last = segs.pop()
            consumed = begin + int(last.start * SAMPLE_RATE)
            embedding = None  # computed on the whole window; recompute isn't worth it here

        if self.language is None and info.language_probability >= 0.5:
            self.language = info.language
            await self.send({
                "type": "language",
                "language": info.language,
                "probability": round(info.language_probability, 3),
            })

        speaker = None
        if self.tracker is not None and segs:
            turn_sec = min(len(turn_audio), MAX_TURN_SAMPLES) / SAMPLE_RATE
            speaker = self.tracker.assign(embedding, turn_sec, update=ends_utterance)
        self._turn_audio = np.zeros(0, np.float32) if ends_utterance else turn_audio[-MAX_TURN_SAMPLES:]

        translations: list[str | None] = [None] * len(segs)
        if self.translator is not None and segs:
            source = "en" if self.config.task == "translate" else (self.language or info.language)
            if self.translator.supports(source):
                translations = await asyncio.to_thread(
                    self.translator.translate, [s.text for s in segs], source, self.config.translate_to
                )

        if self._partial_shown and not segs:
            at = round(offset, 2)
            await self.send({"type": "partial", "text": "", "start": at, "end": at})
        for seg, translation in zip(segs, translations, strict=True):
            text = seg.text.strip()
            if not text:
                continue
            self._final_id += 1
            event: dict[str, Any] = {
                "type": "final",
                "id": self._final_id,
                "start": round(offset + seg.start, 2),
                "end": round(offset + seg.end, 2),
                "text": text,
            }
            if speaker is not None:
                event["speaker"] = speaker
            if translation is not None:
                event["translation"] = translation
            await self.send(event)
            self._context = (self._context + " " + text)[-CONTEXT_CHARS * 2 :]
        self._partial_shown = False
        self._last_partial_at = self.received
        self._boundary = None
        self._drop(consumed)

    def _final_pass(self, audio: np.ndarray, turn_audio: np.ndarray, opts: TranscribeOptions):
        segs, info = transcribe_audio(self.model, audio, opts)
        embedding = None
        if self.speakers is not None and len(turn_audio) >= MIN_MATCH_SEC * SAMPLE_RATE and segs:
            embedding = self.speakers.embed(turn_audio)
        return segs, info, embedding


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


async def pump_url(url: str, queue: asyncio.Queue, stop: threading.Event) -> None:
    """Decodes `url` in a worker thread into `queue`, then puts None (end of stream)."""
    loop = asyncio.get_running_loop()

    def put(item) -> None:
        future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        while not stop.is_set():
            try:
                future.result(timeout=0.5)
                return
            except concurrent.futures.TimeoutError:  # a distinct class on Python 3.10
                continue
        future.cancel()

    def worker() -> None:
        try:
            for chunk in iter_url_audio(url, stop=stop.is_set):
                put(chunk)
                if stop.is_set():
                    return
        finally:
            if not stop.is_set():
                put(None)

    await asyncio.to_thread(worker)
