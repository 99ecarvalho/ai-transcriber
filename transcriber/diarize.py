# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Speaker identification ("who is speaking"), on ONNX models via sherpa-onnx.

Two methods for files (DIARIZATION_METHOD):

- segments (default): one speaker embedding per transcript segment, then
  average-linkage clustering. Labels line up exactly with the transcript.
- pyannote: pyannote's segmentation model finds speaker turns independently of
  the transcript; each transcript segment gets the speaker it overlaps most.

Live streams always use the embedding approach, assigning each finished
utterance to the closest known speaker as it arrives (OnlineSpeakerTracker).

Models are downloaded on first use from the sherpa-onnx releases into
DIARIZATION_CACHE_DIR. Labels are SPEAKER_1, SPEAKER_2, ... in order of first appearance.
"""
from __future__ import annotations

import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE

EMBEDDING_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/"
    "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
# Embeddings of short audio are unreliable. Measured with the default model at
# threshold 0.5: 1 s clips were classified correctly 57% of the time, 2 s 77%,
# 3 s 92%. So only segments >= MIN_EMBED_SEC can found a new speaker; shorter
# ones (>= MIN_MATCH_SEC) join the closest existing speaker, and anything
# shorter still borrows a neighbour's label.
MIN_EMBED_SEC = 3.0
MIN_MATCH_SEC = 1.0
# pyannote clustering threshold (sherpa's scale, not cosine similarity).
PYANNOTE_THRESHOLD = 0.9


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".download-")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest


def resolve_embedding_model(name: str, cache_dir: str) -> str:
    """A local .onnx path, an http(s) URL, or a sherpa-onnx release name."""
    if os.path.isfile(name):
        return name
    if name.startswith(("http://", "https://")):
        url = name
    else:
        url = EMBEDDING_RELEASE + name.removesuffix(".onnx") + ".onnx"
    dest = Path(cache_dir) / url.rsplit("/", 1)[-1]
    return str(dest if dest.exists() else _download(url, dest))


def resolve_segmentation_model(cache_dir: str) -> str:
    target = Path(cache_dir) / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    if not target.exists():
        archive = _download(SEGMENTATION_URL, Path(cache_dir) / "pyannote-segmentation.tar.bz2")
        with tarfile.open(archive) as tar:
            tar.extractall(cache_dir, filter="data")
        archive.unlink()
    return str(target)


class SpeakerModel:
    """Speaker embeddings (+ the optional pyannote pipeline), loaded once."""

    def __init__(self, method: str, model: str, cache_dir: str, threshold: float, threads: int):
        import sherpa_onnx

        self.method = method
        self.threshold = threshold
        self._model_path = resolve_embedding_model(model, cache_dir)
        self._threads = threads
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=self._model_path, num_threads=threads)
        )
        self._segmentation = resolve_segmentation_model(cache_dir) if method == "pyannote" else None

    def embed(self, audio: np.ndarray) -> np.ndarray:
        stream = self.extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        vector = np.asarray(self.extractor.compute(stream), dtype=np.float32)
        return vector / (np.linalg.norm(vector) or 1.0)

    def label_segments(
        self, audio: np.ndarray, spans: list[tuple[float, float]], num_speakers: int | None = None
    ) -> list[str]:
        """One speaker label per (start, end) span of `audio`, in seconds."""
        if not spans:
            return []
        if self.method == "pyannote":
            return self._label_with_pyannote(audio, spans, num_speakers)
        return self._label_with_embeddings(audio, spans, num_speakers)

    def _label_with_embeddings(self, audio, spans, num_speakers) -> list[str]:
        long_idx = [i for i, (a, b) in enumerate(spans) if b - a >= MIN_EMBED_SEC]
        if not long_idx:  # nothing long enough to tell voices apart
            long_idx = [max(range(len(spans)), key=lambda i: spans[i][1] - spans[i][0])]
        embeddings = np.stack([self.embed(_cut(audio, *spans[i])) for i in long_idx])
        clusters = cluster(embeddings, self.threshold, num_speakers)

        cluster_of = {idx: c for idx, c in zip(long_idx, clusters, strict=True)}
        # Short segments: the cluster whose centroid they resemble most, if their
        # audio is usable at all; otherwise the nearest long segment's speaker.
        centroids = {c: _normalize(embeddings[clusters == c].mean(axis=0)) for c in set(clusters.tolist())}
        for i, (a, b) in enumerate(spans):
            if i in cluster_of:
                continue
            if b - a >= MIN_MATCH_SEC:
                vector = self.embed(_cut(audio, a, b))
                cluster_of[i] = max(centroids, key=lambda c: float(vector @ centroids[c]))
            else:
                nearest = min(long_idx, key=lambda j: abs(spans[j][0] - a))
                cluster_of[i] = cluster_of[nearest]
        return _labels_in_order([cluster_of[i] for i in range(len(spans))])

    def _label_with_pyannote(self, audio, spans, num_speakers) -> list[str]:
        import sherpa_onnx

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=self._segmentation),
                num_threads=self._threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self._model_path, num_threads=self._threads
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers or -1, threshold=PYANNOTE_THRESHOLD
            ),
        )
        turns = sherpa_onnx.OfflineSpeakerDiarization(config).process(audio).sort_by_start_time()
        turns = [(t.start, t.end, t.speaker) for t in turns]
        result = []
        for a, b in spans:
            overlap: dict[int, float] = {}
            for ta, tb, spk in turns:
                o = min(b, tb) - max(a, ta)
                if o > 0:
                    overlap[spk] = overlap.get(spk, 0.0) + o
            if overlap:
                result.append(max(overlap, key=overlap.get))
            else:  # no turn overlaps (e.g. noise) — take the closest one in time
                mid = (a + b) / 2
                result.append(min(turns, key=lambda t: abs((t[0] + t[1]) / 2 - mid))[2] if turns else 0)
        return _labels_in_order(result)


def cluster(embeddings: np.ndarray, threshold: float, num_speakers: int | None = None) -> np.ndarray:
    """Average-linkage agglomerative clustering on cosine similarity.

    Merges the most similar pair of clusters until no pair is at least
    `threshold` similar, or until `num_speakers` clusters remain if given.
    Returns a cluster index per embedding.
    """
    n = len(embeddings)
    if n == 1:
        return np.zeros(1, dtype=int)
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)
    sizes = np.ones(n)
    members = {i: [i] for i in range(n)}
    active = np.ones(n, dtype=bool)
    while active.sum() > 1:
        masked = np.where(active[:, None] & active[None, :], sim, -np.inf)
        i, j = np.unravel_index(np.argmax(masked), masked.shape)
        best = masked[i, j]
        if num_speakers is not None:
            if active.sum() <= num_speakers:
                break
        elif best < threshold:
            break
        # Average linkage (Lance-Williams update) for the merged cluster i.
        sim[i, :] = (sizes[i] * sim[i, :] + sizes[j] * sim[j, :]) / (sizes[i] + sizes[j])
        sim[:, i] = sim[i, :]
        sim[i, i] = -np.inf
        sizes[i] += sizes[j]
        members[i] += members.pop(j)
        active[j] = False
    labels = np.zeros(n, dtype=int)
    for c, idx in enumerate(members.values()):
        labels[idx] = c
    return labels


class OnlineSpeakerTracker:
    """Assigns live utterances to speakers as they arrive."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []
        self._last: int | None = None

    def assign(self, embedding: np.ndarray | None, duration: float, update: bool = True) -> str | None:
        """Long utterances may start a new speaker; short ones join the closest
        known speaker; very short ones (embedding=None) keep the previous one.

        update=False labels without folding the embedding into the speaker's
        profile — for a turn that is still going on and will be passed again."""
        if embedding is None or duration < MIN_MATCH_SEC:
            return None if self._last is None else f"SPEAKER_{self._last + 1}"
        if self._centroids:
            sims = [float(embedding @ c) for c in self._centroids]
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold or duration < MIN_EMBED_SEC:
                if update:
                    n = self._counts[best]
                    self._centroids[best] = _normalize((self._centroids[best] * n + embedding) / (n + 1))
                    self._counts[best] += 1
                self._last = best
                return f"SPEAKER_{best + 1}"
        self._centroids.append(embedding)
        self._counts.append(1)
        self._last = len(self._centroids) - 1
        return f"SPEAKER_{self._last + 1}"


def _cut(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    return audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) or 1.0)


def _labels_in_order(clusters: list[int]) -> list[str]:
    names: dict[int, str] = {}
    for c in clusters:
        names.setdefault(c, f"SPEAKER_{len(names) + 1}")
    return [names[c] for c in clusters]
