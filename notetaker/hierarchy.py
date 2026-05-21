"""Group chunks into topic sections and generate meaningful section titles."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .models import ChunkSummary, TopicSection


def build_sections(
    summaries: List[ChunkSummary],
    embedder,
    title_fn: Optional[Callable[[str], str]] = None,
    min_sections: int = 2,
    max_sections: int = 8,
) -> List[TopicSection]:
    """Group consecutive chunks into topic sections via embedding clustering.

    Chooses k by silhouette score, then re-segments along the timeline so each
    section is a contiguous run of chunks (no time-traveling sections).

    `title_fn` is called with the concatenated chunk summaries of each section
    to produce a human-readable title. Falls back to the first chunk's headline
    when no title function is provided.
    """
    if not summaries:
        return []
    if len(summaries) <= min_sections:
        return [TopicSection(title=_title(summaries, title_fn), chunks=summaries)]

    texts = [f"{s.summary} {s.text}" for s in summaries]
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    k = _choose_k(embeddings, min_sections, min(max_sections, len(summaries) - 1))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit(embeddings).labels_

    sections: List[TopicSection] = []
    current: List[ChunkSummary] = [summaries[0]]
    for i in range(1, len(summaries)):
        if labels[i] != labels[i - 1]:
            sections.append(TopicSection(title=_title(current, title_fn), chunks=current))
            current = []
        current.append(summaries[i])
    sections.append(TopicSection(title=_title(current, title_fn), chunks=current))
    return sections


def _choose_k(embeddings: np.ndarray, k_min: int, k_max: int) -> int:
    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _title(
    chunks: List[ChunkSummary], title_fn: Optional[Callable[[str], str]]
) -> str:
    if not chunks:
        return "Section"
    if title_fn is None:
        return chunks[0].summary[:80].rstrip(".") or "Section"
    parts: List[str] = []
    for chunk in chunks:
        if chunk.summary:
            parts.append(chunk.summary)
        if chunk.bullets:
            parts.extend(chunk.bullets[:2])
        elif chunk.text:
            parts.append(chunk.text)
    blob = " ".join(parts).strip()
    return title_fn(blob) if blob else "Section"
