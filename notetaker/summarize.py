"""Abstractive chunk summarization.

Pipeline per chunk:
    1. Split into sentences.
    2. Rank sentences by embedding centrality (extractive filter).
    3. Concatenate the top-K salient sentences in their original order.
    4. Run an abstractive summarizer over that condensed passage to produce
       a multi-sentence summary, then split it into bullets.
    5. The first generated sentence becomes the chunk headline.

A purely extractive fallback is kept for `--no-abstractive` runs.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Callable, List, Optional, Tuple

import numpy as np

from .models import ChunkSummary, TranscriptChunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return parts or [text]


def summarize_chunks(
    chunks: List[TranscriptChunk],
    embedder,
    bullets_per_chunk: int = 3,
    summarizer: Optional[Callable[[str, int], List[str]]] = None,
    salient_sentences: int = 6,
) -> List[ChunkSummary]:
    """Produce a headline + bullets per chunk.

    If `summarizer` is provided, output is abstractive (rewritten by an
    instruction-tuned model). Otherwise falls back to verbatim extractive
    sentences.
    """
    results: List[ChunkSummary] = []
    for chunk in chunks:
        if summarizer is not None:
            summary, bullets = _abstractive(
                chunk.text, embedder, summarizer, bullets_per_chunk, salient_sentences
            )
        else:
            summary, bullets = _extractive(chunk.text, embedder, bullets_per_chunk)
        results.append(
            ChunkSummary(
                start=chunk.start,
                end=chunk.end,
                text=chunk.text,
                summary=summary,
                bullets=bullets,
            )
        )
    return results


def _rank_sentences(sentences: List[str], embedder) -> np.ndarray:
    embeddings = embedder.encode(
        sentences, convert_to_numpy=True, normalize_embeddings=True
    )
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, 0.0)
    return sims.mean(axis=1)


def _abstractive(
    text: str,
    embedder,
    summarizer: Callable[[str, int], List[str]],
    n_bullets: int,
    n_salient: int,
) -> Tuple[str, List[str]]:
    """Embedding-filtered salience → instruction-guided bullet rewrite.

    `n_bullets` is treated as an upper bound. The actual bullet count is
    adapted to chunk complexity, which prevents low-information padding.
    """
    sentences = split_sentences(text)
    if not sentences:
        return "", []

    target = _adaptive_bullet_count(sentences, n_bullets)

    if len(sentences) > 1:
        centrality = _rank_sentences(sentences, embedder)
        keep = sorted(
            np.argsort(-centrality)[: max(target, min(n_salient, len(sentences)))]
        )
        condensed = " ".join(sentences[i] for i in keep)
    else:
        condensed = sentences[0]

    bullets = summarizer(condensed, target)
    bullets = _clean_and_dedupe(bullets, target)
    if not bullets:
        # Model produced nothing usable; fall back to extractive on the same input.
        return _extractive(condensed, embedder, target)

    headline = bullets[0].rstrip(".!?").strip()
    support = [b for b in bullets[1:] if not _near_duplicate(b, headline)]
    return headline, support


def _extractive(
    text: str, embedder, n_bullets: int
) -> Tuple[str, List[str]]:
    sentences = split_sentences(text)
    if not sentences:
        return "", []
    if len(sentences) == 1:
        return sentences[0], []

    target = _adaptive_bullet_count(sentences, n_bullets)

    centrality = _rank_sentences(sentences, embedder)
    order = np.argsort(-centrality)
    headline_idx = int(order[0])
    top_idx = sorted(order[: max(1, min(target, len(sentences)))])

    headline = sentences[headline_idx].strip()
    support = [sentences[i].strip() for i in top_idx if i != headline_idx]
    support = _clean_and_dedupe(support, target - 1)
    return headline, support


def _adaptive_bullet_count(sentences: List[str], max_bullets: int) -> int:
    """Choose a flexible bullet target based on chunk density.

    This reduces forced padding on simple chunks while still allowing richer
    chunks to emit multiple supporting bullets.
    """
    n = len(sentences)
    if n <= 2:
        return 1
    if n <= 5:
        return min(2, max_bullets)
    return max(1, max_bullets)


def _clean_and_dedupe(lines: List[str], limit: int) -> List[str]:
    cleaned: List[str] = []
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if _is_low_information(line):
            continue
        if any(_near_duplicate(line, existing) for existing in cleaned):
            continue
        cleaned.append(line)
        if len(cleaned) >= max(1, limit):
            break
    return cleaned


def _is_low_information(text: str) -> bool:
    low_patterns = (
        r"\b(as of this month|that\'s how it is|right\?|currently\s*,?\s*right)\b",
        r"^that is\b",
    )
    if len(text.split()) < 4:
        return True
    if text.rstrip().endswith("?") and len(text.split()) < 10:
        return True
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in low_patterns)


def _near_duplicate(a: str, b: str) -> bool:
    a_norm = a.strip().lower().rstrip(".!?")
    b_norm = b.strip().lower().rstrip(".!?")
    if not a_norm or not b_norm:
        return False
    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    return ratio >= 0.88
