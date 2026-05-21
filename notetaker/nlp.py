"""Instruction-guided rewrite helpers for bullets and section headings.

The generation path uses an instruction-tuned seq2seq model (Flan-T5 by
default) with direct model.generate calls so behavior stays stable even when
pipeline task registration changes across transformer versions.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List

import re

DEFAULT_SUMMARIZER = "google/flan-t5-base"

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "with", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "these", "those", "it", "its", "as", "at", "by", "from", "we", "you",
    "i", "they", "he", "she", "them", "our", "your", "their", "his", "her", "so",
    "do", "does", "did", "not", "no", "yes", "can", "will", "would", "should",
    "could", "have", "has", "had", "about", "into", "than", "such", "also", "just",
    "like", "what", "when", "where", "which", "who", "how", "there", "here", "out",
    "up", "down", "over", "more", "most", "some", "any", "all", "one", "two",
    "very", "really", "actually", "okay", "right", "going", "got", "get", "go",
    "know", "think", "thing", "things", "kind", "lot", "well", "way", "see", "say",
    "said", "make", "made", "now", "because", "probably", "keep", "getting", "final",
}

_CLEAN_NOTES_PROMPT = (
    "Rewrite this lecture excerpt into formal study notes of 3 to 5 short "
    "sentences. Keep only key ideas. Remove filler, anecdotes, and all "
    "first-person or second-person wording. Use objective language.\n\n"
    "Excerpt:\n{text}\n\nFormal notes:"
)

_BULLET_PROMPT = (
    "Convert the formal notes below into exactly {n} concise bullet points. "
    "Each bullet must be a complete declarative sentence in neutral style and "
    "must not contain first-person or second-person wording. Return one bullet "
    "per line, each beginning with '- '.\n\nFormal notes:\n{text}\n\nBullets:"
)

_BULLET_REWRITE_PROMPT = (
    "Rewrite the sentence below into one concise objective study-note sentence "
    "of at most 18 words. Remove conversational style and any first-person or "
    "second-person wording. Keep the original meaning. Output only one sentence.\n\n"
    "Sentence:\n{text}\n\nRewrite:"
)

_SECTION_IDEA_PROMPT = (
    "Summarize the central topic of this passage in one neutral sentence. Focus "
    "on the main concept, not speaker phrasing. Avoid first-person wording and "
    "conversational language.\n\nPassage:\n{text}\n\nCore idea:"
)

_TITLE_PROMPT = (
    "Write one complete topical heading of 4 to 8 words for this core idea. "
    "Prefer a noun-phrase style heading, not a conversational sentence. Use "
    "title case, no quotation marks, and no trailing punctuation. Output only "
    "the heading.\n\n"
    "Example core idea: Engineers must review AI-generated code and define constraints.\n"
    "Example heading: Human Oversight for AI Code\n\n"
    "Example core idea: Version control visuals improve understanding of commit history.\n"
    "Example heading: Visual Git Workflows for Teams\n\n"
    "Core idea:\n{text}\n\nHeading:"
)

_TITLE_REPAIR_PROMPT = (
    "Rewrite this heading as a complete, non-truncated title of 4 to 8 words "
    "capturing the same idea. Keep it topic-focused and noun-phrase style. No "
    "trailing punctuation. Output only the title.\n\n"
    "Heading draft:\n{text}\n\nFinal heading:"
)

_PERSONAL_RE = re.compile(
    r"\b(i|me|my|mine|myself|we|us|our|ours|ourselves|you|your|yours|yourself|yourselves)\b",
    re.IGNORECASE,
)

_CONVERSATIONAL_RE = re.compile(
    r"\b(uh|um|basically|honestly|literally|actually|kind of|kinda|sort of|"
    r"you know|i think|i guess|i mean|let's|lets|wanna|gonna|like|know|mean)\b",
    re.IGNORECASE,
)

_BROKEN_RE = re.compile(r"(^|\s)'(re|ve|ll|d|m)\b", re.IGNORECASE)

_SKIP_SENTENCE_RE = re.compile(
    r"(end of this video|don't forget|like button|let me know|referral link|"
    r"sponsoring this video|see you next time|goodbye|talk about that later)",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_TITLE_BAD_ENDINGS = {
    "and", "or", "to", "for", "with", "of", "in", "on", "at", "from", "that", "this",
    "is", "are", "was", "were", "be", "being", "been", "those", "these", "there",
    "all", "some", "many", "most", "more", "less", "only", "whatever", "best", "final",
}

_TITLE_BAD_STARTS = {
    "and", "or", "but", "so", "it", "its", "this", "that", "there", "i", "we", "you", "a", "an", "the",
}

_TITLE_VERBS = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "can", "could",
    "will", "would", "should", "may", "might", "do", "does", "did", "using", "able", "going",
}

_BAD_BULLET_ENDINGS = {
    "of", "to", "and", "or", "that", "this", "with", "in", "on", "for", "the", "a", "an",
    "self", "all", "as", "about", "by", "from",
}


@lru_cache(maxsize=4)
def _load_pipeline(model_name: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _generate(bundle, prompt: str, max_new_tokens: int, min_new_tokens: int = 1) -> str:
    tokenizer, model, device = bundle
    import torch

    if len(prompt) > 6000:
        prompt = prompt[:6000]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(tokens[0], skip_special_tokens=True).strip()


def get_summarizer(
    model_name: str = DEFAULT_SUMMARIZER,
) -> Callable[[str, int], List[str]]:
    """Return a callable `(text, n_bullets) -> list[str]` of rewritten bullets."""
    bundle = _load_pipeline(model_name)

    def _run(text: str, n_bullets: int) -> List[str]:
        text = text.strip()
        if not text:
            return []

        # Select salient candidate sentences first, then rewrite each sentence
        # into an objective bullet. This is more stable than generating an
        # entire bullet list in one shot on noisy transcripts.
        bullets = _select_salient_sentences(text, max(n_bullets * 2, 4))
        if not bullets:
            bullets = _split_sentences(text)

        cleaned: List[str] = []
        for line in bullets:
            source = _sanitize_source_sentence(line)
            if not source:
                continue

            line = _rewrite_line(bundle, source)
            line = _normalize_bullet_text(line)
            if not line:
                continue
            if _line_needs_repair(line) or len(line.split()) > 24:
                line = _rewrite_line(bundle, line)
                line = _normalize_bullet_text(line)
            if _line_needs_repair(line):
                line = _rewrite_line(bundle, line)
                line = _normalize_bullet_text(line)
            if _line_needs_repair(line):
                line = _neutralize_personal_language(line)
                line = _normalize_bullet_text(line)
            if _line_needs_repair(line):
                continue
            if _is_bad_bullet(line):
                continue
            if line and line not in cleaned:
                cleaned.append(line)
            if len(cleaned) >= n_bullets:
                break

        if cleaned:
            return cleaned[:n_bullets]
        return _fallback_bullets(text, n_bullets)

    return _run


def get_headline_generator(
    model_name: str = DEFAULT_SUMMARIZER,
) -> Callable[[str], str]:
    """Return a callable producing a short, complete heading for a section."""
    bundle = _load_pipeline(model_name)

    def _run(text: str) -> str:
        text = text.strip()
        if not text:
            return "Section"

        salient = _select_salient_sentences(text, 4)
        distilled: List[str] = []
        for sentence in salient[:3]:
            source = _sanitize_source_sentence(sentence, max_words=30)
            if not source:
                continue
            line = _rewrite_line(bundle, source)
            line = _normalize_bullet_text(line)
            if not line or _line_needs_repair(line):
                continue
            distilled.append(line.rstrip(".!?"))

        focus_text = " ".join(distilled).strip() if distilled else (" ".join(salient).strip() if salient else text)

        concept = _generate(
            bundle,
            _SECTION_IDEA_PROMPT.format(text=focus_text),
            max_new_tokens=48,
            min_new_tokens=14,
        )

        raw_title = _generate(
            bundle,
            _TITLE_PROMPT.format(text=concept),
            max_new_tokens=16,
            min_new_tokens=4,
        )
        title = _clean_title(raw_title)

        if _title_needs_repair(title):
            repaired = _generate(
                bundle,
                _TITLE_REPAIR_PROMPT.format(text=title or concept),
                max_new_tokens=16,
                min_new_tokens=4,
            )
            repaired_title = _clean_title(repaired)
            if repaired_title:
                title = repaired_title

        if _title_needs_repair(title):
            title = _fallback_title(concept)

        return title or "Section"

    return _run


def _rewrite_line(bundle, line: str) -> str:
    raw = _generate(
        bundle,
        _BULLET_REWRITE_PROMPT.format(text=line),
        max_new_tokens=26,
        min_new_tokens=6,
    )
    parsed = _parse_bullets(raw, 1)
    return parsed[0] if parsed else raw


def _parse_bullets(raw: str, n_bullets: int) -> List[str]:
    """Split model output into likely bullet strings."""
    lines: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if ":" in line and line.lower().startswith(("bullet", "bullets", "rewrite", "rewritten")):
            line = line.split(":", 1)[1].strip()
        line = line.lstrip("-*0123456789.) ").strip()
        if not line:
            continue
        lines.append(line)

    if len(lines) <= 1 and raw:
        candidate = lines[0] if lines else raw.strip()
        candidate = candidate.replace(" - ", "\n").replace(";", "\n")
        split = [part.strip(" -*•") for part in candidate.split("\n") if part.strip()]
        if len(split) > 1:
            lines = split
        else:
            lines = _split_sentences(candidate)

    return lines[:n_bullets]


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    bits = []
    for part in text.split(";"):
        bits.extend(_SENTENCE_SPLIT_RE.split(part))
    out: List[str] = []
    for piece in bits:
        piece = piece.strip()
        if not piece:
            continue
        # Recover additional sentence boundaries in long run-ons.
        if len(piece.split()) > 40 and ", and " in piece:
            sub = [s.strip() for s in piece.split(", and ") if s.strip()]
            out.extend(sub)
        else:
            out.append(piece)
    return out


def _normalize_bullet_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(excerpt|bullet|bullets|rewrite|rewritten)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip(" -*")
    text = _CONVERSATIONAL_RE.sub("", text)
    text = re.sub(r"\b(let me know|talk about it|like and subscribe)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(i\'ll|we\'ll|you\'ll|i\'ve|we\'ve|you\'ve)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(and|but|so|or)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    if not text:
        return ""
    if text.startswith("'"):
        return ""
    if len(text.split()) < 3:
        return ""
    last = text.split()[-1].lower().strip(".,;:!?")
    if last in _BAD_BULLET_ENDINGS:
        return ""
    if text[0].islower():
        text = text[0].upper() + text[1:]
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def _line_needs_repair(text: str) -> bool:
    return bool(_PERSONAL_RE.search(text) or _CONVERSATIONAL_RE.search(text) or _BROKEN_RE.search(text))


def _neutralize_personal_language(text: str) -> str:
    substitutions = [
        (r"\bI am\b", "The speaker is"),
        (r"\bI'm\b", "The speaker is"),
        (r"\bI've\b", "The speaker has"),
        (r"\bI'll\b", "The speaker will"),
        (r"\bI'd\b", "The speaker would"),
        (r"\bwe are\b", "The team is"),
        (r"\bwe're\b", "The team is"),
        (r"\bwe've\b", "The team has"),
        (r"\bwe'll\b", "The team will"),
        (r"\byou are\b", "Developers are"),
        (r"\byou\b", "developers"),
        (r"\byour\b", "developer"),
        (r"\bour\b", "team"),
        (r"\bmy\b", "the speaker's"),
    ]
    out = text
    for pattern, repl in substitutions:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    out = _BROKEN_RE.sub(" ", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _strip_personal_words(text: str) -> str:
    text = re.sub(r"\b(i|we)\s+(think|guess|believe|feel)\b\s*(that)?", "", text, flags=re.IGNORECASE)
    text = _PERSONAL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _clean_title(raw: str) -> str:
    """Normalize generated heading text."""
    title = raw.strip().splitlines()[0].strip()
    title = re.sub(r"^(title|heading)\s*:\s*", "", title, flags=re.IGNORECASE)
    title = title.lstrip("-*•").strip()
    title = title.strip("\"'`").strip()
    title = title.split(",", 1)[0].strip()
    title = title.replace("'", "")
    title = title.rstrip(" .!?,;:--").strip()

    words = title.split()
    if len(words) > 6:
        title = " ".join(words[:6]).rstrip(" .!?,;:--")

    words = title.split()
    if len(words) >= 2 and words[-1].lower() in _TITLE_BAD_ENDINGS:
        title = " ".join(words[:-1]).strip()

    if title == title.lower():
        title = title.title()
    title = re.sub(r"\bAi\b", "AI", title)
    return title


def _title_needs_repair(title: str) -> bool:
    if not title:
        return True
    if "'" in title or "," in title:
        return True
    words = title.split()
    if len(words) < 3 or len(words) > 6:
        return True
    if words[0].lower() in _TITLE_BAD_STARTS:
        return True
    if title.lower().startswith("its "):
        return True
    if words[-1].lower() in _TITLE_BAD_ENDINGS:
        return True
    if any(w.lower().strip(".,;:!?") in _TITLE_VERBS for w in words):
        return True
    if _PERSONAL_RE.search(title):
        return True
    if title.lower().startswith("main idea"):
        return True
    return False


def _fallback_title(text: str) -> str:
    text = _PERSONAL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in _TOKEN_RE.findall(text)]
    cleaned: List[str] = []
    for w in words:
        lw = w.lower().strip(".,;:!?")
        if lw in _STOPWORDS or lw in _TITLE_VERBS or lw in _TITLE_BAD_STARTS or lw in _TITLE_BAD_ENDINGS:
            continue
        if lw in {"its", "it's", "s"}:
            continue
        cleaned.append(w)

    # Preserve order while deduplicating tokens.
    seen = set()
    uniq: List[str] = []
    for w in cleaned:
        lw = w.lower()
        if lw in seen:
            continue
        seen.add(lw)
        uniq.append(w)

    if len(uniq) >= 3:
        title = " ".join(uniq[:5]).strip(" .!?,;:--")
    else:
        if not words:
            return "Section"
        title = " ".join(words[:6]).strip(" .!?,;:--")
    if len(title.split()) < 3:
        return "Section"
    title = title.replace("'", "")
    if title == title.lower():
        title = title.title()
    title = re.sub(r"\bAi\b", "AI", title)
    return title


def _fallback_bullets(text: str, n_bullets: int) -> List[str]:
    out: List[str] = []
    for sentence in _split_sentences(text):
        line = _normalize_bullet_text(_strip_personal_words(sentence))
        if not line:
            continue
        if _is_bad_bullet(line):
            continue
        out.append(line)
        if len(out) >= n_bullets:
            break
    return out


def _select_salient_sentences(text: str, k: int) -> List[str]:
    sentences = _split_sentences(text)
    sentences = [s for s in sentences if not _SKIP_SENTENCE_RE.search(s)]
    sentences = [s for s in sentences if len(s.split()) >= 6]
    if not sentences:
        return []
    if len(sentences) <= k:
        return sentences

    tokenized = []
    freq: dict[str, int] = {}
    for sentence in sentences:
        words = [w.lower() for w in _TOKEN_RE.findall(sentence)]
        words = [w for w in words if w not in _STOPWORDS and len(w) > 2]
        tokenized.append(words)
        seen = set(words)
        for w in seen:
            freq[w] = freq.get(w, 0) + 1

    scored = []
    for idx, words in enumerate(tokenized):
        if not words:
            score = 0.0
        else:
            score = sum(freq.get(w, 0) for w in words) / (len(words) ** 0.5)
        # Mild preference for earlier context without forcing first-sentence titles.
        score += 0.1 * (1.0 - (idx / max(1, len(sentences) - 1)))
        scored.append((idx, score))

    top_idx = [i for i, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]
    top_idx.sort()
    return [sentences[i] for i in top_idx]


def _sanitize_source_sentence(text: str, max_words: int = 28) -> str:
    text = text.strip()
    if not text:
        return ""
    if _SKIP_SENTENCE_RE.search(text):
        return ""
    text = _CONVERSATIONAL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;:-")

    if "," in text:
        clauses = [c.strip() for c in re.split(r",+", text) if c.strip()]
        if clauses:
            def _clause_score(clause: str) -> float:
                words = [w.lower() for w in _TOKEN_RE.findall(clause)]
                words = [w for w in words if w not in _STOPWORDS and len(w) > 2]
                return float(len(words))

            best = max(clauses, key=_clause_score)
            if len(best.split()) >= 6:
                text = best

    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
    text = " ".join(words).strip()
    if len(text.split()) < 6:
        return ""
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def _is_bad_bullet(text: str) -> bool:
    words = text.split()
    if not words:
        return True
    tail = words[-1].lower().strip(".,;:!?")
    if tail in _BAD_BULLET_ENDINGS:
        return True
    if len(words) > 24 and text.count(",") > 4:
        return True
    return False
