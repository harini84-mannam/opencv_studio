from __future__ import annotations

import re
from collections import Counter

_STOPWORDS = frozenset("""
a an the and or but if while is are was were be been being
to of in on at by for with about against between into through
during before after above below from up down out off over under
again further then once here there when where why how all any
both each few more most other some such no nor not only own same
so than too very s t can will just don should now i me my myself
we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their
theirs themselves what which who whom this that these those am
having do does did doing would could shall might must
""".split())

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u201c\"'])")
_WORD_RE = re.compile(r"[A-Za-z']+")
_CODE_TOKEN_RE = re.compile(r"[_(){}\[\];]|->|==|::")
_REAL_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
MAX_SUMMARY_CHARS = 700
MIN_REAL_WORDS_FOR_PROSE = 3
MAX_CODE_TOKEN_RATIO = 0.3


def _looks_like_code_fragment(unit: str) -> bool:
    tokens = unit.split()
    if not tokens:
        return True

    code_like = sum(1 for t in tokens if _CODE_TOKEN_RE.search(t))
    code_ratio = code_like / len(tokens)

    real_words = sum(1 for t in tokens if _REAL_WORD_RE.fullmatch(t.strip(".,;:!?\"'()")))

    return code_ratio > MAX_CODE_TOKEN_RATIO or real_words < MIN_REAL_WORDS_FOR_PROSE


def _split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    units: list[str] = []
    for line in raw_lines:
        line = re.sub(r"[ \t]+", " ", line)
        line = re.sub(r"^[\u2022\u25cf\u25aa\-\*\u2013\u2014]\s*", "", line)
        parts = _SENTENCE_SPLIT_RE.split(line)
        units.extend(p.strip() for p in parts if p.strip())

    
    return [u for u in units if len(u) >= 8]


def _join_readably(sentences: list[str]) -> str:
    parts = []
    for s in sentences:
        parts.append(s if s.endswith((".", "!", "?")) else s + ".")
    return " ".join(parts)


def _enforce_char_cap(sentences: list[str], max_chars: int = MAX_SUMMARY_CHARS) -> list[str]:
    kept = []
    total = 0
    for s in sentences:
        if total + len(s) > max_chars and kept:
            break
        kept.append(s)
        total += len(s) + 1
    return kept


def summarize_text(text: str, max_sentences: int = 3) -> str:
    
    sentences = _split_sentences(text)

    if not sentences:
       
        stripped = text.strip()
        return stripped[:MAX_SUMMARY_CHARS] + ("..." if len(stripped) > MAX_SUMMARY_CHARS else "")

    prose_candidates = [s for s in sentences if not _looks_like_code_fragment(s)]
    
    candidates = prose_candidates if prose_candidates else sentences

    if len(candidates) <= max_sentences:
        return _join_readably(_enforce_char_cap(candidates))

    word_counts: Counter[str] = Counter()
    for sentence in sentences:  
        for word in _WORD_RE.findall(sentence.lower()):
            if word not in _STOPWORDS and len(word) > 1:
                word_counts[word] += 1

    if not word_counts:
        return _join_readably(_enforce_char_cap(candidates[:max_sentences]))

    max_count = max(word_counts.values())
    word_scores = {w: c / max_count for w, c in word_counts.items()}

    scored: list[tuple[int, float, str]] = []
    for idx, sentence in enumerate(candidates):
        words = [w for w in _WORD_RE.findall(sentence.lower()) if w not in _STOPWORDS]
        if not words:
            continue
        score = sum(word_scores.get(w, 0.0) for w in words) / len(words)
        scored.append((idx, score, sentence))

    if not scored:
        return _join_readably(_enforce_char_cap(candidates[:max_sentences]))

    top = sorted(scored, key=lambda t: t[1], reverse=True)[:max_sentences]
    top.sort(key=lambda t: t[0])

    return _join_readably(_enforce_char_cap([sentence for _, _, sentence in top]))