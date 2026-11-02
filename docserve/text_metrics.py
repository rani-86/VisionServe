"""OCR quality metrics: character and word error rate (edit distance / reference length)."""
from __future__ import annotations

import re

try:
    from rapidfuzz.distance import Levenshtein as _lev

    def edit_distance(a, b) -> int:
        return _lev.distance(a, b)
except ImportError:  # pragma: no cover - pure-python fallback
    def edit_distance(a, b) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference).split(), normalize_text(hypothesis).split()
    return edit_distance(ref, hyp) / max(1, len(ref))
