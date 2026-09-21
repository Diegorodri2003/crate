"""B — in-memory sample index. upsert + search over SampleRecord."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Iterable, Optional

import numpy as np

from contracts import SearchHit, SearchQuery, SampleRecord

_STORE: dict[str, SampleRecord] = {}
_DEFAULT_DB = os.path.join("fixtures", "index.json")


def clear() -> None:
    _STORE.clear()


def all_records() -> list[SampleRecord]:
    return list(_STORE.values())


def upsert(records: Iterable[SampleRecord]) -> None:
    for rec in records:
        _STORE[rec.path] = rec


def _cosine(a: Optional[list[float]], b: Optional[list[float]]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(va, vb) / (na * nb))


def _band_score(a: list[float], b: list[float], contrast: bool) -> float:
    if len(a) != 8 or len(b) != 8:
        return 0.0
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    if contrast:
        gap = np.maximum(vb - va, 0.0)
        return float(gap.sum() / 2.0)
    return float(1.0 - 0.5 * np.abs(va - vb).sum())


def _text_blob(rec: SampleRecord) -> str:
    parts = [
        rec.filename,
        rec.family,
        rec.instrument,
        rec.sample_type,
        rec.key or "",
        " ".join(rec.descriptors),
        rec.path,
    ]
    return " ".join(parts).lower()


def _score(rec: SampleRecord, query: SearchQuery, ref: Optional[SampleRecord]) -> SearchHit:
    score = 0.35
    reasons: list[str] = []

    if rec.error:
        score = 0.05
        reasons.append(f"error: {rec.error}")
        return SearchHit(record=rec, score=score, reasons=reasons)

    if query.text:
        blob = _text_blob(rec)
        tokens = [t for t in query.text.lower().split() if t]
        hits = sum(1 for t in tokens if t in blob)
        if tokens:
            frac = hits / len(tokens)
            score += 0.35 * frac
            if frac:
                reasons.append(f"text {hits}/{len(tokens)}")
            else:
                score -= 0.2

    if query.bpm_min is not None or query.bpm_max is not None:
        if rec.bpm is None:
            score -= 0.15
        else:
            lo = query.bpm_min if query.bpm_min is not None else rec.bpm
            hi = query.bpm_max if query.bpm_max is not None else rec.bpm
            if lo <= rec.bpm <= hi:
                score += 0.15
                reasons.append(f"bpm {rec.bpm}")
            else:
                score -= 0.2

    if query.key and rec.key:
        if rec.key.lower() == query.key.lower():
            score += 0.1
            reasons.append(f"key {rec.key}")

    if ref is not None:
        if rec.bpm is not None and ref.bpm is not None and ref.bpm:
            delta = abs(rec.bpm - ref.bpm) / ref.bpm
            fit = max(0.0, 1.0 - delta * 4)
            score += 0.15 * fit
            reasons.append(f"{rec.bpm} -> {ref.bpm} ({(rec.bpm - ref.bpm):+.1f})")
        band = _band_score(rec.band_energy, ref.band_energy, query.contrast)
        score += 0.2 * band
        reasons.append("fills bands" if query.contrast else "matches spectrum")
        emb = _cosine(rec.embedding, ref.embedding)
        if emb is not None:
            adjusted = (1.0 - emb) if query.contrast else max(emb, 0.0)
            score += 0.15 * adjusted
            reasons.append(f"embed {adjusted:.2f}")

    score = max(0.0, min(1.0, score))
    if not reasons:
        reasons.append("library match")
    return SearchHit(record=rec, score=round(score, 4), reasons=reasons)


def search(query: SearchQuery) -> list[SearchHit]:
    ref = _STORE.get(query.fit_to) if query.fit_to else None
    hits: list[SearchHit] = []
    for rec in _STORE.values():
        if query.family and rec.family != query.family:
            continue
        if query.instrument and rec.instrument != query.instrument:
            continue
        if query.sample_type and rec.sample_type != query.sample_type:
            continue
        if query.key and rec.key and rec.key.lower() != query.key.lower():
            continue
        hits.append(_score(rec, query, ref))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[: query.limit]


def save(path: str = _DEFAULT_DB) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in _STORE.values()], f, indent=2)


def load(path: str = _DEFAULT_DB) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        rows = json.load(f)
    recs = []
    for row in rows:
        recs.append(SampleRecord(**row))
    upsert(recs)
    return len(recs)
