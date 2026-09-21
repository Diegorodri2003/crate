"""index.py — B: an in-memory search index over SampleRecord.

Pure functions over the shapes in contracts.py. No database: upsert()
keeps records in a module-level dict keyed by `path`, and save()/load()
persist that dict as plain JSON. That dict is the only mutable state.

`_embed_text` is a deterministic hashed pseudo-embedding standing in for
the CLAP text encoder analyzer.py doesn't have yet — a drop-in swap later.

CLI demo: python index.py "dark kick"
"""

import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict
from difflib import SequenceMatcher

from contracts import BANDS_HZ, SampleRecord, SearchHit, SearchQuery

_EMBED_DIM = 512

_records: dict[str, SampleRecord] = {}

# storage: upsert / all_records / save / load

def upsert(records: list[SampleRecord]) -> None:
    """Insert or replace records by path (the primary key)."""
    for r in records:
        _records[r.path] = r

def all_records() -> list[SampleRecord]:
    return list(_records.values())

def save(path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(r) for r in _records.values()], f, indent=2)

def load(path: str) -> None:
    with open(path) as f:
        rows = json.load(f)
    upsert([SampleRecord(**row) for row in rows])

# numeric helpers

def _cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def _embed_text(text: str) -> list[float]:
    """Deterministic hashed bag-of-words vector, same width as
    SampleRecord.embedding so cosine similarity is always well-defined."""
    vec = [0.0] * _EMBED_DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        idx = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "big") % _EMBED_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec

# key / circle-of-fifths helpers

_NOTE_SEMITONE = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "Fb": 4, "F": 5, "E#": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}
_SEMITONE_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def _parse_key(key: str) -> tuple[str, bool]:
    if key.endswith("m"):
        return key[:-1], True
    return key, False

def _format_key(semitone: int, minor: bool) -> str:
    return _SEMITONE_NOTE[semitone % 12] + ("m" if minor else "")

def _relative_key(key: str) -> str | None:
    note, minor = _parse_key(key)
    semi = _NOTE_SEMITONE.get(note)
    if semi is None:
        return None
    return _format_key(semi + 3, False) if minor else _format_key(semi - 3, True)

def _fifths_neighbours(key: str) -> tuple[str | None, str | None]:
    note, minor = _parse_key(key)
    semi = _NOTE_SEMITONE.get(note)
    if semi is None:
        return None, None
    return _format_key(semi + 7, minor), _format_key(semi - 7, minor)

def _key_relation(query_key: str, candidate_key: str) -> str | None:
    """Reason string if compatible (same / relative / fifths-neighbour), else None."""
    if candidate_key == query_key:
        return "same key"
    if candidate_key == _relative_key(query_key):
        _, minor = _parse_key(candidate_key)
        return "relative minor" if minor else "relative major"
    if candidate_key in _fifths_neighbours(query_key):
        return "neighbouring key (fifths)"
    return None

# bpm helpers (with half/double-time tolerance)

def _in_range(v: float, lo: float | None, hi: float | None) -> bool:
    return (lo is None or v >= lo) and (hi is None or v <= hi)

def _bpm_match(bpm: float, lo: float | None, hi: float | None):
    """(effective_bpm, reason) if bpm or its half/double is in [lo, hi],
    else None. reason is None for a direct hit, set for half/double."""
    for mult, label in ((1.0, None), (2.0, "double time"), (0.5, "half time")):
        v = bpm * mult
        if _in_range(v, lo, hi):
            return v, (None if label is None else f"{bpm:g} -> {v:g} ({label})")
    return None

# hard filters

def _passes_filters(r: SampleRecord, q: SearchQuery) -> list[str] | None:
    """None if r is filtered out, else the filter-derived reasons."""
    if q.family and r.family != q.family:
        return None
    if q.instrument and r.instrument != q.instrument:
        return None
    if q.sample_type and r.sample_type != q.sample_type:
        return None

    reasons: list[str] = []

    if q.bpm_min is not None or q.bpm_max is not None:
        if r.bpm is None:
            return None
        match = _bpm_match(r.bpm, q.bpm_min, q.bpm_max)
        if match is None:
            return None
        _, reason = match
        if reason:
            reasons.append(reason)

    if q.key is not None:
        if r.key is None:
            reasons.append("atonal (key-agnostic)")
        else:
            relation = _key_relation(q.key, r.key)
            if relation is None:
                return None
            reasons.append(relation)

    if r.error:
        reasons.append(f"flagged: {r.error}")

    return reasons

# text scoring

def _text_score(r: SampleRecord, q_text: str, q_embedding: list[float]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    tokens = re.findall(r"[a-z0-9]+", q_text.lower())

    hit_descriptors = [d for d in r.descriptors if d.lower() in tokens]
    if hit_descriptors:
        reasons.append("matches " + ", ".join(hit_descriptors))

    if r.embedding is not None:
        sim = _cosine(q_embedding, r.embedding)
        score = max(0.0, min(1.0, (sim + 1) / 2))
        reasons.append("semantic match")
        return score, reasons

    # No embedding for this record (e.g. it errored out): fall back to
    # fuzzy text matching over filename + instrument + family + descriptors.
    haystack = " ".join([r.filename, r.instrument, r.family, *r.descriptors]).lower()
    score = SequenceMatcher(None, q_text.lower(), haystack).ratio()
    token_hits = sum(1 for t in tokens if t in haystack)
    if tokens:
        score = max(score, token_hits / len(tokens))
    if token_hits:
        reasons.append("keyword match in filename/tags")
    return score, reasons

# fit-to-reference scoring

_KEY_RELATION_VALUE = {"same key": 1.0, "relative major": 0.8,
                        "relative minor": 0.8, "neighbouring key (fifths)": 0.6}

def _fit_score(candidate: SampleRecord, reference: SampleRecord, contrast: bool) -> tuple[float, list[str]]:
    """Weighted-average fit score; a signal is skipped (not zeroed) when
    its data is missing, so partial records never crash or get punished
    for a signal nobody could compute."""
    weighted: list[tuple[float, float]] = []
    reasons: list[str] = []

    if len(candidate.band_energy) == 8 and len(reference.band_energy) == 8:
        gap = sum(c * (1 - b) for c, b in zip(candidate.band_energy, reference.band_energy))
        weighted.append((0.4, gap))
        weakest = min(range(8), key=lambda i: reference.band_energy[i])
        lo, hi = BANDS_HZ[weakest]
        band = f"{lo/1000:g}-{hi/1000:g}kHz" if hi >= 1000 else f"{lo}-{hi}Hz"
        reasons.append(f"fills {band}")

    if candidate.bpm is not None and reference.bpm:
        best_pct = best_val = best_label = None
        for mult, label in ((1.0, None), (2.0, "double time"), (0.5, "half time")):
            v = candidate.bpm * mult
            pct = (v - reference.bpm) / reference.bpm * 100
            if best_pct is None or abs(pct) < abs(best_pct):
                best_pct, best_val, best_label = pct, v, label
        closeness = max(0.0, 1.0 - abs(best_pct) / 100.0)
        weighted.append((0.2, closeness))
        if best_label:
            reasons.append(f"{candidate.bpm:g} -> {best_val:g} ({best_label})")
        else:
            sign = "+" if best_pct >= 0 else ""
            reasons.append(f"{candidate.bpm:g} -> {reference.bpm:g} ({sign}{best_pct:.1f}%)")

    if candidate.key is not None and reference.key is not None:
        relation = _key_relation(reference.key, candidate.key)
        weighted.append((0.2, _KEY_RELATION_VALUE.get(relation, 0.0)))
        if relation:
            reasons.append(relation)

    if candidate.embedding is not None and reference.embedding is not None:
        sim = (_cosine(candidate.embedding, reference.embedding) + 1) / 2
        if contrast:
            sim = 1.0 - sim
        weighted.append((0.2, sim))
        reasons.append("contrasting character" if contrast else "similar character")

    if not weighted:
        return 0.5, reasons  # nothing comparable — neutral, not a crash

    total_w = sum(w for w, _ in weighted)
    score = sum(w * v for w, v in weighted) / total_w
    return max(0.0, min(1.0, score)), reasons

# search

def search(q: SearchQuery) -> list[SearchHit]:
    q_embedding = _embed_text(q.text) if q.text else None
    reference = _records.get(q.fit_to) if q.fit_to else None

    hits: list[SearchHit] = []
    for r in all_records():
        if q.fit_to and r.path == q.fit_to:
            continue  # don't recommend fitting a track to itself

        filter_reasons = _passes_filters(r, q)
        if filter_reasons is None:
            continue

        scores: list[float] = []
        reasons = list(filter_reasons)

        if q.text:
            s, rs = _text_score(r, q.text, q_embedding)
            scores.append(s)
            reasons.extend(rs)

        if reference is not None:
            s, rs = _fit_score(r, reference, q.contrast)
            scores.append(s)
            reasons.extend(rs)

        score = sum(scores) / len(scores) if scores else 1.0

        if len(reasons) < 2:
            reasons.append(f"{r.instrument} · {r.family}")
        reasons = reasons[:4]

        hits.append(SearchHit(record=r, score=round(score, 4), reasons=reasons))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[: q.limit]

# CLI demo

def _print_hits(hits: list[SearchHit]) -> None:
    if not hits:
        print("no matches")
        return
    for rank, hit in enumerate(hits, 1):
        print(f"{rank:>2}. {hit.score:.3f}  {hit.record.filename}")
        print(f"      {' | '.join(hit.reasons)}")

def _load_real_records(root: str) -> None:
    """Analyse every file under root with analyzer.analyze(). Falls back to
    the JSON fixtures (with a banner) if analyzer.py is missing or broken,
    so the demo still runs."""
    try:
        from analyzer import analyze
        paths = [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs]
        upsert([analyze(p) for p in paths])
    except Exception as exc:
        print(f"[index] analyzer unavailable ({exc}); falling back to fixtures", file=sys.stderr)
        here = os.path.dirname(os.path.abspath(__file__))
        load(os.path.join(here, "fixtures", "records.json"))


if __name__ == "__main__":
    _load_real_records(os.path.join(os.path.dirname(os.path.abspath(__file__)), "messy"))

    query_text = " ".join(sys.argv[1:]) or None
    _print_hits(search(SearchQuery(text=query_text, limit=10)))
