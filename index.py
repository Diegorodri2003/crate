"""index.py — B: an in-memory search index over SampleRecord.

Pure functions over the shapes in contracts.py. No database: upsert()
keeps records in a module-level dict keyed by `path`, and save()/load()
persist that dict as plain JSON. That dict is the only mutable state.

Text search uses analyzer.embed_text (a real CLAP text encoder) when it's
importable and a record has an embedding; otherwise it always falls back
to fuzzy matching over filename/instrument/family/descriptors — no hash-
based placeholder embedding, since that only produced noise rankings.

CLI demo: python index.py "dark kick"
"""

import json
import math
import os
import re
import sys
from dataclasses import asdict

from contracts import BANDS_HZ, SampleRecord, SearchHit, SearchQuery

try:
    from analyzer import embed_text
except Exception:
    embed_text = None

_records: dict[str, SampleRecord] = {}

# True once this module has fallen back to fixtures/records.json instead of
# real analyzer output. api.py reads this to warn in the UI.
USING_FIXTURES = False


def _fixture_fallback_banner(reason: str) -> None:
    global USING_FIXTURES
    USING_FIXTURES = True
    line = "=" * 72
    print(line, file=sys.stderr)
    print("!!  index.py IS SEARCHING FIXTURES, NOT YOUR FOLDER  !!", file=sys.stderr)
    print(f"!!  reason: {reason}", file=sys.stderr)
    print("!!  every hit below comes from fixtures/records.json — fake data.", file=sys.stderr)
    print(line, file=sys.stderr)

# The persisted store is scratch state, not project data: it lives in
# .cache/ (gitignored) so a stale index can never be committed and survive
# into a later run the way it did when this sat inside fixtures/.
STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "index.json")

# storage: upsert / remove / clear / all_records / save / load

def upsert(records: list[SampleRecord]) -> None:
    """Insert or replace records by path (the primary key)."""
    for r in records:
        _records[r.path] = r

def remove(path: str) -> None:
    """Drop one record by path. A reference upload (a bounce sent to /fit)
    is only in the store so fit_to can look it up — it is removed again the
    moment that is done, so it never shows up as an ordinary library hit."""
    _records.pop(path, None)

def clear() -> None:
    """Drop every record. Callers that (re)ingest a folder call this first so
    the index reflects exactly that folder and no earlier run's paths."""
    _records.clear()

def all_records() -> list[SampleRecord]:
    return list(_records.values())

def save(path: str = STORE_PATH) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in _records.values()], f, indent=2)

def load(path: str = STORE_PATH) -> None:
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

def _hits(token: str, value: str) -> bool:
    return token in value or value in token


def _fuzzy_score(r: SampleRecord, tokens: list[str]) -> tuple[float, list[str]]:
    """Match query tokens against instrument + descriptors (high weight)
    and family + filename (low weight). Case-insensitive; a token counts
    as a hit on partial (substring) overlap, not just exact equality."""
    if not tokens:
        return 0.0, []

    instrument = r.instrument.lower()
    family = r.family.lower()
    filename = r.filename.lower()
    descriptors = [d.lower() for d in r.descriptors]

    HIGH, LOW = 1.0, 0.35
    matched: list[str] = []
    total = 0.0
    for t in tokens:
        if _hits(t, instrument) or any(_hits(t, d) for d in descriptors):
            total += HIGH
            matched.append(t)
        elif _hits(t, family) or t in filename:
            total += LOW
            matched.append(t)

    score = min(1.0, total / (len(tokens) * HIGH))
    reasons = [f"matched: {', '.join(matched)}"] if matched else []
    return score, reasons


def _text_score(r: SampleRecord, q_text: str, q_embedding: list[float] | None) -> tuple[float, list[str]]:
    if q_embedding is not None and r.embedding is not None:
        sim = _cosine(q_embedding, r.embedding)
        score = max(0.0, min(1.0, (sim + 1) / 2))
        return score, ["matched: semantic embedding"]

    # No usable embedding path for this query/record: always fuzzy-match
    # over filename + instrument + family + descriptors, never a hash.
    tokens = re.findall(r"[a-z0-9]+", q_text.lower())
    return _fuzzy_score(r, tokens)

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
    q_embedding = None
    if q.text and embed_text is not None:
        try:
            q_embedding = embed_text(q.text)
        except Exception:
            q_embedding = None
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
        _fixture_fallback_banner(f"analyzer unavailable ({exc})")
        here = os.path.dirname(os.path.abspath(__file__))
        load(os.path.join(here, "fixtures", "records.json"))


if __name__ == "__main__":
    _load_real_records(os.path.join(os.path.dirname(os.path.abspath(__file__)), "messy"))

    query_text = " ".join(sys.argv[1:]) or None
    _print_hits(search(SearchQuery(text=query_text, limit=10)))
