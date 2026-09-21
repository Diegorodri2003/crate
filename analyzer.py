"""A — analyze a single audio path into a SampleRecord. Never raises.

CLI: python analyzer.py <file-or-folder> prints one JSON record per line.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict
from typing import Optional, Sequence

import numpy as np
import librosa
import soundfile as sf

from contracts import BANDS_HZ, FAMILIES, INSTRUMENTS, SampleRecord, SampleType

ANALYSIS_SR = 22050
AUDIO_EXT = {".wav", ".aiff", ".aif", ".flac", ".mp3", ".ogg", ".m4a"}

# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key profiles — 12 major + 12 minor, one per tonic.
# Profile index 0 aligns with chroma bin 0 (C), matching librosa's chroma_cqt
# bin ordering, so a profile for tonic i is the base profile rolled by i.
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def _build_key_profiles() -> dict[str, np.ndarray]:
    profiles: dict[str, np.ndarray] = {}
    for i, name in enumerate(_NOTE_NAMES):
        profiles[name] = np.roll(_MAJOR_PROFILE, i)
        profiles[f"{name}m"] = np.roll(_MINOR_PROFILE, i)
    return profiles


_KEY_PROFILES = _build_key_profiles()

# ---------------------------------------------------------------------------
# Fallback filename keywords, used only when CLAP is unavailable.
# ---------------------------------------------------------------------------

_FILENAME_KEYWORDS: list[tuple[str, str, str]] = [
    ("kick", "Drums", "kick"),
    ("snare", "Drums", "snare"),
    ("clap", "Drums", "clap"),
    ("hihat", "Drums", "hat"),
    ("hat", "Drums", "hat"),
    ("hh", "Drums", "hat"),
    ("ride", "Drums", "ride"),
    ("tom", "Drums", "tom"),
    ("perc", "Drums", "perc"),
    ("808", "Bass", "808"),
    ("sub", "Bass", "sub"),
    ("reese", "Bass", "reese"),
    ("bass", "Bass", "plucked-bass"),
    ("violin", "Acoustic", "violin"),
    ("guitar", "Acoustic", "guitar"),
    ("piano", "Acoustic", "piano"),
    ("strings", "Acoustic", "strings"),
    ("brass", "Acoustic", "brass"),
    ("woodwind", "Acoustic", "woodwind"),
    ("pad", "Synth", "pad"),
    ("lead", "Synth", "lead"),
    ("pluck", "Synth", "pluck"),
    ("arp", "Synth", "arp"),
    ("stab", "Synth", "stab"),
    ("vox", "Vocal", "vocal-phrase"),
    ("vocal", "Vocal", "vocal-oneshot"),
    ("acapella", "Vocal", "acapella"),
    ("chop", "Vocal", "chop"),
    ("riser", "FX", "riser"),
    ("impact", "FX", "impact"),
    ("downlifter", "FX", "downlifter"),
    ("foley", "FX", "foley"),
    ("ambience", "FX", "ambience"),
]

# ---------------------------------------------------------------------------
# CLAP zero-shot classifier — lazy-loaded, never imported at module scope.
# ---------------------------------------------------------------------------

_CLAP_MODEL_NAME = "laion/clap-htsat-unfused"
_CLAP_MODEL = None
_CLAP_PROCESSOR = None
_CLAP_UNAVAILABLE = False

_PROMPT_TEMPLATES = ["a {label} sample", "an isolated {label} one-shot"]


def _build_prompts() -> tuple[list[str], list[tuple[str, str]]]:
    prompts: list[str] = []
    mapping: list[tuple[str, str]] = []
    for family, instruments in INSTRUMENTS.items():
        if family == "unsorted":
            continue
        for instrument in instruments:
            label = instrument.replace("-", " ")
            for tmpl in _PROMPT_TEMPLATES:
                prompts.append(tmpl.format(label=label))
                mapping.append((family, instrument))
    return prompts, mapping


_PROMPTS, _PROMPT_MAP = _build_prompts()


def _load_clap():
    """Load CLAP into module globals on first use. Any failure disables it
    for the rest of the process so the fallback path runs instead."""
    global _CLAP_MODEL, _CLAP_PROCESSOR, _CLAP_UNAVAILABLE
    if _CLAP_UNAVAILABLE:
        return None, None
    if _CLAP_MODEL is not None and _CLAP_PROCESSOR is not None:
        return _CLAP_MODEL, _CLAP_PROCESSOR
    try:
        from transformers import ClapModel, ClapProcessor

        model = ClapModel.from_pretrained(_CLAP_MODEL_NAME)
        processor = ClapProcessor.from_pretrained(_CLAP_MODEL_NAME)
        model.eval()
        _CLAP_MODEL, _CLAP_PROCESSOR = model, processor
        return model, processor
    except Exception:
        _CLAP_UNAVAILABLE = True
        return None, None


def _fit_dim(vec: Sequence[float], n: int = 512) -> list[float]:
    vec = [float(x) for x in vec]
    if len(vec) >= n:
        return vec[:n]
    return vec + [0.0] * (n - len(vec))


def _classify_with_clap(y: np.ndarray, sr: int) -> Optional[tuple[str, str, float, list[float]]]:
    model, processor = _load_clap()
    if model is None or processor is None:
        return None
    try:
        import torch

        target_sr = 48000
        audio = y if sr == target_sr else librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
        if audio.size == 0:
            return None
        inputs = processor(
            text=_PROMPTS, audio=audio, sampling_rate=target_sr, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits_per_audio, dim=-1)[0].tolist()
            embedding = outputs.audio_embeds[0].tolist()
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        family, instrument = _PROMPT_MAP[best_idx]
        confidence = float(probs[best_idx])
        return family, instrument, confidence, _fit_dim(embedding)
    except Exception:
        return None


def _classify_fallback(filename: str) -> tuple[str, str, float]:
    lower = filename.lower()
    for needle, family, instrument in _FILENAME_KEYWORDS:
        if needle in lower:
            return family, instrument, 0.4
    return "unsorted", "unknown", 0.4


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------


def is_loop(onsets: Sequence[float], duration: float, bpm: Optional[float]) -> bool:
    """Pure predicate: True only if all three loop conditions hold.

    (a) at least 3 onsets
    (b) inter-onset intervals are regular (stdev / mean < 0.25)
    (c) duration is within 8% of 60/bpm * n for n in {1, 2, 4, 8}

    Anything under 0.3s is always a one-shot.
    """
    if duration < 0.3:
        return False
    if bpm is None or bpm <= 0:
        return False
    times = sorted(onsets)
    if len(times) < 3:
        return False
    intervals = [b - a for a, b in zip(times, times[1:]) if b > a]
    if len(intervals) < 2:
        return False
    mean = sum(intervals) / len(intervals)
    if mean <= 0:
        return False
    variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
    stdev = math.sqrt(variance)
    if stdev / mean >= 0.25:
        return False
    beat = 60.0 / bpm
    for bars in (1, 2, 4, 8):
        target = beat * bars
        if target <= 0:
            continue
        if abs(duration - target) <= 0.08 * target:
            return True
    return False


def _sample_type(onsets: Sequence[float], duration: float, bpm: Optional[float]) -> SampleType:
    if duration < 0.3:
        return "oneshot"
    return "loop" if is_loop(onsets, duration, bpm) else "oneshot"


def _peak_db(y: np.ndarray) -> float:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-9:
        return -100.0
    return max(-100.0, min(0.0, round(20.0 * math.log10(peak), 2)))


def _band_energy(y: np.ndarray, sr: int) -> list[float]:
    if y.size == 0:
        return [round(1.0 / 8, 6)] * 8
    spec = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2 * (spec.shape[0] - 1))
    raw = []
    for lo, hi in BANDS_HZ:
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            raw.append(0.0)
            continue
        raw.append(float(np.sqrt(np.mean(spec[mask, :] ** 2))))
    total = sum(raw)
    if total <= 0:
        return [round(1.0 / 8, 6)] * 8
    normalized = [round(b / total, 6) for b in raw]
    diff = round(1.0 - sum(normalized), 6)
    normalized[-1] = round(normalized[-1] + diff, 6)
    return normalized


def _onsets(y: np.ndarray, sr: int) -> list[float]:
    if y.size < 512:
        return []
    try:
        times = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    except Exception:
        return []
    return [round(float(t), 3) for t in times]


def _bpm(y: np.ndarray, sr: int, duration: float, onsets: Sequence[float]) -> Optional[float]:
    if duration < 1.0 or len(onsets) < 3:
        return None
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except Exception:
        return None
    tempo_val = float(np.atleast_1d(tempo)[0])
    if not math.isfinite(tempo_val) or tempo_val <= 0:
        return None
    return round(tempo_val, 1)


def _key(y: np.ndarray, sr: int) -> Optional[str]:
    if y.size < sr * 0.1:
        return None
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    except Exception:
        return None
    if not math.isfinite(flatness) or flatness > 0.4:
        return None
    chroma_mean = np.nan_to_num(np.mean(chroma, axis=1))
    if np.std(chroma_mean) == 0:
        return None
    best_name, best_corr = None, -1.0
    for name, profile in _KEY_PROFILES.items():
        corr = float(np.corrcoef(chroma_mean, profile)[0, 1])
        if math.isfinite(corr) and corr > best_corr:
            best_name, best_corr = name, corr
    if best_name is None or best_corr < 0.6:
        return None
    return best_name


def _descriptors(y: np.ndarray, sr: int) -> list[str]:
    if y.size == 0:
        return []
    try:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    except Exception:
        centroid, flatness = 0.0, 0.0
    rms_val = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    crest_db = 20.0 * math.log10(peak / rms_val) if rms_val > 1e-9 and peak > 0 else 0.0

    words: list[str] = []
    if math.isfinite(centroid):
        if centroid < 1200:
            words.append("dark")
        elif centroid > 4500:
            words.append("bright")
    if crest_db > 14:
        words.append("punchy")
    elif crest_db < 6:
        words.append("compressed")
    if math.isfinite(flatness):
        if flatness > 0.35:
            words.append("noisy")
        elif flatness < 0.05:
            words.append("tonal")
    return words[:3]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _blank_record(path: str, filename: str, error: str) -> SampleRecord:
    return SampleRecord(
        path=os.path.abspath(path) if path else path,
        filename=filename,
        duration_s=0.0,
        sample_rate=0,
        channels=0,
        peak_db=-100.0,
        sample_type="unknown",
        bpm=None,
        key=None,
        family="unsorted",
        instrument="unknown",
        confidence=0.0,
        band_energy=[0.0] * 8,
        onsets=[],
        embedding=None,
        descriptors=[],
        error=error,
    )


def _analyze_file(path: str, abs_path: str, filename: str) -> SampleRecord:
    info = sf.info(path)
    sample_rate = int(info.samplerate)
    channels = int(info.channels)
    duration_s = round(float(info.frames) / sample_rate, 3) if sample_rate else 0.0

    y, sr = librosa.load(path, sr=ANALYSIS_SR, mono=True)
    if duration_s <= 0:
        duration_s = round(len(y) / sr, 3) if sr else 0.0

    peak_db = _peak_db(y)
    band_energy = _band_energy(y, sr)
    onsets = _onsets(y, sr)
    bpm = _bpm(y, sr, duration_s, onsets)
    sample_type = _sample_type(onsets, duration_s, bpm)
    key = _key(y, sr)
    descriptors = _descriptors(y, sr)

    clap_result = _classify_with_clap(y, sr)
    if clap_result is not None:
        family, instrument, confidence, embedding = clap_result
    else:
        family, instrument, confidence = _classify_fallback(filename)
        embedding = None

    if confidence < 0.5:
        family, instrument = "unsorted", "unknown"

    assert family in FAMILIES, f"unknown family: {family}"
    assert instrument in INSTRUMENTS[family], f"unknown instrument: {instrument} for {family}"

    return SampleRecord(
        path=abs_path,
        filename=filename,
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
        peak_db=peak_db,
        sample_type=sample_type,
        bpm=bpm,
        key=key,
        family=family,
        instrument=instrument,
        confidence=round(confidence, 3),
        band_energy=band_energy,
        onsets=onsets,
        embedding=embedding,
        descriptors=descriptors,
        error=None,
    )


def analyze(path: str) -> SampleRecord:
    filename = os.path.basename(path) if path else ""
    if not path or not os.path.isfile(path):
        return _blank_record(path, filename, "file not found")
    try:
        abs_path = os.path.abspath(path)
        record = _analyze_file(path, abs_path, filename)
        assert record.family in FAMILIES
        assert record.instrument in INSTRUMENTS[record.family]
        return record
    except Exception as exc:
        return _blank_record(path, filename, f"could not decode: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _iter_paths(target: str):
    if os.path.isdir(target):
        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = sorted(d for d in dirnames if d not in {".git", "__pycache__"})
            for name in sorted(filenames):
                if os.path.splitext(name)[1].lower() in AUDIO_EXT:
                    yield os.path.join(dirpath, name)
        return
    # a single file path — yield it even if missing so analyze() can report
    # the error instead of the CLI silently printing nothing.
    yield target


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python analyzer.py <file-or-folder>", file=sys.stderr)
        return 1
    for path in _iter_paths(argv[1]):
        print(json.dumps(asdict(analyze(path))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
