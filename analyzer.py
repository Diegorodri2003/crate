"""A — analyze a single audio path into a SampleRecord. Never raises."""

from __future__ import annotations

import math
import os
import wave
from typing import Optional

import numpy as np

from contracts import BANDS_HZ, FAMILIES, INSTRUMENTS, SampleRecord, SampleType

FILENAME_HINTS: list[tuple[str, str, str]] = [
    ("kick", "Drums", "kick"),
    ("bd", "Drums", "kick"),
    ("snare", "Drums", "snare"),
    ("sd", "Drums", "snare"),
    ("clap", "Drums", "clap"),
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
    ("vocal", "Vocal", "vocal-oneshot"),
    ("acapella", "Vocal", "acapella"),
    ("vox", "Vocal", "vocal-phrase"),
    ("chop", "Vocal", "chop"),
    ("riser", "FX", "riser"),
    ("impact", "FX", "impact"),
    ("downlifter", "FX", "downlifter"),
    ("foley", "FX", "foley"),
    ("ambience", "FX", "ambience"),
    ("fx", "FX", "impact"),
]


def _empty(path: str, filename: str, error: str) -> SampleRecord:
    return SampleRecord(
        path=os.path.abspath(path),
        filename=filename,
        duration_s=0.0,
        sample_rate=0,
        channels=0,
        peak_db=-120.0,
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


def _read_wav(path: str) -> tuple[np.ndarray, int, int]:
    with wave.open(path, "rb") as w:
        channels = w.getnchannels()
        sr = w.getframerate()
        n = w.getnframes()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        ints = np.where(ints >= 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float32) / 8388608.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sw}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr, channels


def _peak_db(samples: np.ndarray) -> float:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0.0:
        return -120.0
    return round(20.0 * math.log10(peak), 2)


def _band_energy(samples: np.ndarray, sr: int) -> list[float]:
    if samples.size < 16:
        return [1.0 / 8] * 8
    window = np.hanning(min(len(samples), 8192))
    chunk = samples[: len(window)] * window
    spec = np.abs(np.fft.rfft(chunk)) ** 2
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
    bands = []
    for lo, hi in BANDS_HZ:
        mask = (freqs >= lo) & (freqs < hi)
        bands.append(float(spec[mask].sum()) if mask.any() else 0.0)
    total = sum(bands) or 1.0
    return [round(b / total, 4) for b in bands]


def _onsets(samples: np.ndarray, sr: int) -> list[float]:
    hop = max(256, sr // 100)
    win = hop * 2
    if len(samples) < win:
        return [0.0]
    frames = []
    for i in range(0, len(samples) - win, hop):
        frames.append(np.sqrt(np.mean(samples[i : i + win] ** 2)))
    env = np.array(frames, dtype=np.float32)
    flux = np.diff(env, prepend=env[0])
    flux = np.maximum(flux, 0.0)
    thresh = float(np.mean(flux) + 1.5 * np.std(flux))
    peaks = []
    for i in range(1, len(flux) - 1):
        if flux[i] >= thresh and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            peaks.append(round(i * hop / sr, 3))
    return peaks[:64] or [0.0]


def _bpm_from_onsets(onsets: list[float], duration_s: float) -> Optional[float]:
    if duration_s < 1.2 or len(onsets) < 3:
        return None
    ivals = np.diff(np.array(onsets, dtype=np.float64))
    ivals = ivals[(ivals > 0.12) & (ivals < 2.0)]
    if ivals.size == 0:
        return None
    period = float(np.median(ivals))
    bpm = 60.0 / period
    while bpm < 70:
        bpm *= 2
    while bpm > 190:
        bpm /= 2
    return round(bpm, 1)


def _key_from_spectrum(samples: np.ndarray, sr: int) -> Optional[str]:
    if samples.size < 512:
        return None
    n = min(len(samples), sr * 2)
    spec = np.abs(np.fft.rfft(samples[:n] * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    acc = np.zeros(12)
    for mag, f in zip(spec, freqs):
        if f < 40 or f > 2000 or mag <= 0:
            continue
        midi = 69 + 12 * math.log2(f / 440.0)
        acc[int(round(midi)) % 12] += mag
    if acc.sum() <= 0:
        return None
    root = int(np.argmax(acc))
    major = sum(acc[(root + i) % 12] for i in (0, 4, 7))
    minor = sum(acc[(root + i) % 12] for i in (0, 3, 7))
    name = notes[root]
    return name if major >= minor else f"{name}m"


def _classify(
    filename: str, bands: list[float], duration_s: float
) -> tuple[str, str, float, list[str]]:
    lower = filename.lower()
    for needle, family, instrument in FILENAME_HINTS:
        if needle in lower:
            return family, instrument, 0.86, ["filename"]
    low = sum(bands[:2])
    mid = sum(bands[2:5])
    high = sum(bands[5:])
    descriptors = []
    if high > 0.45:
        descriptors.append("bright")
    if low > 0.45:
        descriptors.append("dark")
    if duration_s < 0.4:
        descriptors.append("tight")
    if low > 0.55 and duration_s < 1.5:
        return "Drums", "kick", 0.62, descriptors or ["low-energy"]
    if high > 0.5 and duration_s < 0.8:
        return "Drums", "hat", 0.58, descriptors or ["high-energy"]
    if duration_s >= 2.0 and mid > 0.35:
        return "Synth", "pad", 0.52, descriptors or ["sustained"]
    return "unsorted", "unknown", 0.28, descriptors or ["no-hint"]


def _sample_type(duration_s: float, onsets: list[float], bpm: Optional[float]) -> SampleType:
    if duration_s < 1.0 and len(onsets) <= 2:
        return "oneshot"
    if bpm is not None or (duration_s >= 1.5 and len(onsets) >= 4):
        return "loop"
    if duration_s < 2.0:
        return "oneshot"
    return "unknown"


def _embedding(samples: np.ndarray, sr: int, bands: list[float]) -> list[float]:
    n = min(len(samples), 4096)
    chunk = samples[:n]
    if n < 4096:
        chunk = np.pad(chunk, (0, 4096 - n))
    spec = np.abs(np.fft.rfft(chunk * np.hanning(4096)))
    spec = spec[:500]
    spec = spec / (np.linalg.norm(spec) + 1e-9)
    vec = np.zeros(512, dtype=np.float32)
    vec[:500] = spec.astype(np.float32)
    vec[500:508] = np.array(bands, dtype=np.float32)
    vec[508] = min(len(samples) / sr / 8.0, 1.0)
    vec[509] = float(np.sqrt(np.mean(chunk**2)))
    vec[510] = float(np.max(np.abs(chunk)))
    vec[511] = float(np.mean(np.diff(chunk) ** 2))
    return [round(float(x), 5) for x in vec]


def analyze(path: str) -> SampleRecord:
    filename = os.path.basename(path)
    abs_path = os.path.abspath(path)
    if not os.path.isfile(path):
        return _empty(path, filename, "file not found")
    try:
        samples, sr, channels = _read_wav(path)
    except Exception as exc:
        return _empty(path, filename, f"could not decode: {exc}")

    duration_s = round(len(samples) / float(sr) if sr else 0.0, 3)
    peak = _peak_db(samples)
    bands = _band_energy(samples, sr)
    onsets = _onsets(samples, sr)
    bpm = _bpm_from_onsets(onsets, duration_s)
    family, instrument, confidence, descriptors = _classify(filename, bands, duration_s)
    if confidence < 0.5:
        family, instrument = "unsorted", "unknown"
    sample_type = _sample_type(duration_s, onsets, bpm)
    key = None
    if family in ("Bass", "Acoustic", "Synth", "Vocal") and sample_type != "oneshot":
        key = _key_from_spectrum(samples, sr)
    if sample_type != "loop":
        bpm = None

    return SampleRecord(
        path=abs_path,
        filename=filename,
        duration_s=duration_s,
        sample_rate=sr,
        channels=channels,
        peak_db=peak,
        sample_type=sample_type,
        bpm=bpm,
        key=key,
        family=family,
        instrument=instrument,
        confidence=round(confidence, 3),
        band_energy=bands,
        onsets=onsets,
        embedding=_embedding(samples, sr, bands),
        descriptors=descriptors,
        error=None,
    )


assert set(FAMILIES)  # keep contracts import live for reviewers
assert "unknown" in INSTRUMENTS["unsorted"]
