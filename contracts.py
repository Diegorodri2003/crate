"""
contracts.py — THE ONLY SHARED FILE. Everyone imports from here.

RULE: nobody edits this without saying it out loud to the other three.
If you need a field that isn't here, say so in the channel and add it
together. Do NOT quietly add one — the other three are coding against
this exact shape right now.

Every module is a pure function over these types:
    A  analyzer.analyze(path)         -> SampleRecord
    B  index.upsert(records)          -> None
       index.search(query)            -> list[SearchHit]
    C  organiser.scan(root)           -> list[str]
       organiser.plan(records, tmpl)  -> list[RenamePlan]
       organiser.apply(plans, dry_run=True)
                                      -> str   (path of undo log; "" on a
                                                dry run, which moves nothing)
       organiser.undo(log_path)       -> None
    D  api.py wires all of the above to HTTP + one HTML page.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal

SampleType = Literal["oneshot", "loop", "unknown"]

FAMILIES = ["Drums", "Bass", "Acoustic", "Synth", "Vocal", "FX", "unsorted"]

INSTRUMENTS = {
    "Drums":    ["kick", "snare", "clap", "hat", "ride", "tom", "perc"],
    "Bass":     ["808", "sub", "reese", "plucked-bass"],
    "Acoustic": ["violin", "guitar", "piano", "strings", "brass", "woodwind"],
    "Synth":    ["pad", "lead", "pluck", "arp", "stab"],
    "Vocal":    ["vocal-oneshot", "vocal-phrase", "acapella", "chop"],
    "FX":       ["riser", "impact", "downlifter", "foley", "ambience"],
    "unsorted": ["unknown"],
}


@dataclass
class SampleRecord:
    """One analysed audio file. Produced by A, consumed by B, C and D."""
    path: str                       # absolute path, the primary key
    filename: str                   # basename, original
    duration_s: float
    sample_rate: int
    channels: int
    peak_db: float                  # <= 0.0

    sample_type: SampleType         # "oneshot" | "loop" | "unknown"
    bpm: Optional[float]            # None when not applicable / not found
    key: Optional[str]              # "F#m", "C", or None for atonal

    family: str                     # one of FAMILIES
    instrument: str                 # one of INSTRUMENTS[family]
    confidence: float               # 0.0 .. 1.0 — below 0.5 => family "unsorted"

    band_energy: list[float] = field(default_factory=list)   # exactly 8, sums to 1.0
    onsets: list[float] = field(default_factory=list)        # seconds from start
    embedding: Optional[list[float]] = None                  # 512 floats, or None
    descriptors: list[str] = field(default_factory=list)     # ["dark", "saturated"]
    error: Optional[str] = None     # set this instead of raising


@dataclass
class RenamePlan:
    """One proposed file move. Produced by C, rendered by D. Nothing moves
    until apply() is called explicitly."""
    old_path: str
    new_path: str
    reason: str                     # human-readable, shown in the UI


@dataclass
class SearchQuery:
    """Everything the UI can ask for. All fields optional — an empty query
    means 'everything', capped at limit."""
    text: Optional[str] = None           # semantic / filename search
    family: Optional[str] = None
    instrument: Optional[str] = None
    sample_type: Optional[SampleType] = None
    bpm_min: Optional[float] = None
    bpm_max: Optional[float] = None
    key: Optional[str] = None
    fit_to: Optional[str] = None         # path to a reference/bounce to fit against
    contrast: bool = False               # invert the character signal
    limit: int = 50


@dataclass
class SearchHit:
    record: SampleRecord
    score: float                    # 0.0 .. 1.0, higher is better
    reasons: list[str]              # ["126 -> 128 (+1.6%)", "fills 4-8kHz"]


# 8 fixed frequency bands, Hz. A fills band_energy against exactly these.
# B uses them for the spectral-gap score. Do not change the count.
BANDS_HZ = [
    (20, 60), (60, 120), (120, 250), (250, 500),
    (500, 1000), (1000, 2000), (2000, 5000), (5000, 20000),
]
