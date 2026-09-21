"""
make_fixtures.py — run this FIRST, before anyone writes a line of real code.

Two jobs:
  1. fixtures/records.json — 24 fake SampleRecords so B, C and D can build
     against realistically-shaped data while A is still writing the analyser.
  2. messy/ — a deliberately chaotic sample folder to demo on, built from
     real audio if you point it at some, or from synthesised tones if not.

    python make_fixtures.py                    # fixtures + synthetic messy/
    python make_fixtures.py ~/Samples/clean    # fixtures + messy/ from real files
"""

import json, os, random, shutil, sys, math, wave, struct
from dataclasses import asdict
from contracts import SampleRecord, FAMILIES, INSTRUMENTS, BANDS_HZ

random.seed(7)  # deterministic: everyone gets the same fixtures

UGLY = [
    "VIOLIN_SOUND_{n:02d}.wav", "v_{n}.wav", "SDJKGH{n}983.wav",
    "SAMPLE_LIBRARIES_{n:02d}.wav", "freesound_org_download_7sax{n}85.wav",
    "Untitled-{n}.wav", "recording {n}.wav", "KICK_FINAL_v{n}_USE_THIS.wav",
    "{n}.wav", "Audio Track-{n:03d}.wav", "bounce_{n}_new_new.wav",
    "WA_TropicalHouse_{n}.wav", "__MACOSX_{n}.wav", "Copy of Copy of {n}.wav",
]

DESCRIPTORS = ["dark", "bright", "saturated", "clean", "vintage",
               "metallic", "warm", "punchy", "washed", "tight"]
KEYS = ["C", "Cm", "D", "Dm", "E", "Em", "F", "F#m", "G", "Gm", "A", "Am", "Bm", None]


def fake_record(i: int) -> SampleRecord:
    family = random.choice([f for f in FAMILIES if f != "unsorted"])
    instrument = random.choice(INSTRUMENTS[family])
    is_loop = random.random() < 0.4
    bpm = round(random.choice([90, 120, 124, 126, 128, 140, 174]) + random.random(), 1) if is_loop else None
    bands = [random.random() for _ in BANDS_HZ]
    total = sum(bands)
    bands = [round(b / total, 4) for b in bands]
    dur = round(random.uniform(1.5, 8.0) if is_loop else random.uniform(0.15, 2.0), 3)
    n_on = random.randint(3, 16) if is_loop else 1
    return SampleRecord(
        path=f"/demo/messy/{UGLY[i % len(UGLY)].format(n=i)}",
        filename=UGLY[i % len(UGLY)].format(n=i),
        duration_s=dur,
        sample_rate=44100,
        channels=random.choice([1, 2]),
        peak_db=round(random.uniform(-12.0, -0.3), 2),
        sample_type="loop" if is_loop else "oneshot",
        bpm=bpm,
        key=random.choice(KEYS) if family in ("Bass", "Acoustic", "Synth", "Vocal") else None,
        family=family,
        instrument=instrument,
        confidence=round(random.uniform(0.45, 0.99), 3),
        band_energy=bands,
        onsets=sorted(round(random.uniform(0, dur), 3) for _ in range(n_on)),
        embedding=[round(random.gauss(0, 0.1), 5) for _ in range(512)],
        descriptors=random.sample(DESCRIPTORS, k=random.randint(1, 3)),
        error=None,
    )


def write_fixtures():
    os.makedirs("fixtures", exist_ok=True)
    records = [fake_record(i) for i in range(24)]
    # one deliberately broken record — everyone must handle it without crashing
    bad = fake_record(99)
    bad.error = "could not decode: unsupported codec"
    bad.family, bad.instrument, bad.confidence = "unsorted", "unknown", 0.0
    bad.embedding, bad.bpm, bad.key = None, None, None
    records.append(bad)
    with open("fixtures/records.json", "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
    print(f"fixtures/records.json — {len(records)} records (1 with error set)")


def tone_wav(path, freq, dur, sr=44100, noise=0.0):
    n = int(sr * dur)
    frames = bytearray()
    for i in range(n):
        env = math.exp(-3.0 * i / n)
        v = math.sin(2 * math.pi * freq * i / sr) * env
        if noise:
            v += random.uniform(-noise, noise) * env
        frames += struct.pack("<h", int(max(-1, min(1, v)) * 30000))
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(bytes(frames))


def make_messy(source=None):
    out = "messy"
    shutil.rmtree(out, ignore_errors=True)
    subdirs = ["", "New Folder", "New Folder/untitled", "packs/VOL 2",
               "from dani", "DRUMS!!!!", "downloads"]
    for s in subdirs:
        os.makedirs(os.path.join(out, s), exist_ok=True)

    if source and os.path.isdir(source):
        files = [os.path.join(dp, f) for dp, _, fs in os.walk(source)
                 for f in fs if f.lower().endswith((".wav", ".aiff", ".aif", ".flac", ".mp3"))]
        random.shuffle(files)
        for i, src in enumerate(files[:120]):
            ext = os.path.splitext(src)[1]
            name = UGLY[i % len(UGLY)].format(n=i).replace(".wav", ext)
            shutil.copy2(src, os.path.join(out, random.choice(subdirs), name))
        print(f"messy/ — {min(len(files),120)} real files with ruined names")
    else:
        specs = [(60, .3, .0), (220, .25, .35), (1800, .12, .6), (80, .8, .0),
                 (440, 1.2, .05), (3000, .08, .8), (110, 2.0, .02), (660, .5, .1)]
        for i in range(60):
            f, d, nz = specs[i % len(specs)]
            name = UGLY[i % len(UGLY)].format(n=i)
            tone_wav(os.path.join(out, random.choice(subdirs), name),
                     f * random.uniform(.85, 1.2), d * random.uniform(.7, 1.4), noise=nz)
        print("messy/ — 60 synthetic files with ruined names")
    print("       (nested folders, duplicate-ish names, no extensions of meaning)")


if __name__ == "__main__":
    write_fixtures()
    make_messy(sys.argv[1] if len(sys.argv) > 1 else None)
    print("\nready. everyone: `git pull` and start.")
