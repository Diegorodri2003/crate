# crate

A little local tool for cleaning up a messy sample library. Point it at a folder full of badly-named WAVs dumped across a dozen subfolders, and it listens to every file, figures out what it is (kick, 808, vocal chop, whatever), and proposes a clean `family/instrument/...` layout for it — nothing moves until you say so, and every move can be undone.

It's four small Python files with one shared contract between them, plus a single HTML page for the UI. No build step, no framework, no database.

<p align="center">
  <img src="static/readme-shots/01-hero.png" width="380" alt="Crate main screen">
</p>

## What each piece does

- **`analyzer.py`** — opens one audio file and turns it into a record: duration, peak level, BPM, key, loop-or-one-shot, a rough family/instrument guess, plus a few descriptor words like "dark" or "punchy". Real DSP via `librosa`, not guesswork — except the classification itself, which is filename-keyword based unless you've also installed `transformers`/`torch` for the CLAP model.
- **`index.py`** — an in-memory search index over those records. Fuzzy text search, family/type/BPM/key filters, and a "fit to this reference track" mode that ranks the whole library against a bounce you drop in.
- **`organiser.py`** — the part that's actually allowed to touch your files. It scans a folder, proposes a rename plan, and applies it. It never deletes anything, ever — a destination collision aborts the whole batch before a single file moves, and every apply writes an undo log first.
- **`api.py`** — stdlib-only HTTP server (`http.server`, no Flask) that wires the three above to the page below.
- **`static/index.html`** — the UI. One file, vanilla JS, no build tool.

## Running it

```
python -m venv .venv
.venv\Scripts\activate          # or source .venv/bin/activate on mac/linux
pip install -r requirements.txt
python api.py
```

Then open `http://127.0.0.1:8000/`. `api.py` itself needs nothing beyond the standard library — if you skip the `pip install`, the server still starts, it just serves demo fixtures instead of analysing real audio, and says so loudly in both the terminal and a banner at the top of the page.

## The interface

### Drop samples, get them filed away

The circle in the middle is the whole point of the app in miniature: drop one file, a batch of files, or an entire folder on it, and every audio file in there gets analysed and moved straight into `family/instrument/...` under your library folder — for real, on disk, one `organiser.apply()` per file with its own undo log. It uses whatever model is actually loaded, no shortcuts: without CLAP that's the filename fallback, which is honest about being unsure a lot of the time and sends most things to `_unsorted/` rather than guess. Install CLAP and it gets a lot better at this without any other change.

<p align="center">
  <img src="static/readme-shots/09-place-batch.png" width="380" alt="Dropping three samples at once — the circle reports how many were filed and how many landed in unsorted">
</p>

### Search your library

Type what you're after — a word, an instrument, a vibe — and it ranks the library against it. Family, type (one-shot/loop), key and BPM filters sit right underneath if you want to narrow it down further; key is a real dropdown, not a button you have to click through one option at a time. Clicking a result plays it through the bar at the bottom.

<p align="center">
  <img src="static/readme-shots/02-search.png" width="380" alt="Searching for kick returns ranked matches">
  <img src="static/readme-shots/08-key-dropdown.png" width="380" alt="The key filter is a plain dropdown">
</p>

### Fit a reference track

The **FIT TRACK** button in the left rail takes a bounce of whatever you're working on and re-ranks the whole library against it instead of a text query — useful for "what in here actually fits this song." **Contrast** flips it around and surfaces what stands *apart* from the track, for when you want something that cuts through instead of blending in.

<p align="center">
  <img src="static/readme-shots/05-fit.png" width="380" alt="Fit Track mode ranking the library against an uploaded bounce">
</p>

### Rescan and reorganise, safely

**RESCAN FOLDERS** walks your library folder and re-analyses everything in it, then builds a full reorganise plan automatically. Nothing happens to your files yet — **View diff** shows exactly what would move and why before you touch **Apply**, and **Undo** is one click for as long as the session lasts.

The folder button at the top right (showing `messy`) opens a real native folder picker — the browser itself can't hand a website an absolute filesystem path, so clicking it asks the Python server to pop the OS's own folder dialog and hand back whatever you picked there. Point it anywhere on disk and Rescan/Place both work on that folder from then on. (If the machine has no desktop environment for that dialog to open on, it quietly falls back to typing a path.)

First run, with no library yet? `messy/` is gitignored on purpose — nobody wants a stranger's demo audio in their git history — so `api.py` notices it's missing on startup and generates a small synthetic one automatically, just so Scan/Search/Place have something to chew on immediately.

<p align="center">
  <img src="static/readme-shots/07-plan.png" width="380" alt="The reorganise plan drawer showing proposed moves before applying">
</p>

## Why the labels are sometimes hardcoded

Real classification needs [CLAP](https://github.com/LAION-AI/CLAP) (`transformers` + `torch`), which is a heavy, slow install for a demo. Without it, `analyzer.py` falls back to matching keywords in the filename — which works, but only when the filename is honest about what it is. For the demo library shipped in `fixtures/records.json`, I ran that same keyword match separately over ~5,400 real samples, kept only the ~1,700 where the name was unambiguous (one match, no conflicts), and threw out anything that failed to decode. That's what `USE_FIXTURES` in `api.py` serves — real audio, filed correctly, without waiting on a multi-gigabyte model download. Flip it to `False` once CLAP is installed and it analyses live instead.

## Project layout

```
analyzer.py     A — one file in, one SampleRecord out
index.py        B — search over records
organiser.py    C — scan / plan / apply / undo (the dangerous one)
api.py          D — HTTP + the page
contracts.py    the shared shapes all four import
static/         the UI
make_fixtures.py   generates fixtures/records.json + a demo messy/ folder
```

`contracts.py` is the only file all four depend on — the `SampleRecord` shape, the family/instrument list, the search query. If you're changing what a record looks like, that's the one file everything else has to agree with.
