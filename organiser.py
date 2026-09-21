"""
organiser.py — C in contracts.py. The most dangerous file in the project:
it moves people's files around on disk.

Public contract (see contracts.py):
    scan(root)                   -> list[str]
    plan(records, template)      -> list[RenamePlan]
    apply(plans, dry_run=True)   -> str   (path to the undo log)
    undo(log_path)               -> None

Non-negotiable rules this file obeys everywhere:
    - Nothing is ever deleted. No os.remove / os.unlink / os.rmdir /
      shutil.rmtree / truncate, anywhere, for any reason. If something needs
      to go away, it gets *moved* to _quarantine/ instead — never removed.
    - Every move is written to the undo log BEFORE it happens, or it does
      not happen at all.
    - apply() defaults to dry_run=True and a dry run changes nothing on disk.
    - A destination that already exists aborts the *whole* batch before a
      single file is touched. We never overwrite.

The CLI at the bottom uses analyzer.analyze() for real records when
analyzer.py is importable, and falls back to fixtures/records.json (or a
placeholder) if it isn't — this module must keep working either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from dataclasses import replace
from datetime import datetime

from contracts import RenamePlan, SampleRecord

# Defensive import: analyzer.py is A's module, not ours. If it isn't present
# or fails to import, the CLI below falls back to fixtures/records.json
# instead of taking the whole tool down.
try:
    import analyzer as _analyzer
except Exception:
    _analyzer = None

# --------------------------------------------------------------------------
# scan()
# --------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".wav", ".aiff", ".aif", ".flac", ".ogg", ".mp3"}
SKIP_DIR_NAMES = {"__macosx", "_quarantine", "_undo"}


def scan(root: str) -> list[str]:
    """Recursively find audio files under root.

    Skips hidden files/dirs, __MACOSX, .DS_Store, and anything under
    _quarantine or _undo (our own bookkeeping folders).
    """
    root = os.path.abspath(root)
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and d.lower() not in SKIP_DIR_NAMES
        )
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                found.append(os.path.join(dirpath, name))
    return found


# --------------------------------------------------------------------------
# slugify / naming helpers
# --------------------------------------------------------------------------

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DASH_RUN_RE = re.compile(r"-+")
_SEPARATOR_RUN_RE = re.compile(r"[-_]{2,}")

STEM_MAX_LEN = 24
PATH_MAX_LEN = 200


def _slugify(text: str | None) -> str:
    """lowercase, ASCII only, punctuation/spaces -> '-', collapse repeats."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = _NON_ALNUM_RE.sub("-", text)
    text = _DASH_RUN_RE.sub("-", text)
    return text.strip("-")


def _stem(filename: str, max_len: int = STEM_MAX_LEN) -> str:
    """Slugified original filename (no ext), truncated. Never empty."""
    base = os.path.splitext(filename)[0]
    slug = _slugify(base)
    slug = slug[:max_len].rstrip("-")
    return slug or "file"


def _tidy_separators(rel_path: str) -> str:
    """Drop the separators an empty template field leaves behind, so an omitted
    descriptor or key/BPM never shows up as '__' or a leading '_'."""
    segments = []
    for segment in rel_path.split("/"):
        segment = _SEPARATOR_RUN_RE.sub(lambda m: "_" if "_" in m.group(0) else "-", segment)
        segments.append(segment.strip("-_"))
    return "/".join(segments)


def _cap_length(rel_path: str, limit: int = PATH_MAX_LEN) -> str:
    """Never let a generated relative path exceed `limit` chars."""
    if len(rel_path) <= limit:
        return rel_path
    if "/" in rel_path:
        dirpart, filename = rel_path.rsplit("/", 1)
    else:
        dirpart, filename = "", rel_path
    name, ext = os.path.splitext(filename)
    overflow = len(rel_path) - limit
    name = name[: max(1, len(name) - overflow)]
    rebuilt = f"{dirpart}/{name}{ext}" if dirpart else f"{name}{ext}"
    if len(rebuilt) > limit:
        # Pathological case (e.g. a very deep custom template): hard-truncate
        # from the left but keep the extension so it stays a valid filename.
        keep = max(1, limit - len(ext))
        rebuilt = rebuilt[:keep] + ext
    return rebuilt


def _resolve_collisions(rel_paths: list[str]) -> list[str]:
    """Append -2, -3, ... to any relative path that repeats, checked across
    the whole batch (not just against its immediate neighbour)."""
    used: set[str] = set()
    resolved: list[str] = []
    for rel in rel_paths:
        candidate = rel
        if candidate in used:
            base, ext = os.path.splitext(rel)
            n = 2
            while True:
                candidate = _cap_length(f"{base}-{n}{ext}")
                if candidate not in used:
                    break
                n += 1
        used.add(candidate)
        resolved.append(candidate)
    return resolved


# --------------------------------------------------------------------------
# plan()
# --------------------------------------------------------------------------

DEFAULT_TEMPLATE = "{family}/{instrument}/{descriptor}{keyOrBpm}{type}_{stem}{ext}"

_TYPE_HUMAN = {"oneshot": "one-shot", "loop": "loop", "unknown": "unknown-type"}


def _descriptor_component(rec: SampleRecord) -> str:
    if rec.descriptors:
        slug = _slugify(rec.descriptors[0])
        if slug:
            return slug + "-"
    return ""


def _key_or_bpm_component(rec: SampleRecord) -> str:
    # BPM for loops, key for tonal (non-loop) material, omitted for neither.
    if rec.sample_type == "loop":
        if rec.bpm is not None:
            return f"{round(rec.bpm)}bpm-"
        return ""
    if rec.key:
        slug = _slugify(rec.key)
        if slug:
            return slug + "-"
    return ""


def _key_or_bpm_human(rec: SampleRecord) -> str | None:
    if rec.sample_type == "loop" and rec.bpm is not None:
        return f"{round(rec.bpm)}bpm"
    if rec.sample_type != "loop" and rec.key:
        return rec.key
    return None


def _reason(rec: SampleRecord) -> str:
    parts = [rec.instrument or "unknown", _TYPE_HUMAN.get(rec.sample_type, rec.sample_type or "unknown-type")]
    extra = _key_or_bpm_human(rec)
    if extra:
        parts.append(extra)
    return f"{', '.join(parts)}, was {rec.filename}"


def _unsorted_reason(rec: SampleRecord) -> str:
    if rec.error:
        return f"error: {rec.error}, was {rec.filename}"
    return f"unsorted (confidence {rec.confidence:.2f}), was {rec.filename}"


def _render_template(template: str, rec: SampleRecord, stem: str, ext: str) -> str:
    subs = {
        "family": _slugify(rec.family) or "unsorted",
        "instrument": _slugify(rec.instrument) or "unknown",
        "descriptor": _descriptor_component(rec),
        "keyOrBpm": _key_or_bpm_component(rec),
        "type": _slugify(rec.sample_type) or "unknown",
        "stem": stem,
        "ext": ext,
    }
    try:
        rel = template.format(**subs)
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"bad rename template {template!r}: unknown field {exc}; "
            f"supported fields are {', '.join('{' + k + '}' for k in subs)}"
        ) from exc
    return _tidy_separators(rel.replace("\\", "/"))


def _common_dir(paths: list[str]) -> str:
    dirs = [os.path.dirname(p) or "." for p in paths]
    try:
        return os.path.commonpath(dirs)
    except ValueError:
        return "."


def plan(records: list[SampleRecord], template: str = DEFAULT_TEMPLATE) -> list[RenamePlan]:
    """Turn analysed records into a list of proposed moves. Nothing on disk
    changes here — this only computes where things *should* go."""
    records = list(records)
    if not records:
        return []

    root = _common_dir([r.path for r in records])

    rels: list[str] = []
    reasons: list[str] = []
    for rec in records:
        name = rec.filename or os.path.basename(rec.path)
        ext = os.path.splitext(name)[1].lower()
        stem = _stem(name)

        if rec.error or rec.family == "unsorted":
            # Never guess a folder for broken or unclassified material.
            rel = f"_unsorted/{stem}{ext}"
            reason = _unsorted_reason(rec)
        else:
            rel = _render_template(template, rec, stem, ext)
            reason = _reason(rec)

        rels.append(_cap_length(rel))
        reasons.append(reason)

    resolved = _resolve_collisions(rels)

    plans = []
    for rec, rel, reason in zip(records, resolved, reasons):
        new_path = os.path.normpath(os.path.join(root, *rel.split("/")))
        plans.append(RenamePlan(old_path=rec.path, new_path=new_path, reason=reason))
    return plans


# --------------------------------------------------------------------------
# apply() / undo()
# --------------------------------------------------------------------------

class OrganiserError(Exception):
    """Raised when a batch cannot be safely applied. Nothing is moved when
    this is raised — apply() validates the entire batch before touching
    the first file."""


def _validate_batch(plans: list[RenamePlan]) -> None:
    seen_targets: dict[str, str] = {}
    for p in plans:
        if p.old_path == p.new_path:
            continue
        if p.new_path in seen_targets:
            raise OrganiserError(
                f"plan collision: both {seen_targets[p.new_path]!r} and "
                f"{p.old_path!r} target {p.new_path!r}"
            )
        seen_targets[p.new_path] = p.old_path

    for p in plans:
        if p.old_path == p.new_path:
            continue
        if not os.path.exists(p.old_path):
            raise OrganiserError(f"source is missing, aborting whole batch: {p.old_path!r}")
        if os.path.exists(p.new_path):
            raise OrganiserError(
                f"destination already exists, aborting whole batch (nothing moved): {p.new_path!r}"
            )


def _dirs_to_create(path: str) -> list[str]:
    """The ancestors of `path` (shallowest first) that do not exist yet, i.e.
    exactly the directories a makedirs(path) call would bring into being."""
    missing: list[str] = []
    current = os.path.abspath(path)
    while current and not os.path.isdir(current):
        missing.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return list(reversed(missing))


def apply(plans: list[RenamePlan], dry_run: bool = True) -> str:
    """Perform (or simulate) the moves in `plans`.

    dry_run=True (the default) validates the batch but changes nothing on
    disk and returns "".

    On a real run, the whole batch is validated up front; if anything is
    wrong (missing source, existing destination, internal collision) an
    OrganiserError is raised and NOT ONE file is moved. Only once validation
    passes do we write the undo log (one line per move, written before that
    move happens) and perform the moves with shutil.move.

    Every directory this call creates is recorded in the log too, so undo()
    can leave the library exactly as it found it.
    """
    plans = list(plans)
    if not plans:
        return ""

    _validate_batch(plans)

    if dry_run:
        return ""

    root = _common_dir([p.old_path for p in plans] + [p.new_path for p in plans])
    undo_dir = os.path.join(root, "_undo")
    os.makedirs(undo_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    log_path = os.path.join(undo_dir, f"undo-{timestamp}.json")

    with open(log_path, "w") as log_f:
        for p in plans:
            if p.old_path == p.new_path:
                continue
            log_f.write(json.dumps({"from": p.old_path, "to": p.new_path}) + "\n")
            log_f.flush()
            os.fsync(log_f.fileno())

            parent = os.path.dirname(p.new_path)
            if parent:
                for created in _dirs_to_create(parent):
                    log_f.write(json.dumps({"mkdir": created}) + "\n")
                    log_f.flush()
                    os.fsync(log_f.fileno())
                os.makedirs(parent, exist_ok=True)
            shutil.move(p.old_path, p.new_path)

    return log_path


def undo(log_path: str) -> None:
    """Replay an undo log in reverse, restoring every moved file to where
    it came from. Never overwrites, never deletes a file; tolerant of a
    partially completed forward run so it stays useful even after a crash.

    The empty directories apply() created are removed afterwards (deepest
    first, os.rmdir only, only when empty and only when this log says apply()
    created them) so the library is left exactly as apply() found it."""
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    created_dirs: list[str] = []
    for entry in reversed(entries):
        if "mkdir" in entry:
            created_dirs.append(entry["mkdir"])
            continue
        src, dst = entry["to"], entry["from"]
        if not os.path.exists(src):
            print(f"organiser: undo skip (missing source): {src}", file=sys.stderr)
            continue
        if os.path.exists(dst):
            print(f"organiser: undo skip (would overwrite): {dst}", file=sys.stderr)
            continue
        parent = os.path.dirname(dst)
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.move(src, dst)

    # Deepest first, so a child is gone before its parent is considered.
    for directory in sorted(set(created_dirs), key=lambda d: d.count(os.sep), reverse=True):
        if not os.path.isdir(directory):
            continue
        if os.listdir(directory):
            print(f"organiser: undo keeping non-empty directory: {directory}", file=sys.stderr)
            continue
        try:
            os.rmdir(directory)
        except OSError as exc:
            print(f"organiser: undo could not remove directory {directory}: {exc}", file=sys.stderr)


# --------------------------------------------------------------------------
# CLI — develops against fixtures/records.json since analyzer.py isn't ready
# --------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIXTURES_PATH = os.path.join(_SCRIPT_DIR, "fixtures", "records.json")


def _placeholder_record() -> SampleRecord:
    """Used only if fixtures/records.json is unavailable — routes everything
    to _unsorted/ rather than guessing."""
    return SampleRecord(
        path="", filename="", duration_s=0.0, sample_rate=0, channels=0,
        peak_db=0.0, sample_type="unknown", bpm=None, key=None,
        family="unsorted", instrument="unknown", confidence=0.0,
        error="no analyzer output available (fixtures/records.json missing)",
    )


def _load_demo_records(paths: list[str], fixtures_path: str) -> list[SampleRecord]:
    """Analyze real files with analyzer.analyze() when it's importable.
    Falls back to fixture analysis data (round-robin) if analyzer.py isn't
    available, so plan()/apply()/undo() can still be exercised."""
    if _analyzer is not None:
        records = []
        for p in paths:
            try:
                records.append(_analyzer.analyze(p))
            except Exception as exc:
                # analyzer.analyze() is contracted to never raise; if it
                # does anyway, degrade to _unsorted rather than crash here.
                records.append(replace(
                    _placeholder_record(), path=p, filename=os.path.basename(p),
                    error=f"analyzer.analyze() raised: {exc}",
                ))
        return records

    fixtures: list[SampleRecord] = []
    if fixtures_path and os.path.exists(fixtures_path):
        with open(fixtures_path) as f:
            raw = json.load(f)
        fixtures = [SampleRecord(**d) for d in raw]
    if not fixtures:
        fixtures = [_placeholder_record()]

    records = []
    for i, p in enumerate(paths):
        base = fixtures[i % len(fixtures)]
        records.append(replace(base, path=p, filename=os.path.basename(p)))
    return records


def _format_plan_diff(plans: list[RenamePlan], root: str) -> str:
    lines = []
    for p in plans:
        old_disp = os.path.relpath(p.old_path, root)
        new_disp = os.path.relpath(p.new_path, root)
        lines.append(f"--- {old_disp}")
        lines.append(f"+++ {new_disp}")
        lines.append(f"    {p.reason}")
    return "\n".join(lines)


def _snapshot(root: str) -> dict[str, str]:
    """sha256 of every file under root, excluding our own _undo bookkeeping,
    keyed by path relative to root. Used by --selftest."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "_undo"]
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            with open(full, "rb") as f:
                out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out


def _selftest(root: str, template: str, fixtures_path: str) -> bool:
    """Copy root, apply a full plan, undo it, and check the copy is
    byte-identical to how it started (ignoring our own _undo folder)."""
    root = os.path.abspath(root)
    with tempfile.TemporaryDirectory(prefix="organiser-selftest-") as tmp:
        copy_root = os.path.join(tmp, "copy")
        shutil.copytree(root, copy_root)

        before = _snapshot(copy_root)

        paths = scan(copy_root)
        records = _load_demo_records(paths, fixtures_path)
        plans = plan(records, template)
        log_path = apply(plans, dry_run=False)
        if log_path:
            undo(log_path)

        after = _snapshot(copy_root)

    if before.keys() != after.keys():
        missing = sorted(set(before) - set(after))
        extra = sorted(set(after) - set(before))
        print(f"organiser: selftest FAILED — missing={missing} extra={extra}", file=sys.stderr)
        return False

    for rel, digest in before.items():
        if after[rel] != digest:
            print(f"organiser: selftest FAILED — content changed: {rel}", file=sys.stderr)
            return False
    return True


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="organiser.py",
        description="Reorganise a messy sample library. Dry-run by default; "
                     "nothing moves until you pass --apply.",
    )
    parser.add_argument("root", help="Folder to scan and (optionally) reorganise.")
    parser.add_argument("--apply", action="store_true",
                         help="Actually perform the moves. Without this flag, nothing on disk changes.")
    parser.add_argument("--dry-run", action="store_true",
                         help="No-op: dry-run is already the default unless --apply is given.")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="Rename template.")
    parser.add_argument("--fixtures", default=DEFAULT_FIXTURES_PATH,
                         help="records.json to stand in for analyzer.py output.")
    parser.add_argument("--undo", metavar="LOG_PATH", help="Undo a previous apply using this log file.")
    parser.add_argument("--selftest", action="store_true",
                         help="Apply+undo on a temp copy of root and verify it's byte-identical afterwards.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.undo:
        undo(args.undo)
        print(f"organiser: undo complete ({args.undo})")
        return 0

    if not os.path.isdir(args.root):
        print(f"organiser: not a directory: {args.root!r}", file=sys.stderr)
        return 1

    if args.selftest:
        ok = _selftest(args.root, args.template, args.fixtures)
        print("organiser: selftest PASSED" if ok else "organiser: selftest FAILED")
        return 0 if ok else 1

    paths = scan(args.root)
    print(f"organiser: found {len(paths)} audio file(s) under {args.root!r}")
    if not paths:
        return 0

    records = _load_demo_records(paths, args.fixtures)
    plans = plan(records, args.template)

    root_abs = os.path.abspath(args.root)
    print(_format_plan_diff(plans, root_abs))
    print(f"\norganiser: {len(plans)} planned move(s), template = {args.template!r}")

    if not args.apply:
        try:
            apply(plans, dry_run=True)
        except OrganiserError as exc:
            print(f"organiser: this plan could NOT be applied as-is: {exc}", file=sys.stderr)
            return 1
        print("organiser: dry run only — nothing was moved. Re-run with --apply to perform it.")
        return 0

    try:
        log_path = apply(plans, dry_run=False)
    except OrganiserError as exc:
        print(f"organiser: aborted, nothing was moved: {exc}", file=sys.stderr)
        return 1

    print(f"organiser: moved {len(plans)} file(s). undo log: {log_path}")
    print(f"organiser: to undo, run: python organiser.py {args.root} --undo {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
