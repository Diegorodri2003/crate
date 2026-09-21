"""C — scan a folder, plan renames, apply with an undo log."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Iterable

from contracts import RenamePlan, SampleRecord

AUDIO_EXT = {".wav", ".aiff", ".aif", ".flac", ".mp3", ".ogg"}
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def scan(root: str) -> list[str]:
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return []
    found: list[str] = []
    for dp, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in AUDIO_EXT:
                found.append(os.path.join(dp, name))
    found.sort()
    return found


def _slug(part: str) -> str:
    part = part.strip().replace(" ", "-")
    part = _UNSAFE.sub("-", part)
    return part.strip("-._") or "unnamed"


def _render(tmpl: str, rec: SampleRecord) -> str:
    stem, ext = os.path.splitext(rec.filename)
    values = {
        "family": rec.family,
        "instrument": rec.instrument,
        "filename": rec.filename,
        "stem": stem,
        "ext": ext.lstrip(".") or "wav",
        "sample_type": rec.sample_type,
        "bpm": "" if rec.bpm is None else f"{rec.bpm:g}",
        "key": rec.key or "atk",
        "confidence": f"{rec.confidence:.2f}",
    }
    out = tmpl
    for key, val in values.items():
        out = out.replace("{" + key + "}", str(val))
    return out


def plan(records: Iterable[SampleRecord], tmpl: str) -> list[RenamePlan]:
    tmpl = tmpl or "{family}/{instrument}/{stem}.{ext}"
    used: set[str] = set()
    plans: list[RenamePlan] = []
    for rec in records:
        if rec.error:
            rel = os.path.join("unsorted", "unknown", rec.filename)
            reason = f"left unsorted ({rec.error})"
        else:
            rel = _render(tmpl, rec)
            reason = f"{rec.family}/{rec.instrument} ({rec.confidence:.2f})"
        parts = [_slug(p) if p else "unnamed" for p in rel.replace("\\", "/").split("/")]
        if parts:
            stem, ext = os.path.splitext(parts[-1])
            parts[-1] = _slug(stem) + (ext.lower() if ext else ".wav")
        dest_name = "/".join(parts)
        n = 2
        candidate = dest_name
        while candidate.lower() in used:
            stem, ext = os.path.splitext(dest_name)
            candidate = f"{stem}-{n}{ext}"
            n += 1
        used.add(candidate.lower())
        plans.append(
            RenamePlan(
                old_path=rec.path,
                new_path=candidate,
                reason=reason,
            )
        )
    return plans


def _dest_root_from_plans(plans: list[RenamePlan]) -> str:
    if not plans:
        return os.getcwd()
    olds = [os.path.dirname(p.old_path) for p in plans]
    common = os.path.commonpath(olds)
    return common


def apply(plans: list[RenamePlan], dest_root: str | None = None) -> str:
    dest_root = os.path.abspath(dest_root or _dest_root_from_plans(plans))
    os.makedirs("undo", exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.abspath(os.path.join("undo", f"undo-{stamp}.json"))
    moves = []
    for item in plans:
        src = item.old_path
        dest = item.new_path
        if not os.path.isabs(dest):
            dest = os.path.join(dest_root, dest)
        dest = os.path.abspath(dest)
        if not os.path.isfile(src):
            continue
        if os.path.abspath(src) == dest:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            stem, ext = os.path.splitext(dest)
            n = 2
            while os.path.exists(f"{stem}-{n}{ext}"):
                n += 1
            dest = f"{stem}-{n}{ext}"
        shutil.move(src, dest)
        moves.append({"old": src, "new": dest})
    with open(log_path, "w") as f:
        json.dump({"dest_root": dest_root, "moves": moves}, f, indent=2)
    return log_path


def undo(log_path: str) -> None:
    with open(log_path) as f:
        payload = json.load(f)
    for move in reversed(payload.get("moves", [])):
        src, dest = move["new"], move["old"]
        if not os.path.isfile(src):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            stem, ext = os.path.splitext(dest)
            n = 2
            while os.path.exists(f"{stem}-{n}{ext}"):
                n += 1
            dest = f"{stem}-{n}{ext}"
        shutil.move(src, dest)
