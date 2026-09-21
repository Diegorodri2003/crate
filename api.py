"""
api.py — D. Wires analyzer / index / organiser to HTTP + one HTML page.

Owned files: api.py, static/index.html. Nothing else.

USE_FIXTURES=False now that analyzer.py / index.py / organiser.py are real.
Every import of the other three is still defensive: if one raises on import
(missing, or mid-edit), we fall back to fixtures for whatever that module
would have done and report it via GET /api/status so the UI can show a
small "not live yet" banner. The page must never fail to load.

Stdlib only — no Flask/etc — so there is nothing to install to run this.
"""

from __future__ import annotations

import io
import json
import math
import mimetypes
import os
import random
import re
import struct
import sys
import tempfile
import threading
import time
import uuid
import wave
import zlib
from dataclasses import asdict, is_dataclass
from email.parser import BytesParser
from email.policy import compat32
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# --------------------------------------------------------------------------
# THE FLAG.
# --------------------------------------------------------------------------
USE_FIXTURES = False

# --------------------------------------------------------------------------
# Defensive imports of the other three modules. Any of these may not exist
# yet, may be half-written, or may raise on import — none of that should
# ever take this server down.
# --------------------------------------------------------------------------
try:
    import analyzer
except Exception:
    analyzer = None

try:
    import index
except Exception:
    index = None

try:
    import organiser
except Exception:
    organiser = None

# contracts.py is the shared foundation everyone already depends on (not
# one of the three modules under integration) — a plain, non-defensive
# import, same as make_fixtures.py's.
from contracts import SearchQuery, SampleRecord, RenamePlan


HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(HERE, "fixtures", "records.json")
STATIC_DIR = os.path.join(HERE, "static")

DEFAULT_TEMPLATE = "{family}/{instrument}/{filename}"


def _load_fixture_records() -> list[dict]:
    try:
        with open(FIXTURES_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


FIXTURE_RECORDS: list[dict] = _load_fixture_records()

# --------------------------------------------------------------------------
# In-memory state. "Nothing fancier" per the brief.
# --------------------------------------------------------------------------
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

LAST_PLANS: list[dict] = []
LAST_SCAN_RECORDS: list[dict] = []
UNDO_LOGS: dict[str, list[dict]] = {}
DEMO_AUDIO_CACHE: dict[str, bytes] = {}


def modules_missing() -> list[str]:
    missing = []
    if analyzer is None:
        missing.append("analyzer")
    if index is None:
        missing.append("index")
    if organiser is None:
        missing.append("organiser")
    return missing


def scan_available() -> bool:
    return (not USE_FIXTURES) and organiser is not None


def plan_available() -> bool:
    return (not USE_FIXTURES) and organiser is not None


def apply_available() -> bool:
    return (not USE_FIXTURES) and organiser is not None


def search_available() -> bool:
    return (not USE_FIXTURES) and index is not None


def fit_available() -> bool:
    return (not USE_FIXTURES) and analyzer is not None and index is not None


def to_dict(rec) -> dict:
    if is_dataclass(rec):
        return asdict(rec)
    if isinstance(rec, dict):
        return rec
    return {"value": rec}


_RECORD_FIELDS = set(SampleRecord.__dataclass_fields__)


def to_record(rec) -> SampleRecord:
    """Normalize a fixture dict (or an already-real SampleRecord) into a
    real dataclass instance. organiser.plan()/index.upsert() do attribute
    access (rec.filename, rec.path, ...) — a plain dict would blow up."""
    if isinstance(rec, SampleRecord):
        return rec
    if isinstance(rec, dict):
        return SampleRecord(**{k: v for k, v in rec.items() if k in _RECORD_FIELDS})
    return rec


def to_plan(p) -> RenamePlan:
    if isinstance(p, RenamePlan):
        return p
    if isinstance(p, dict):
        return RenamePlan(old_path=p.get("old_path"), new_path=p.get("new_path"), reason=p.get("reason", ""))
    return p


def _error_record(path: str, message: str) -> SampleRecord:
    return SampleRecord(
        path=path, filename=os.path.basename(path), duration_s=0.0, sample_rate=0,
        channels=0, peak_db=0.0, sample_type="unknown", bpm=None, key=None,
        family="unsorted", instrument="unknown", confidence=0.0, error=message,
    )


def strip_heavy(rec: dict) -> dict:
    """Embeddings are 512 floats each and the UI never touches them — drop
    them before they go over the wire, especially since scan progress ships
    the growing records list on every poll."""
    if "embedding" not in rec:
        return rec
    out = dict(rec)
    out.pop("embedding", None)
    return out


# --------------------------------------------------------------------------
# /api/status
# --------------------------------------------------------------------------
def status_payload() -> dict:
    return {
        "use_fixtures": USE_FIXTURES,
        "modules": {
            "analyzer": analyzer is not None,
            "index": index is not None,
            "organiser": organiser is not None,
        },
        "missing": modules_missing(),
    }


# --------------------------------------------------------------------------
# /api/scan + /api/scan/{id}
# --------------------------------------------------------------------------
def start_scan(root: str) -> dict:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"done": False, "total": 0, "records": [], "error": None, "root": root}
    t = threading.Thread(target=_run_scan_job, args=(job_id, root), daemon=True)
    t.start()
    return {"job_id": job_id}


def get_scan_status(job_id: str) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return {"error": "unknown job_id", "done": True, "total": 0, "records": []}
        return {"done": job["done"], "total": job["total"], "records": list(job["records"])}


def _run_scan_job(job_id: str, root: str) -> None:
    global LAST_SCAN_RECORDS
    try:
        if scan_available():
            paths = organiser.scan(root)
            with JOBS_LOCK:
                JOBS[job_id]["total"] = len(paths)
            records: list[SampleRecord] = []
            json_records: list[dict] = []
            for p in paths:
                try:
                    rec = analyzer.analyze(p) if analyzer is not None else to_record(_fixture_like_record(p))
                except Exception as e:
                    rec = _error_record(p, str(e))
                records.append(rec)
                json_records.append(strip_heavy(to_dict(rec)))
                with JOBS_LOCK:
                    JOBS[job_id]["records"] = list(json_records)
            with JOBS_LOCK:
                JOBS[job_id]["done"] = True
            LAST_SCAN_RECORDS = records
            if index is not None:
                try:
                    index.clear()  # the index must hold exactly this folder
                    index.upsert(records)
                except Exception:
                    pass  # search still works against whatever was already indexed
        else:
            total = len(FIXTURE_RECORDS)
            with JOBS_LOCK:
                JOBS[job_id]["total"] = total
            batch_json: list[dict] = []
            batch_obj: list[SampleRecord] = []
            for rec in FIXTURE_RECORDS:
                time.sleep(0.09)
                batch_json.append(strip_heavy(rec))
                batch_obj.append(to_record(rec))
                with JOBS_LOCK:
                    JOBS[job_id]["records"] = list(batch_json)
            with JOBS_LOCK:
                JOBS[job_id]["done"] = True
            LAST_SCAN_RECORDS = batch_obj
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["done"] = True


def run_ingest(root: str) -> dict:
    """Synchronous scan + analyse + index of `root`, for callers that want a
    record count back instead of a progress job. Clears the index first, so
    ingesting the same folder twice reports the same count both times."""
    global LAST_SCAN_RECORDS
    records: list[SampleRecord] = []
    if scan_available():
        for p in organiser.scan(root):
            try:
                rec = analyzer.analyze(p) if analyzer is not None else to_record(_fixture_like_record(p))
            except Exception as e:
                rec = _error_record(p, str(e))
            records.append(rec)
    else:
        records = [to_record(r) for r in FIXTURE_RECORDS]
    LAST_SCAN_RECORDS = records
    indexed = 0
    if index is not None:
        index.clear()
        index.upsert(records)
        indexed = len(index.all_records())
    return {"root": root, "count": len(records), "indexed": indexed}


def _fixture_like_record(path: str) -> dict:
    rec = dict(random.Random(zlib.crc32(path.encode())).choice(FIXTURE_RECORDS) or {})
    rec = strip_heavy(rec)
    rec["path"] = path
    rec["filename"] = os.path.basename(path)
    return rec


# --------------------------------------------------------------------------
# /api/plan
# --------------------------------------------------------------------------
def build_plan(template: str | None) -> list[dict]:
    global LAST_PLANS
    if plan_available():
        records = LAST_SCAN_RECORDS or []
        # organiser.plan() has its own default template with its own
        # placeholder fields ({descriptor}, {keyOrBpm}, ...) — ours
        # ({filename}) is only valid for the fixture fallback below, so we
        # only pass one through when the caller actually supplied it.
        plans = organiser.plan(records, template) if template else organiser.plan(records)
    else:
        plans = [to_plan(p) for p in _fixture_plan(template)]
    LAST_PLANS = list(plans)
    return [to_dict(p) for p in plans]


def _fixture_plan(template: str | None) -> list[dict]:
    tmpl = template or DEFAULT_TEMPLATE
    plans = []
    for rec in FIXTURE_RECORDS:
        old_path = rec["path"]
        if rec.get("error"):
            new_path = "/demo/sorted/_errors/" + rec["filename"]
            reason = f"quarantined — {rec['error']}"
        else:
            try:
                new_path = "/demo/sorted/" + tmpl.format(**rec).lstrip("/")
            except Exception:
                new_path = "/demo/sorted/" + DEFAULT_TEMPLATE.format(**rec)
            bits = [f"{rec['family']}/{rec['instrument']} match ({round(rec.get('confidence', 0) * 100)}% confidence)"]
            if rec.get("bpm"):
                bits.append(f"{rec['bpm']} BPM")
            if rec.get("key"):
                bits.append(f"key {rec['key']}")
            reason = ", ".join(bits)
        plans.append({"old_path": old_path, "new_path": new_path, "reason": reason})
    return plans


# --------------------------------------------------------------------------
# /api/apply + /api/undo
# --------------------------------------------------------------------------
def do_apply(dry_run: bool) -> dict:
    plans = LAST_PLANS or []
    if apply_available():
        # contracts.py documents apply(plans) -> str; the real organiser.py
        # takes an extra dry_run kwarg (default True) not in that contract.
        # We pass it through explicitly rather than special-case around it.
        undo_log = organiser.apply(plans, dry_run=dry_run)
        moved = 0 if dry_run else sum(1 for p in plans if p.old_path != p.new_path)
        return {"undo_log": undo_log or None, "moved": moved}
    plans_d = [to_dict(p) for p in plans]
    if dry_run:
        return {"undo_log": None, "moved": len(plans_d)}
    log_id = uuid.uuid4().hex[:12]
    UNDO_LOGS[log_id] = plans_d
    return {"undo_log": log_id, "moved": len(plans_d)}


def do_undo(log) -> dict:
    if apply_available() and isinstance(log, str) and os.path.exists(log):
        with open(log) as f:
            restored = sum(1 for line in f if line.strip())
        organiser.undo(log)
        return {"restored": restored}
    plans = UNDO_LOGS.pop(log, None) if isinstance(log, str) else None
    if plans is None:
        return {"restored": 0, "error": "unknown undo log"}
    return {"restored": len(plans)}


# --------------------------------------------------------------------------
# /api/search
# --------------------------------------------------------------------------
def run_search(body: dict) -> list[dict]:
    if search_available():
        q = SearchQuery(
            text=body.get("text"),
            family=body.get("family"),
            instrument=body.get("instrument"),
            sample_type=body.get("sample_type"),
            bpm_min=body.get("bpm_min"),
            bpm_max=body.get("bpm_max"),
            key=body.get("key"),
            fit_to=body.get("fit_to"),
            contrast=bool(body.get("contrast", False)),
            limit=int(body.get("limit", 50)),
        )
        hits = index.search(q)
        return [_hit_to_dict(h) for h in hits]
    return _fixture_search(body)


def _hit_to_dict(hit) -> dict:
    d = to_dict(hit)
    if "record" in d:
        d = dict(d)
        d["record"] = strip_heavy(to_dict(d["record"]))
    return d


def _fixture_search(query: dict) -> list[dict]:
    text = (query.get("text") or "").strip().lower()
    family = query.get("family") or None
    instrument = query.get("instrument") or None
    sample_type = query.get("sample_type") or None
    key = query.get("key") or None
    bpm_min = query.get("bpm_min")
    bpm_max = query.get("bpm_max")
    limit = int(query.get("limit") or 50)

    hits = []
    for rec in FIXTURE_RECORDS:
        if rec.get("error"):
            continue
        if family and rec.get("family") != family:
            continue
        if instrument and rec.get("instrument") != instrument:
            continue
        if sample_type and rec.get("sample_type") != sample_type:
            continue
        if key and rec.get("key") != key:
            continue
        if bpm_min is not None and (rec.get("bpm") is None or rec["bpm"] < float(bpm_min)):
            continue
        if bpm_max is not None and (rec.get("bpm") is None or rec["bpm"] > float(bpm_max)):
            continue

        score = 0.4 + 0.3 * rec.get("confidence", 0.5)
        reasons = [f"{rec.get('family')} / {rec.get('instrument')}"]
        if text:
            hay = " ".join([
                rec.get("filename", ""), rec.get("family", ""), rec.get("instrument", ""),
                " ".join(rec.get("descriptors") or []),
            ]).lower()
            if text not in hay:
                continue
            score += 0.3
            reasons.insert(0, f"matches \u201c{text}\u201d")
        if rec.get("bpm"):
            reasons.append(f"{rec['bpm']} BPM")
        if rec.get("key"):
            reasons.append(f"key {rec['key']}")
        hits.append({"record": strip_heavy(rec), "score": round(min(score, 1.0), 3), "reasons": reasons})

    hits.sort(key=lambda h: -h["score"])
    return hits[:limit]


# --------------------------------------------------------------------------
# /api/fit (multipart upload)
# --------------------------------------------------------------------------
def run_fit(temp_path: str, contrast: bool, limit: int = 20) -> list[dict]:
    if fit_available():
        rec = analyzer.analyze(temp_path)
        # index.search()'s fit_to looks the reference up by path in its own
        # store — it has to be upserted first or the lookup misses.
        index.upsert([rec])
        q = SearchQuery(fit_to=temp_path, limit=limit, contrast=contrast)
        hits = index.search(q)
        return [_hit_to_dict(h) for h in hits]
    return _fixture_fit_hits(contrast, limit)


def _fixture_fit_hits(contrast: bool, limit: int) -> list[dict]:
    recs = [r for r in FIXTURE_RECORDS if not r.get("error")]
    rnd = random.Random(42 if not contrast else 137)
    picks = rnd.sample(recs, k=min(limit, len(recs)))
    hits = []
    for rec in picks:
        base = rec.get("confidence", 0.5)
        score = round(min(1.0, base * (0.5 if contrast else 1.0) + rnd.uniform(0.0, 0.35)), 3)
        reasons = [f"{rec['family']} / {rec['instrument']}"]
        if rec.get("bpm"):
            delta = rnd.uniform(-4, 4)
            pct = round(delta / rec["bpm"] * 100, 1)
            reasons.append(f"{rec['bpm']} \u2192 {round(rec['bpm'] + delta, 1)} BPM ({'+' if pct >= 0 else ''}{pct}%)")
        reasons.append("opposite brightness" if contrast else "fills 4\u20138kHz gap")
        hits.append({"record": strip_heavy(rec), "score": score, "reasons": reasons})
    hits.sort(key=lambda h: -h["score"])
    return hits


# --------------------------------------------------------------------------
# /api/audio
# --------------------------------------------------------------------------
def demo_tone_bytes(path: str) -> bytes:
    if path in DEMO_AUDIO_CACHE:
        return DEMO_AUDIO_CACHE[path]
    seed = zlib.crc32((path or "silence").encode())
    rnd = random.Random(seed)
    sr = 22050
    freq = rnd.uniform(110, 1400)
    dur = rnd.uniform(0.4, 1.1)
    n = int(sr * dur)
    frames = bytearray()
    for i in range(n):
        env = math.exp(-3.0 * i / n)
        v = math.sin(2 * math.pi * freq * i / sr) * env
        frames += struct.pack("<h", int(max(-1.0, min(1.0, v)) * 22000))
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))
    data = buf.getvalue()
    DEMO_AUDIO_CACHE[path] = data
    return data


def load_audio_bytes(path: str) -> tuple[bytes, str]:
    if path and os.path.isfile(path):
        ctype = mimetypes.guess_type(path)[0] or "audio/wav"
        with open(path, "rb") as f:
            return f.read(), ctype
    return demo_tone_bytes(path), "audio/wav"


# --------------------------------------------------------------------------
# multipart/form-data parsing (stdlib-only, no cgi module)
# --------------------------------------------------------------------------
def parse_multipart(content_type: str, body: bytes) -> dict:
    """Returns {field_name: {"filename": str|None, "data": bytes|str}}."""
    header = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
    msg = BytesParser(policy=compat32).parsebytes(header + body)
    fields: dict[str, dict] = {}
    if not msg.is_multipart():
        return fields
    for part in msg.get_payload():
        disp = part.get("Content-Disposition", "")
        m = re.search(r'name="([^"]*)"', disp)
        if not m:
            continue
        name = m.group(1)
        fm = re.search(r'filename="([^"]*)"', disp)
        payload = part.get_payload(decode=True)
        if fm:
            fields[name] = {"filename": fm.group(1), "data": payload or b""}
        else:
            try:
                fields[name] = {"filename": None, "data": (payload or b"").decode("utf-8")}
            except Exception:
                fields[name] = {"filename": None, "data": payload}
    return fields


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "CrateAPI/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers -----------------------------------------------------------
    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _bytes(self, data: bytes, content_type: str, extra_headers: dict | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(length) if length else b""

    def _read_json(self) -> dict:
        raw = self._read_body()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _serve_static(self, rel_path: str) -> None:
        full = os.path.normpath(os.path.join(HERE, rel_path))
        if not full.startswith(HERE) or not os.path.isfile(full):
            return self._json({"error": f"not found: {rel_path}"}, 404)
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            data = f.read()
        self._bytes(data, ctype)

    # -- routing -------------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                return self._serve_static("static/index.html")
            if path.startswith("/static/"):
                return self._serve_static(path.lstrip("/"))
            if path == "/api/status":
                return self._json(status_payload())
            if path == "/api/audio":
                data, ctype = load_audio_bytes((qs.get("path") or [""])[0])
                return self._bytes(data, ctype, {"Cache-Control": "no-store"})
            m = re.match(r"^/api/scan/([^/]+)$", path)
            if m:
                return self._json(get_scan_status(m.group(1)))
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        try:
            if path == "/api/scan":
                body = self._read_json()
                return self._json(start_scan(body.get("root", "")))
            if path in ("/api/ingest", "/ingest"):
                body = self._read_json()
                return self._json(run_ingest(body.get("root", "")))
            if path == "/api/plan":
                body = self._read_json()
                return self._json({"plans": build_plan(body.get("template"))})
            if path == "/api/apply":
                body = self._read_json()
                return self._json(do_apply(bool(body.get("dry_run", True))))
            if path == "/api/undo":
                body = self._read_json()
                return self._json(do_undo(body.get("log")))
            if path == "/api/search":
                body = self._read_json()
                return self._json({"hits": run_search(body)})
            if path == "/api/fit":
                return self._json({"hits": self._handle_fit(qs)})
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    def _handle_fit(self, qs: dict) -> list[dict]:
        content_type = self.headers.get("Content-Type", "")
        body = self._read_body()
        contrast = (qs.get("contrast") or ["false"])[0].lower() == "true"
        temp_path = None
        if content_type.startswith("multipart/form-data"):
            fields = parse_multipart(content_type, body)
            file_field = fields.get("file") or next(
                (v for v in fields.values() if v.get("filename")), None
            )
            if fields.get("contrast") and isinstance(fields["contrast"].get("data"), str):
                contrast = fields["contrast"]["data"].strip().lower() == "true"
            if file_field and file_field.get("data"):
                suffix = os.path.splitext(file_field.get("filename") or "upload.wav")[1] or ".wav"
                fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="crate_fit_")
                with os.fdopen(fd, "wb") as f:
                    f.write(file_field["data"])
        try:
            return run_fit(temp_path or "", contrast)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


def run_server(port: int = 8000) -> None:
    if modules_missing():
        print(f"[api] running with fixtures — not live yet: {', '.join(modules_missing())}")
    print(f"[api] USE_FIXTURES={USE_FIXTURES}  http://127.0.0.1:{port}/")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8000))
    run_server(port)
