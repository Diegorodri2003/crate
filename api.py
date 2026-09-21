"""D — HTTP API + one HTML page that wires A/B/C together."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# re-export for tests
__all__ = ["Handler", "main", "ThreadingHTTPServer"]
from urllib.parse import urlparse

from analyzer import analyze
from contracts import SearchQuery
from index import all_records, load, save, search as index_search, upsert
from organiser import apply as org_apply
from organiser import plan as org_plan
from organiser import scan as org_scan
from organiser import undo as org_undo

HOST = os.environ.get("CRATE_HOST", "127.0.0.1")
PORT = int(os.environ.get("CRATE_PORT", "8765"))
PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>crate</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { margin: 0; font: 14px/1.45 ui-sans-serif, system-ui, sans-serif;
           background: #111214; color: #ececec; }
    header { padding: 20px 24px 8px; display: flex; gap: 16px; align-items: baseline; }
    h1 { font-size: 28px; letter-spacing: -0.04em; margin: 0; }
    .muted { color: #9a9aa3; }
    main { display: grid; grid-template-columns: 320px 1fr; gap: 20px; padding: 12px 24px 40px; }
    section { background: #1a1b1f; border: 1px solid #2a2b31; border-radius: 12px; padding: 16px; }
    label { display: block; font-size: 12px; color: #9a9aa3; margin: 10px 0 4px; }
    input, select, button { width: 100%; padding: 8px 10px; border-radius: 8px;
      border: 1px solid #34353c; background: #101114; color: inherit; }
    button { cursor: pointer; background: #ececec; color: #111; font-weight: 600; margin-top: 10px; }
    button.ghost { background: transparent; color: #ececec; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid #2a2b31; vertical-align: top; }
    th { color: #9a9aa3; font-size: 12px; font-weight: 500; }
    .pill { display: inline-block; padding: 1px 8px; border-radius: 999px;
            background: #2a2b31; font-size: 12px; }
    .err { color: #ff8b8b; }
    pre { white-space: pre-wrap; font-size: 12px; color: #9a9aa3; }
  </style>
</head>
<body>
  <header>
    <h1>crate</h1>
    <span class="muted">sample librarian — scan, tag, search, rename</span>
  </header>
  <main>
    <section>
      <label>library folder</label>
      <input id="root" value="messy"/>
      <button onclick="ingest()">scan + analyze</button>
      <label>search text</label>
      <input id="text" placeholder="kick dark 128"/>
      <div class="row">
        <div>
          <label>family</label>
          <select id="family">
            <option value="">any</option>
            <option>Drums</option><option>Bass</option><option>Acoustic</option>
            <option>Synth</option><option>Vocal</option><option>FX</option>
            <option>unsorted</option>
          </select>
        </div>
        <div>
          <label>type</label>
          <select id="sample_type">
            <option value="">any</option>
            <option>oneshot</option><option>loop</option><option>unknown</option>
          </select>
        </div>
      </div>
      <div class="row">
        <div><label>bpm min</label><input id="bpm_min" type="number" step="0.1"/></div>
        <div><label>bpm max</label><input id="bpm_max" type="number" step="0.1"/></div>
      </div>
      <label>rename template</label>
      <input id="tmpl" value="{family}/{instrument}/{stem}.{ext}"/>
      <button onclick="runSearch()">search</button>
      <button class="ghost" onclick="previewPlan()">preview rename plan</button>
      <button onclick="doApply()">apply plan</button>
      <button class="ghost" onclick="doUndo()">undo last apply</button>
      <pre id="status">idle</pre>
    </section>
    <section>
      <table>
        <thead>
          <tr><th>file</th><th>class</th><th>type</th><th>score</th><th>why</th></tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </section>
  </main>
<script>
let lastLog = null;
function setStatus(s) { document.getElementById('status').textContent = s; }
async function api(path, body) {
  const res = await fetch(path, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify(body || {})
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
function renderHits(hits) {
  const tb = document.getElementById('rows');
  tb.innerHTML = (hits || []).map(h => {
    const r = h.record || h;
    const cls = r.error ? 'err' : '';
    const klass = r.family + ' / ' + r.instrument;
    return `<tr class="${cls}">
      <td>${r.filename}<div class="muted">${r.path}</div></td>
      <td><span class="pill">${klass}</span><div class="muted">${(r.descriptors||[]).join(', ')}</div></td>
      <td>${r.sample_type}${r.bpm ? ' · ' + r.bpm : ''}${r.key ? ' · ' + r.key : ''}</td>
      <td>${h.score != null ? h.score : r.confidence}</td>
      <td>${(h.reasons || [r.error || '']).join('<br/>')}</td>
    </tr>`;
  }).join('');
}
async function ingest() {
  setStatus('analyzing…');
  const data = await api('/ingest', { root: document.getElementById('root').value });
  setStatus(`indexed ${data.count} files`);
  renderHits(data.records.map(r => ({record: r, score: r.confidence, reasons: r.error ? [r.error] : r.descriptors})));
}
async function runSearch() {
  const q = {
    text: document.getElementById('text').value || null,
    family: document.getElementById('family').value || null,
    sample_type: document.getElementById('sample_type').value || null,
    bpm_min: document.getElementById('bpm_min').value ? Number(document.getElementById('bpm_min').value) : null,
    bpm_max: document.getElementById('bpm_max').value ? Number(document.getElementById('bpm_max').value) : null,
  };
  const data = await api('/search', q);
  setStatus(`${data.hits.length} hits`);
  renderHits(data.hits);
}
async function previewPlan() {
  const data = await api('/plan', { tmpl: document.getElementById('tmpl').value });
  setStatus(`${data.plans.length} planned moves`);
  const tb = document.getElementById('rows');
  tb.innerHTML = data.plans.map(p => `<tr>
    <td>${p.old_path}</td><td colspan="3">${p.new_path}</td><td>${p.reason}</td>
  </tr>`).join('');
}
async function doApply() {
  const data = await api('/apply', {
    tmpl: document.getElementById('tmpl').value,
    dest_root: document.getElementById('root').value
  });
  lastLog = data.log_path;
  setStatus('applied. undo log: ' + data.log_path);
}
async function doUndo() {
  const data = await api('/undo', { log_path: lastLog });
  setStatus('undone: ' + (data.log_path || lastLog));
}
</script>
</body>
</html>
"""


def _json(handler: SimpleHTTPRequestHandler, code: int, payload: dict) -> None:
    raw = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _read_json(handler: SimpleHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or "0")
    if length == 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8") or "{}")


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print("[api]", fmt % args)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/records":
            _json(self, 200, {"records": [asdict(r) for r in all_records()]})
            return
        _json(self, 404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = _read_json(self)
            if path == "/ingest":
                root = body.get("root") or "messy"
                files = org_scan(root)
                recs = [analyze(p) for p in files]
                upsert(recs)
                save()
                _json(self, 200, {"count": len(recs), "records": [asdict(r) for r in recs]})
                return
            if path == "/search":
                q = SearchQuery(
                    text=body.get("text") or None,
                    family=body.get("family") or None,
                    instrument=body.get("instrument") or None,
                    sample_type=body.get("sample_type") or None,
                    bpm_min=body.get("bpm_min"),
                    bpm_max=body.get("bpm_max"),
                    key=body.get("key") or None,
                    fit_to=body.get("fit_to") or None,
                    contrast=bool(body.get("contrast") or False),
                    limit=int(body.get("limit") or 50),
                )
                hits = index_search(q)
                _json(
                    self,
                    200,
                    {
                        "hits": [
                            {"record": asdict(h.record), "score": h.score, "reasons": h.reasons}
                            for h in hits
                        ]
                    },
                )
                return
            if path == "/plan":
                plans = org_plan(all_records(), body.get("tmpl") or "")
                _json(self, 200, {"plans": [asdict(p) for p in plans]})
                return
            if path == "/apply":
                plans = org_plan(all_records(), body.get("tmpl") or "")
                log_path = org_apply(plans, body.get("dest_root"))
                recs = [analyze(p.new_path if os.path.isabs(p.new_path) else os.path.join(os.path.abspath(body.get("dest_root") or "."), p.new_path)) for p in plans]
                # reindex from remaining library
                root = body.get("dest_root") or "messy"
                recs = [analyze(p) for p in org_scan(root)]
                upsert(recs)
                save()
                _json(self, 200, {"log_path": log_path, "count": len(recs)})
                return
            if path == "/undo":
                log_path = body.get("log_path")
                if not log_path:
                    files = sorted(
                        (os.path.join("undo", n) for n in os.listdir("undo") if n.endswith(".json")),
                        reverse=True,
                    ) if os.path.isdir("undo") else []
                    log_path = files[0] if files else None
                if not log_path:
                    raise ValueError("no undo log")
                org_undo(log_path)
                _json(self, 200, {"log_path": log_path})
                return
            _json(self, 404, {"error": "not found"})
        except Exception as exc:
            _json(self, 400, {"error": str(exc)})


def main() -> None:
    load()
    server = ThreadingHTTPServer((HOST, PORT), partial(Handler, directory="."))
    print(f"crate ui on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
