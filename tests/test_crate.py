"""End-to-end tests for analyzer, index, organiser, and the HTTP API."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import analyzer
import api as crate_api
import index
import organiser
from contracts import SearchQuery, SampleRecord
from make_fixtures import fake_record, tone_wav, write_fixtures


def _tone(path: str, freq: float = 110.0, dur: float = 0.4) -> None:
    tone_wav(path, freq, dur, noise=0.02)


class AnalyzerTests(unittest.TestCase):
    def test_missing_file_sets_error(self):
        rec = analyzer.analyze("/definitely/not/here.wav")
        self.assertIsNotNone(rec.error)
        self.assertEqual(rec.family, "unsorted")
        self.assertEqual(rec.instrument, "unknown")

    def test_wav_fills_required_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "KICK_FINAL_v1_USE_THIS.wav")
            _tone(path, 60, 0.35)
            rec = analyzer.analyze(path)
            self.assertIsNone(rec.error)
            self.assertEqual(rec.filename, "KICK_FINAL_v1_USE_THIS.wav")
            self.assertEqual(rec.sample_rate, 44100)
            self.assertEqual(len(rec.band_energy), 8)
            self.assertAlmostEqual(sum(rec.band_energy), 1.0, places=2)
            self.assertEqual(len(rec.embedding), 512)
            self.assertEqual(rec.family, "Drums")
            self.assertEqual(rec.instrument, "kick")
            self.assertGreaterEqual(rec.confidence, 0.5)
            self.assertLessEqual(rec.peak_db, 0.0)


class IndexTests(unittest.TestCase):
    def setUp(self):
        index.clear()
        recs = [fake_record(i) for i in range(8)]
        recs[0].family = "Drums"
        recs[0].instrument = "kick"
        recs[0].filename = "KICK_FINAL.wav"
        recs[0].path = "/demo/messy/KICK_FINAL.wav"
        recs[0].descriptors = ["dark", "punchy"]
        recs[1].error = "could not decode: unsupported codec"
        recs[1].family, recs[1].instrument, recs[1].confidence = "unsorted", "unknown", 0.0
        recs[1].embedding = None
        index.upsert(recs)
        self.recs = recs

    def test_family_filter_and_text(self):
        hits = index.search(SearchQuery(family="Drums", text="kick", limit=20))
        self.assertTrue(hits)
        self.assertTrue(all(h.record.family == "Drums" for h in hits))
        self.assertEqual(hits[0].record.filename, "KICK_FINAL.wav")

    def test_broken_record_does_not_crash(self):
        hits = index.search(SearchQuery(limit=50))
        self.assertEqual(len(hits), 8)
        broken = [h for h in hits if h.record.error]
        self.assertEqual(len(broken), 1)
        self.assertLess(broken[0].score, 0.2)

    def test_fit_to_contrast(self):
        ref = self.recs[0]
        hits = index.search(SearchQuery(fit_to=ref.path, contrast=True, limit=10))
        self.assertTrue(hits)
        self.assertTrue(any("band" in " ".join(h.reasons) or "embed" in " ".join(h.reasons) or "fills" in " ".join(h.reasons) for h in hits))


class OrganiserTests(unittest.TestCase):
    def test_scan_plan_apply_undo(self):
        with tempfile.TemporaryDirectory() as td:
            nested = os.path.join(td, "New Folder", "untitled")
            os.makedirs(nested)
            a = os.path.join(td, "KICK_FINAL_v1.wav")
            b = os.path.join(nested, "Untitled-2.wav")
            _tone(a, 60, 0.3)
            _tone(b, 1800, 0.12)
            files = organiser.scan(td)
            self.assertEqual(len(files), 2)
            recs = [analyzer.analyze(p) for p in files]
            plans = organiser.plan(recs, "{family}/{instrument}/{stem}.{ext}")
            self.assertEqual(len(plans), 2)
            self.assertTrue(all("/" in p.new_path for p in plans))
            log = organiser.apply(plans, dest_root=td)
            self.assertTrue(os.path.isfile(log))
            remaining = organiser.scan(td)
            self.assertEqual(len(remaining), 2)
            self.assertFalse(os.path.isfile(a))
            organiser.undo(log)
            restored = {os.path.basename(p) for p in organiser.scan(td)}
            self.assertEqual(restored, {"KICK_FINAL_v1.wav", "Untitled-2.wav"})


class FixtureShapeTests(unittest.TestCase):
    def test_write_fixtures_shape(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd()
            os.chdir(td)
            try:
                write_fixtures()
                with open("fixtures/records.json") as f:
                    rows = json.load(f)
            finally:
                os.chdir(cwd)
        self.assertEqual(len(rows), 25)
        self.assertTrue(any(r.get("error") for r in rows))
        SampleRecord(**{k: v for k, v in rows[0].items()})


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.root = Path(cls.td.name) / "lib"
        cls.root.mkdir()
        _tone(str(cls.root / "KICK_FINAL_v3.wav"), 55, 0.25)
        _tone(str(cls.root / "pad_warm.wav"), 220, 1.4)
        cls.port = 18765
        os.environ["CRATE_HOST"] = "127.0.0.1"
        os.environ["CRATE_PORT"] = str(cls.port)
        index.clear()
        crate_api.HOST = "127.0.0.1"
        crate_api.PORT = cls.port
        cls.httpd = crate_api.ThreadingHTTPServer(("127.0.0.1", cls.port), crate_api.Handler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.15)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.td.cleanup()

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as res:
            return json.loads(res.read().decode())

    def test_page_and_ingest_search_plan(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=5) as res:
            html = res.read().decode()
        self.assertIn("<title>crate</title>", html)
        ingested = self._post("/ingest", {"root": str(self.root)})
        self.assertEqual(ingested["count"], 2)
        hits = self._post("/search", {"text": "kick", "family": "Drums"})
        self.assertGreaterEqual(len(hits["hits"]), 1)
        planned = self._post("/plan", {"tmpl": "{family}/{instrument}/{stem}.{ext}"})
        self.assertEqual(len(planned["plans"]), 2)


if __name__ == "__main__":
    unittest.main()
