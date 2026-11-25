import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phase_c
import run_cuda as c


class MultiUnitTests(unittest.TestCase):
    def test_order_and_unit_receipts(self):
        plan = c.generate_plan(8,431)
        starts = [r["order"] for r in plan[::4]]
        self.assertEqual(starts.count("AB"),4)
        specs = list(c.expected_events({"plan":plan,"purpose":"multi_unit_comparison"}))
        self.assertEqual([e["index"] for e in specs if e["type"]=="unit"],list(range(8)))
        for i in range(0,len(plan),4):
            self.assertNotEqual(plan[i]["order"],plan[i+2]["order"])

    def test_flag_without_deletion(self):
        values = [1,1,1,1,7]
        lo,hi = phase_c.fences(values)
        self.assertEqual([x for x in values if not lo<=x<=hi],[7])

    def test_failure_excludes_partial_unit(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            manifest = {"unit_count":2,"plan":c.generate_plan(2,1)}
            c.archive.write_new(p/"manifest.json",manifest)
            attempt = p/"attempt"
            attempt.mkdir()
            c.archive.write_new(attempt/"terminal.json",{"status":"failed"})
            (attempt/"events.jsonl").write_bytes(b'{"partial"')
            (p/"telemetry.jsonl").write_bytes(b'')
            phase_c.report(p)
            pairs = json.loads((p/"pairs.json").read_text())
            self.assertEqual(len(pairs),4)
            self.assertFalse(any(r["valid"] for r in pairs))

    def test_missing_telemetry(self):
        with patch.object(phase_c.subprocess,"run",side_effect=OSError("no sampler")):
            sample = phase_c.sample()
        self.assertIsNone(sample["power_w"])
        self.assertIn("error",sample)
