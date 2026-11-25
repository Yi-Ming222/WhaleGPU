"""CPU-only fault injection. All fake executable identities live in a temporary archive."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_cuda as c


class FakeWorker:
    pid = 1701

    def __init__(self, events, exit_code):
        self.stdout = io.StringIO("".join(json.dumps(e)+"\n" for e in events))
        self.exit_code = exit_code
        self.returncode = None

    def wait(self):
        self.returncode = self.exit_code
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.exit_code = -9


class CudaLinkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/"r"
        root_patch = patch.object(c,"ROOT",self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    @staticmethod
    def fake_compile(work):
        exe = work/c.EXE
        exe.write_bytes(b"CPU fixture - never executed")
        return {"headers":{}, "exe_sha256":c.sha(exe)}

    def fixture(self):
        with patch.object(c,"compile_executor",self.fake_compile):
            run_id = c.build(431)
        p = c.run_path(run_id)
        manifest = c.archive.read_json(p/"manifest.json")
        events = list(c.expected_events(manifest))
        for event in events:
            if event["type"] == "ready":
                event.update(manifest["matrix_hashes"],os_pid=1701,input_roundtrip=True,
                             device="NVIDIA GeForce RTX 5070 Laptop GPU")
            if event["type"] == "check":
                event.update(poison_byte=255,elements=513*769,mismatches=0,
                             max_abs_error=0.0,max_error_ratio=0.0)
            if event["type"] == "batch":
                event.update(stop_synchronized=True,stream=0,submitted=10,completed=10,event_total_ms=1)
        return p,events

    def rejected(self,p,events,rc=0):
        with patch.object(c.subprocess,"Popen",return_value=FakeWorker(events,rc)):
            with self.assertRaises(ValueError):
                c.verify(p.name)
        terminal = c.archive.read_json(p/"attempt/terminal.json")
        self.assertFalse(terminal["unit_complete"])
        self.assertEqual(terminal["status"],"failed")
        return terminal

    def test_build_failure(self):
        with patch.object(c,"compile_executor",side_effect=OSError("compiler failed")):
            with self.assertRaises(OSError):
                c.build(431)
        p, = self.root.iterdir()
        self.assertTrue((p/"build-error.json").exists())
        self.assertFalse((p/"build.json").exists())

    def test_async_device_failure(self):
        p,events = self.fixture()
        failure = {"type":"batch_error","index":0,"submitted_acknowledged":10,"completed":None}
        self.rejected(p,events[:4]+[failure],1)
        last = (p/"attempt/events.jsonl").read_text().splitlines()[-1]
        self.assertEqual(json.loads(last)["event"],failure)

    def test_wrong_answer_after_all_batches(self):
        p,events = self.fixture()
        events[-1]["mismatches"] = 1
        self.rejected(p,events)

    def test_count_mismatch(self):
        p,events = self.fixture()
        next(x for x in events if x["type"]=="batch")["submitted"] = 9
        self.rejected(p,events)

    def test_abnormal_exit_after_receipts(self):
        p,events = self.fixture()
        self.assertEqual(self.rejected(p,events,37)["exit_code"],37)

    def test_exit_in_flight(self):
        p,events = self.fixture()
        self.assertEqual(self.rejected(p,events[:4],37)["unacknowledged_completion"],"unknown")

    def test_collision_does_not_overwrite(self):
        p,events = self.fixture()
        with patch.object(c.subprocess,"Popen",return_value=FakeWorker(events,0)):
            c.verify(p.name)
        sealed = (p/"attempt/terminal.json").read_bytes()
        with self.assertRaises(FileExistsError):
            c.verify(p.name)
        self.assertEqual((p/"attempt/terminal.json").read_bytes(),sealed)

    def test_plan_actual_mismatch(self):
        p,events = self.fixture()
        row = next(x for x in events if x["type"]=="begin")
        row["kernel_id"] = next(k for k in c.KERNELS.values() if k != row["kernel_id"])
        self.rejected(p,events)

    def test_input_identity_mismatch(self):
        p,events = self.fixture()
        events[0]["a_sha256"] = "0"*64
        self.rejected(p,events)

    def test_unsynchronized_batch(self):
        p,events = self.fixture()
        next(x for x in events if x["type"]=="batch")["stop_synchronized"] = False
        self.rejected(p,events)


if __name__ == "__main__":
    unittest.main()
