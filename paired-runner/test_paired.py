import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import archive
from plan import create_manifest, generate_plan, validate_manifest
import run_paired


class InfrastructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = Path(tempfile.mkdtemp(prefix="whalegpu-phase-a-"))
        print(f"Retained test archives: {cls.scratch}", flush=True)

    def setUp(self):
        # The sandbox's Windows TEMP prefix is long. Keep fixture paths short.
        self.work = Path(tempfile.mkdtemp(prefix="t", dir=self.scratch))
        self.root = self.work / "artifacts"
        setting = patch.object(archive, "ROOT", self.root)
        setting.start()
        self.addCleanup(setting.stop)

    def reserve(self):
        m = create_manifest(2, 431)
        archive.reserve(m)
        return m

    def test_seed_replay_and_variation(self):
        self.assertEqual(generate_plan(8, 42), generate_plan(8, 42))
        orders = {tuple(x["kernel_id"] for x in generate_plan(8, s)) for s in range(8)}
        self.assertGreater(len(orders), 1)

    def test_even_and_odd_balance_and_ids(self):
        for n in (1, 2, 7, 8):
            rows = generate_plan(n, 19)
            self.assertEqual(len({r["batch_id"] for r in rows}), n*4)
            self.assertEqual(len({r["block_id"] for r in rows}), n*2)
            first = [r["order"] for r in rows[::4]]
            self.assertLessEqual(abs(first.count("AB")-first.count("BA")), 1)
            for i in range(0, len(rows), 4):
                u = rows[i:i+4]
                self.assertEqual({r["unit_id"] for r in u}, {u[0]["unit_id"]})
                self.assertEqual({u[0]["order"], u[2]["order"]}, {"AB", "BA"})
                self.assertEqual([r["position"] for r in u], [1, 2, 1, 2])

    def test_illegal_arguments(self):
        for n, seed in [(0, 10), (-1, 3), (True, 1), (2, "1"), (2, False)]:
            with self.assertRaises(ValueError):
                generate_plan(n, seed)
        for bad in ("../WhaleGPU-Phase34", "", "C:/Windows"):
            with self.assertRaises(ValueError):
                archive.directory(bad)

    def test_manifest_roundtrip_and_process_namespaces(self):
        m = self.reserve()
        self.assertEqual(archive.load(m["process_id"]), m)
        second = create_manifest(2, 431, m["session_id"], m["experiment_id"])
        self.assertEqual([r["kernel_id"] for r in second["experiment_plan"]],
                         [r["kernel_id"] for r in m["experiment_plan"]])
        self.assertTrue({r["batch_id"] for r in second["experiment_plan"]}.isdisjoint(
                        r["batch_id"] for r in m["experiment_plan"]))
        self.assertNotEqual(second["process_id"], m["process_id"])
        archive.reserve(second)
        self.assertEqual(archive.inspect(second["process_id"])["execution_status"], "planned")

    def test_manifest_rejects_missing_invalid_and_truncated_plans(self):
        m = create_manifest(2, 17)
        for action in (lambda x: x.pop("input_hash"),
                       lambda x: x.update(executable_hash="bad"),
                       lambda x: x["experiment_plan"].pop(),
                       lambda x: x.update(creation_timestamp="invalid"),
                       lambda x: x.update(schema_version=True)):
            candidate = copy.deepcopy(m)
            action(candidate)
            with self.assertRaises(ValueError):
                validate_manifest(candidate)

    def test_conflict_preserves_manifest_and_existing_result(self):
        m = self.reserve()
        p = archive.directory(m["process_id"]) / "manifest.json"
        original = p.read_bytes()
        with self.assertRaises(FileExistsError):
            archive.reserve(m)
        archive.execute(m["process_id"], run_paired.observations)
        before = archive.inspect(m["process_id"])
        with self.assertRaises(FileExistsError):
            archive.execute(m["process_id"], run_paired.observations)
        self.assertEqual(archive.inspect(m["process_id"]), before)
        self.assertEqual(p.read_bytes(), original)
        self.assertEqual(before["execution_status"], "success")
        journal = (p.parent / "attempt" / "batches.jsonl").read_bytes()
        self.assertEqual(hashlib.sha256(journal).hexdigest(), before["terminal"]["journal_sha256"])
        self.assertEqual(before["eligible_measurement_units"], [])
        self.assertTrue(all(r["actual_launch_count"] == 0 and r["os_pid"] != os.getpid()
                            for r in before["records"]))

    def test_failure_preserves_partial_unit(self):
        m = self.reserve()
        def failing(plan):
            for i, row in enumerate(plan["experiment_plan"][:5]):
                yield archive.receipt(row, i, os.getpid())
            raise RuntimeError("injected observation failure")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            archive.execute(m["process_id"], failing)
        result = archive.inspect(m["process_id"])
        self.assertEqual(result["execution_status"], "failed")
        unit1, unit2 = result["units"].values()
        self.assertTrue(unit1["complete"])
        self.assertFalse(unit2["complete"])
        self.assertEqual(len(unit2["missing_batch_ids"]), 3)
        self.assertEqual(len(result["records"]), 5)
        self.assertEqual(result["terminal"]["error_type"], "RuntimeError")
        self.assertEqual(result["eligible_measurement_units"], [])

    def test_empty_worker_and_reordered_receipt_fail(self):
        for source in (lambda m: iter(()),
                       lambda m: [archive.receipt(m["experiment_plan"][1], 0, os.getpid())]):
            m = self.reserve()
            with self.assertRaises(ValueError):
                archive.execute(m["process_id"], source)
            self.assertEqual(archive.inspect(m["process_id"])["execution_status"], "failed")

    def test_keyboard_interrupt(self):
        m = self.reserve()
        def interrupt(plan):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            archive.execute(m["process_id"], interrupt)
        self.assertEqual(archive.inspect(m["process_id"])["execution_status"], "interrupted")

    def test_worker_failure_after_last_receipt(self):
        m = self.reserve()
        def late_failure(plan):
            for i, row in enumerate(plan["experiment_plan"]):
                yield archive.receipt(row, i, os.getpid())
            raise subprocess.CalledProcessError(37, "test-worker")
        with self.assertRaises(subprocess.CalledProcessError):
            archive.execute(m["process_id"], late_failure)
        result = archive.inspect(m["process_id"])
        self.assertTrue(all(unit["complete"] for unit in result["units"].values()))
        self.assertEqual(result["execution_status"], "failed")
        self.assertEqual(result["eligible_measurement_units"], [])

    def test_incomplete_reservation(self):
        m = create_manifest(1, 2)
        path = archive.directory(m["process_id"])
        path.mkdir(parents=True)
        result = archive.inspect(m["process_id"])
        self.assertEqual(result["execution_status"], "incomplete")
        self.assertTrue(result["issues"])
        with self.assertRaises(FileExistsError):
            archive.reserve(m)

    def test_process_death_after_durable_batch(self):
        m = self.reserve()
        script = """
import os,sys
from pathlib import Path
import archive
archive.ROOT = Path(sys.argv[1])
def die(plan):
    yield archive.receipt(plan['experiment_plan'][0], 0, os.getpid())
    os._exit(37)
archive.execute(sys.argv[2], die)
"""
        rc = subprocess.run([sys.executable, "-B", "-c", script, str(self.root), m["process_id"]],
                            cwd=Path(__file__).resolve().parent).returncode
        self.assertEqual(rc, 37)
        report = archive.inspect(m["process_id"])
        self.assertEqual(report["execution_status"], "incomplete")
        self.assertEqual(len(report["records"]), 1)
        self.assertFalse(next(iter(report["units"].values()))["complete"])
        self.assertIsNone(report["terminal"])

    def test_torn_journal_not_success(self):
        m = self.reserve()
        attempt = archive.directory(m["process_id"]) / "attempt"
        attempt.mkdir()
        archive.write_new(attempt / "start.json", {**archive.identity(m),
                          "manifest_sha256": archive.digest(m)})
        (attempt / "batches.jsonl").write_bytes(b'{"partial"')
        archive.write_new(attempt / "terminal.json", {**archive.identity(m),
                          "execution_status": "success", "record_count": 0})
        report = archive.inspect(m["process_id"])
        self.assertEqual(report["execution_status"], "incomplete")
        self.assertTrue(report["issues"])

    def test_cli_isolated_from_historical_tree(self):
        project = self.work / "stand-in-project"
        old = project / "WhaleGPU-Phase34" / "runs"
        old.mkdir(parents=True)
        history = old / "samples.csv"
        history.write_bytes(b"id,round\nold,0\n")
        program = project / "paired-runner"
        program.mkdir()
        for name in ("run_paired.py", "archive.py", "plan.py"):
            shutil.copyfile(Path(__file__).resolve().with_name(name), program / name)
        def cli(*args):
            return subprocess.check_output([sys.executable, "-B", str(program / "run_paired.py"),
                                            *args], cwd=old, text=True).strip()
        pid = cli("plan", "--units", "2", "--seed", "3")
        self.assertEqual(json.loads(cli("dry-run", pid))["execution_status"], "success")
        result = json.loads(cli("inspect", pid))
        self.assertEqual(len(result["records"]), 8)
        self.assertTrue(all(unit["complete"] for unit in result["units"].values()))
        self.assertEqual(list(old.iterdir()), [history])
        self.assertEqual(history.read_bytes(), b"id,round\nold,0\n")


if __name__ == "__main__":
    unittest.main()
