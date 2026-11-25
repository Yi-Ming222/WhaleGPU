"""Exclusive, local Phase A archives. The plan stays immutable after reservation."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import stat

from plan import canonical, check_id, digest, now, validate_manifest


ROOT = Path(__file__).resolve().parent / "artifacts"


def checked(path):
    # Windows junctions are reparse points even when is_symlink() is false.
    if path.is_symlink() or (path.exists() and
            getattr(path.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError(f"archive links/reparse points are forbidden: {path.name}")
    return path


def directory(process_id):
    return checked(checked(ROOT) / check_id(process_id))


def write_new(path, obj):
    with checked(path).open("x", encoding="utf-8", newline="\n") as f:
        append(f, obj)


def append(f, obj):
    line = canonical(obj) + "\n"
    f.write(line)
    f.flush()
    os.fsync(f.fileno())
    return line.encode("utf-8")


def read_json(path):
    return json.loads(checked(path).read_text(encoding="utf-8"))


def reserve(m):
    validate_manifest(m)
    p = directory(m["process_id"])
    ROOT.mkdir(exist_ok=True)
    p.mkdir()  # A collision, including a partial previous reservation, is fatal.
    write_new(p / "manifest.json", m)
    return p


def load(process_id):
    m = read_json(directory(process_id) / "manifest.json")
    validate_manifest(m)
    if m["process_id"] != process_id:
        raise ValueError("directory/manifest process_id mismatch")
    return m


def identity(m):
    return {k: m[k] for k in ("experiment_id", "session_id", "process_id")}


def receipt(row, index, pid):
    return {**row, "actual_sequence": index, "actual_launch_count": 0,
            "mode": "infrastructure_only", "os_pid": pid, "timestamp": now()}


def validate_receipt(m, item, index):
    if not isinstance(item, dict):
        raise ValueError("batch receipt must be an object")
    if index >= len(m["experiment_plan"]):
        raise ValueError("extra batch receipt")
    if type(item.get("os_pid")) is not int or item["os_pid"] <= 0:
        raise ValueError("invalid worker pid")
    if datetime.fromisoformat(item["timestamp"]).utcoffset() is None:
        raise ValueError("receipt timestamp needs an offset")
    expected = receipt(m["experiment_plan"][index], index, item["os_pid"])
    expected["timestamp"] = item["timestamp"]
    if canonical(expected) != canonical(item):
        raise ValueError("invalid receipt or actual order differs from plan")


def execute(process_id, observations):
    """Consume a worker's receipts. An exclusive attempt may never be resumed."""
    m = load(process_id)
    attempt = checked(directory(process_id) / "attempt")
    attempt.mkdir()
    context = identity(m)
    count = 0
    journal_hash = hashlib.sha256()
    try:
        write_new(attempt / "start.json", {**context, "timestamp": now(),
                  "controller_pid": os.getpid(), "manifest_sha256": digest(m)})
        with checked(attempt / "batches.jsonl").open("x", encoding="utf-8", newline="\n") as f:
            for item in observations(m):
                validate_receipt(m, item, count)
                journal_hash.update(append(f, {**context, **item}))
                count += 1
        if count != len(m["experiment_plan"]):
            raise ValueError("worker finished before all planned batches")
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        plan = m["experiment_plan"]
        write_new(attempt / "terminal.json", {**context, "execution_status": status,
                  "timestamp": now(), "error_type": type(exc).__name__, "message": str(exc),
                  "next_expected": plan[count] if count < len(plan) else None, "record_count": count,
                  "journal_sha256": journal_hash.hexdigest()})
        raise
    write_new(attempt / "terminal.json", {**context, "execution_status": "success",
              "timestamp": now(), "record_count": count, "journal_sha256": journal_hash.hexdigest()})


def inspect(process_id):
    """Read-only reconciliation. A missing terminal never implies success."""
    p = directory(process_id)
    if not p.is_dir():
        raise FileNotFoundError(p)
    try:
        m = load(process_id)
    except (OSError, ValueError, TypeError) as exc:
        return {"process_id": process_id, "execution_status": "incomplete",
                "issues": [f"unreadable reservation: {exc}"], "units": {},
                "blocks": {}, "records": [], "terminal": None, "eligible_measurement_units": []}
    context = identity(m)
    attempt = checked(directory(process_id) / "attempt")
    rows = []
    issues = []
    terminal = None
    status = "planned"
    journal_hash = hashlib.sha256()
    if attempt.exists():
        status = "incomplete"
        try:
            start = read_json(attempt / "start.json")
            if (start["manifest_sha256"] != digest(m)
                    or any(start[k] != v for k, v in context.items())):
                raise ValueError("manifest digest differs from claimed plan")
            with checked(attempt / "batches.jsonl").open("rb") as f:
                for line in f:
                    if not line.endswith(b"\n"):
                        raise ValueError("partial final journal record")
                    record = json.loads(line)
                    item = dict(record)
                    if any(item.pop(k) != v for k, v in context.items()):
                        raise ValueError("record identity mismatch")
                    validate_receipt(m, item, len(rows))
                    rows.append(record)
                    journal_hash.update(line)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            issues.append(str(exc))
        end = checked(attempt / "terminal.json")
        if end.exists():
            try:
                terminal = read_json(end)
                state = terminal["execution_status"]
                if (any(terminal[k] != v for k, v in context.items())
                        or terminal["record_count"] != len(rows)
                        or terminal["journal_sha256"] != journal_hash.hexdigest()
                        or state not in ("success", "failed", "interrupted")):
                    raise ValueError("invalid terminal record")
                if state == "success" and (issues or len(rows) != len(m["experiment_plan"])):
                    raise ValueError("success contradicts journal")
                status = state
            except (OSError, KeyError, ValueError, TypeError) as exc:
                issues.append(str(exc))
    seen = {row["batch_id"] for row in rows}
    units = {}
    blocks = {}
    for planned in m["experiment_plan"]:
        unit = units.setdefault(planned["unit_id"], {"missing_batch_ids": []})
        block = blocks.setdefault(planned["block_id"], {"order": planned["order"],
                                  "unit_id": planned["unit_id"], "missing_batch_ids": []})
        if planned["batch_id"] not in seen:
            unit["missing_batch_ids"].append(planned["batch_id"])
            block["missing_batch_ids"].append(planned["batch_id"])
    for group in (*units.values(), *blocks.values()):
        group["complete"] = not group["missing_batch_ids"]
    return {**context, "execution_status": status, "terminal": terminal, "issues": issues,
            "units": units, "blocks": blocks, "records": rows, "eligible_measurement_units": []}
