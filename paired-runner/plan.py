"""Version 1 infrastructure plans. Only Python's standard library is used."""

import hashlib
import json
import random
import uuid
from datetime import datetime, timezone


KERNELS = {"A": "T32_R4_row", "B": "T64_R4_row"}
INPUT = {"m": 513, "n": 769, "k": 257, "dtype": "FP32",
         "layout": "row_major", "pattern": "random", "seed": 20261117}
MEASUREMENT = {"mode": "infrastructure_only", "planned_launch_count": 10,
               "latency_unit": "us/GEMM", "timing_enabled": False}
WARMUP = {"enabled": False, "actual_launch_count": 0}
ALGORITHM = "python-random-mt19937-shuffle-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def check_id(value):
    if not isinstance(value, str) or uuid.UUID(hex=value).hex != value:
        raise ValueError("expected a lowercase 32-character UUID")
    return value


def generate_plan(units, seed, process_id=None):
    if type(units) is not int or units < 1 or type(seed) is not int:
        raise ValueError("units must be a positive integer and seed an integer")
    prefix = "" if process_id is None else check_id(process_id) + "-"
    rng = random.Random(seed)
    starts = ["AB", "BA"] * (units // 2)
    if units % 2:
        starts.append(rng.choice(("AB", "BA")))
    rng.shuffle(starts)
    batches = []
    for unit_index, first in enumerate(starts, 1):
        unit_id = f"{prefix}u{unit_index}"
        for block_index, order in enumerate((first, first[::-1]), 1):
            block_id = f"{unit_id}-b{block_index}"
            for position, kernel in enumerate(order, 1):
                batches.append({"unit_id": unit_id, "block_id": block_id,
                               "batch_id": f"{block_id}-p{position}", "order": order,
                               "position": position, "kernel_id": KERNELS[kernel]})
    return batches


def _manifest(units, seed, session_id, experiment_id, process_id, timestamp):
    plan = generate_plan(units, seed, process_id)
    manifest = {
        "schema_version": 1, "experiment_id": check_id(experiment_id),
        "session_id": check_id(session_id), "process_id": check_id(process_id),
        "creation_timestamp": timestamp,
        "random_seed": seed, "unit_count": units, "random_algorithm": ALGORITHM,
        "experiment_plan": plan, "plan_sha256": digest(plan),
        "kernel_configuration": dict(KERNELS), "input_spec": dict(INPUT),
        "input_hash": digest(INPUT), "input_hash_scope": "canonical_input_spec",
        "matrix_hashes": None, "executable_hash": None,
        "warmup_config": dict(WARMUP), "measurement_config": dict(MEASUREMENT),
        "execution_status": "planned",
    }
    return manifest


def create_manifest(units, seed, session_id=None, experiment_id=None):
    return _manifest(units, seed,
                     uuid.uuid4().hex if session_id is None else session_id,
                     uuid.uuid4().hex if experiment_id is None else experiment_id,
                     uuid.uuid4().hex, now())


def validate_manifest(m):
    if not isinstance(m, dict):
        raise ValueError("manifest must be an object")
    try:
        if datetime.fromisoformat(m["creation_timestamp"]).utcoffset() is None:
            raise ValueError("timestamp needs an offset")
        expected = _manifest(m["unit_count"], m["random_seed"], m["session_id"],
                             m["experiment_id"], m["process_id"], m["creation_timestamp"])
        if canonical(expected) != canonical(m):
            raise ValueError("invalid manifest schema, configuration, or randomization plan")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"invalid manifest: {exc}") from exc
