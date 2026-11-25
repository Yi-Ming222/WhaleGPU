"""Plan, exercise and inspect the infrastructure. No CUDA executor is registered."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

import archive
from plan import canonical, create_manifest, validate_manifest


def worker():
    manifest = json.load(sys.stdin)
    validate_manifest(manifest)
    for index, planned in enumerate(manifest["experiment_plan"]):
        print(canonical(archive.receipt(planned, index, os.getpid())), flush=True)


def observations(manifest):
    # No shell, experiment-supplied command, or executable discovery.
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "_worker"]
    child = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             text=True, encoding="utf-8")
    try:
        child.stdin.write(canonical(manifest))
        child.stdin.close()
        for line in child.stdout:
            item = json.loads(line)
            if item["os_pid"] != child.pid:
                raise ValueError("receipt belongs to another worker")
            yield item
        if child.wait() != 0:
            raise subprocess.CalledProcessError(child.returncode, cmd)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        child.stdin.close()
        child.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="reserve and seal a new process plan")
    plan.add_argument("--units", type=int, required=True)
    plan.add_argument("--seed", type=int, required=True)
    plan.add_argument("--session-id")
    plan.add_argument("--experiment-id")
    for name in ("dry-run", "inspect"):
        commands.add_parser(name).add_argument("process_id")
    commands.add_parser("_worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == "_worker":
        worker()
    elif args.command == "plan":
        m = create_manifest(args.units, args.seed, args.session_id, args.experiment_id)
        archive.reserve(m)
        print(m["process_id"])
    elif args.command == "dry-run":
        archive.execute(args.process_id, observations)
        print(canonical({"process_id": args.process_id, "mode": "infrastructure_only",
                         "execution_status": "success"}))
    else:
        print(canonical(archive.inspect(args.process_id)))


if __name__ == "__main__":
    main()
