"""WG-EXP-003: build a new executor, then verify exactly one sealed unit."""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

sys.dont_write_bytecode = True
import archive
from plan import INPUT, KERNELS, canonical, check_id, digest, generate_plan, now

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = HERE / "cuda-artifacts"
CUDA = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")
VCVARS = Path(r"C:\BuildTools2022\VC\Auxiliary\Build\vcvars64.bat")
COMMON = PROJECT / "WhaleGPU-Phase1/include"
P3 = PROJECT / "WhaleGPU-Phase1/phase3/src"
DISPATCH = PROJECT / "WhaleGPU-Phase34/runs/20261118-092322-b6e99f/source"
SOURCE = HERE / "cuda_executor.cu"
EXE = "paired_cuda.exe"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def inputs():
    state = INPUT["seed"]
    hashes = {}
    for key, size in (("a_sha256", INPUT["m"]*INPUT["k"]), ("b_sha256", INPUT["k"]*INPUT["n"])):
        values = array("f")
        for _ in range(size):
            state = (1664525*state + 1013904223) & 0xffffffff
            values.append(((state >> 8)-8388608)/8388608)
        hashes[key] = hashlib.sha256(values.tobytes()).hexdigest()
    return hashes


def run_path(run_id):
    return archive.checked(archive.checked(ROOT) / check_id(run_id))


def reserve(run_id):
    path = run_path(run_id)
    ROOT.mkdir(exist_ok=True)
    path.mkdir()
    return path


def tool_environment(work):
    cmd = ["cmd", "/d", "/c", f'call "{VCVARS}" -vcvars_ver=14.39 >nul && set']
    # Obtain vcvars' child environment without persisting private environment values.
    result = subprocess.run(subprocess.list2cmdline(cmd[:3])+" "+cmd[3],
                            capture_output=True, check=True)
    env = dict(os.environ)
    for line in result.stdout.decode("mbcs").splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            env[key.upper()] = value
    env["PATH"] = os.pathsep.join((str(CUDA/"bin"), str(CUDA/"bin/x64"), env["PATH"]))
    # NVCC 13's host probe embeds TMP without quoting; cwd is the exclusive build dir.
    env["TEMP"] = env["TMP"] = "."
    return env


def resolved_headers(work):
    # NVCC's actual compilation intermediates carry MSVC/preprocessor line markers.
    files = set()
    marker = re.compile(rb'^\s*#\s*(?:line\s+)?\d+\s+"([^"]+)"')
    for pp in work.glob("*.ii"):
        with pp.open("rb") as f:
            for line in f:
                found = marker.match(line)
                if not found:
                    continue
                raw = found[1].replace(b"\\\\", b"\\")
                for enc in ("utf-8", "mbcs"):
                    try:
                        p = (work / raw.decode(enc)).resolve()
                        if p.is_file() and p != SOURCE and work not in p.parents:
                            files.add(p)
                            break
                    except (UnicodeError, OSError, ValueError):
                        continue
    required = {PROJECT/"WhaleGPU-Phase34/register.cuh", DISPATCH/"dispatch.inc",
                P3/"gemm.cuh", P3/"reference.hpp", COMMON/"common.cuh", CUDA/"include/cuda_runtime.h"}
    missing = required - files
    if missing:
        raise ValueError(f"missing compiled include evidence: {sorted(map(str,missing))}")
    return {str(p): sha(p) for p in sorted(files)}


def compile_executor(work):
    env = tool_environment(work)
    nvcc = str(CUDA/"bin/nvcc.exe")
    args = [nvcc, "-arch=sm_120", "-std=c++17", "-O3", "-lineinfo", "-Xcompiler", "/openmp",
            "-I"+os.path.relpath(COMMON,work), "-I"+os.path.relpath(DISPATCH,work),
            "-Xptxas=-v", "--keep", os.path.relpath(SOURCE,work), "bcrypt.lib", "-o", EXE]
    archive.write_new(work/"command.json", {"argv":args,"cwd":str(work),"vcvars":str(VCVARS),
                                            "vcvars_version":"14.39","source_sha256":sha(SOURCE)})
    with (work/"compiler.log").open("xb") as log:
        cl = shutil.which("cl.exe", path=env["PATH"])
        if not cl:
            raise FileNotFoundError("vcvars did not expose cl.exe")
        for cmd in ([nvcc, "--version"], [cl, "/Bv", "/?"], args):
            subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {"headers":resolved_headers(work), "exe_sha256":sha(work/EXE)}


def build(seed, units=None):
    if units is not None and (type(units) is not int or not 2 <= units <= 4096):
        raise ValueError("Phase C accepts 2..4096 units per process")
    run_id = uuid.uuid4().hex
    path = reserve(run_id)
    print(f"archive: {path}", flush=True)
    work = path/"build"
    try:
        work.mkdir()
        plan = generate_plan(units or 1,seed,run_id)
        manifest = {"schema_version":"cuda-link-1", "purpose":"one_unit_link_verification",
                    "process_id":run_id,"session_id":uuid.uuid4().hex,"experiment_id":"WG-EXP-003",
                    "created":now(),"random_seed":seed,"plan":plan,"plan_sha256":digest(plan),
                    "input":INPUT,"matrix_hashes":inputs(),"planned_per_batch":10,
                    "validation_per_phase":2,"warmup_count":0,"stream":"legacy default, one thread"}
        if units is not None:
            manifest.update(schema_version="cuda-multi-1",experiment_id="Phase-C",
                            purpose="multi_unit_comparison",unit_count=units)
        archive.write_new(path/"manifest.json",manifest)
        evidence = compile_executor(work)
        archive.write_new(path/"build.json", {"status":"success","manifest_sha256":digest(manifest),
                                              **evidence})
    except BaseException as exc:
        archive.write_new(path/"build-error.json",{"error":str(exc),"type":type(exc).__name__,"at":now()})
        raise
    return run_id


def expected_events(m):
    yield {"type":"ready"}
    for kernel in KERNELS.values():
        yield {"type":"check","phase":"pre","kernel_id":kernel}
    for i,row in enumerate(m["plan"]):
        for kind in ("begin","batch"):
            yield {"type":kind,"index":i,"kernel_id":row["kernel_id"],"planned":10}
        if m.get("purpose")=="multi_unit_comparison" and i%4==3:
            yield {"type":"unit","index":i//4}
    for kernel in KERNELS.values():
        yield {"type":"check","phase":"post","kernel_id":kernel}


def validate_event(event, expected, hashes, pid):
    if not isinstance(event,dict) or any(canonical(event.get(k)) != canonical(v)
                                         for k,v in expected.items()):
        raise ValueError(f"unexpected event; expected {expected}")
    kind = event["type"]
    if kind == "ready":
        if (event["os_pid"] != pid or event["input_roundtrip"] is not True or
                any(event[k] != v for k,v in hashes.items()) or
                event["device"] != "NVIDIA GeForce RTX 5070 Laptop GPU"):
            raise ValueError("device, process or input identity mismatch")
    elif kind == "check":
        if (event["poison_byte"] != 255 or event["elements"] != 513*769 or
                event["mismatches"] != 0 or not math.isfinite(event["max_abs_error"]) or
                not 0 <= event["max_error_ratio"] <= 1):
            raise ValueError("correctness failure")
    elif kind == "unit":
        if type(event["start_utc_us"]) is not int or not math.isfinite(event["wall_us"]) or event["wall_us"]<=0:
            raise ValueError("invalid unit clock")
    elif kind == "batch":
        if (event["submitted"] != 10 or event["completed"] != 10 or event["stream"] != 0 or
                event["stop_synchronized"] is not True or
                not math.isfinite(event["event_total_ms"]) or event["event_total_ms"] <= 0):
            raise ValueError("invalid count, stream, synchronization or timing")


def verify(run_id):
    path = run_path(run_id)
    attempt = archive.checked(path/"attempt")
    attempt.mkdir()
    events = []
    rc = None
    pid = None
    try:
        m = archive.read_json(path/"manifest.json")
        built = archive.read_json(path/"build.json")
        exe = archive.checked(archive.checked(path/"build")/EXE)
        if (built["status"] != "success" or digest(m) != built["manifest_sha256"] or
                m["process_id"] != run_id or m["plan"] != generate_plan(m.get("unit_count",1),m["random_seed"],run_id)
                or digest(m["plan"]) != m["plan_sha256"] or sha(exe) != built["exe_sha256"]):
            raise ValueError("sealed manifest, plan or executable mismatch")
        order = "".join("A" if row["kernel_id"] == KERNELS["A"] else "B" for row in m["plan"])
        specs = list(expected_events(m))
        env = dict(os.environ, OMP_NUM_THREADS="8", CUDA_CACHE_DISABLE="1")
        with (attempt/"events.jsonl").open("x",encoding="utf-8",newline="\n") as log, (
                attempt/"stderr.log").open("xb") as err:
            multi = m.get("purpose")=="multi_unit_comparison"
            child = subprocess.Popen([str(exe),order]+(["--multi"] if multi else []),cwd=attempt,env=env,
                                     stdout=subprocess.PIPE,stderr=err,text=True,encoding="utf-8")
            pid = child.pid
            try:
                for line in child.stdout:
                    event = json.loads(line)
                    record = {"at":now(),"process_id":run_id,"os_pid":pid,"event":event}
                    index = event.get("index")
                    if event.get("type")=="unit" and type(index) is int and 0<=index<len(m["plan"])//4:
                        record["unit_id"] = m["plan"][4*index]["unit_id"]
                    elif type(index) is int and 0 <= index < len(m["plan"]):
                        record["planned_batch"] = m["plan"][index]
                    archive.append(log,record)
                    if len(events) >= len(specs):
                        raise ValueError("extra executor event")
                    validate_event(event,specs[len(events)],m["matrix_hashes"],pid)
                    events.append(event)
                rc = child.wait()
            finally:
                if child.poll() is None:
                    child.kill()
                rc = child.wait()
                child.stdout.close()
        if rc != 0 or len(events) != len(specs):
            raise ValueError(f"worker exit {rc}, validated {len(events)}/{len(specs)} events")
        terminal = {"status":"success","unit_complete":True,"exe_sha256":built["exe_sha256"],
                    "events_sha256":sha(attempt/"events.jsonl")}
    except BaseException as exc:
        archive.write_new(attempt/"terminal.json",{"status":"interrupted" if isinstance(exc,KeyboardInterrupt)
                          else "failed","unit_complete":False,"error_type":type(exc).__name__,
                          "error":str(exc),"validated_events":len(events),"os_pid":pid,
                          "exit_code":rc,"at":now(),"unacknowledged_completion":"unknown"})
        raise
    archive.write_new(attempt/"terminal.json",{**terminal,"at":now(),"os_pid":pid,"exit_code":rc})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="cmd",required=True)
    subs.add_parser("build").add_argument("--seed",type=int,default=431)
    subs.add_parser("verify").add_argument("process_id")
    args = parser.parse_args()
    if args.cmd == "build":
        print(build(args.seed))
    else:
        verify(args.process_id)
