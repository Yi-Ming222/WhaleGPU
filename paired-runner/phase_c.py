"""Multi-unit measurement: python -B phase_c.py --units 8 --seed 431."""
import argparse
import csv
import json
import math
import statistics as st
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
import run_cuda as c


def sample():
    start = time.time_ns()//1000
    row = dict(start_utc_us=start, temperature_c=None, graphics_mhz=None, power_w=None)
    try:
        output = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-gpu=name,temperature.gpu,clocks.gr,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
            timeout=5).stdout
        name, *values = next(csv.reader([output.strip()], skipinitialspace=True))
        if name != "NVIDIA GeForce RTX 5070 Laptop GPU" or len(values)!=3:
            raise ValueError("GPU identity or field count mismatch")
        for key,value in zip(("temperature_c","graphics_mhz","power_w"), values):
            if value not in ("N/A","[N/A]","[Not Supported]"):
                number = float(value)
                if math.isfinite(number):
                    row[key] = number
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        row["error"] = str(exc)
    row["end_utc_us"] = time.time_ns()//1000
    return row


def fences(values):
    if len(values)<4:
        return -math.inf, math.inf
    q1,_,q3 = st.quantiles(values, n=4, method="inclusive")
    return q1-1.5*(q3-q1), q3+1.5*(q3-q1)


def report(path):
    manifest = c.archive.read_json(path/"manifest.json")
    attempt = path/"attempt"
    end = attempt/"terminal.json"
    terminal = c.archive.read_json(end) if end.exists() else {"status":"incomplete"}
    events = []
    journal = attempt/"events.jsonl"
    if journal.exists():
        for line in journal.read_bytes().splitlines():
            try:
                events.append(json.loads(line)["event"])
            except (KeyError,ValueError):
                break
    good = terminal["status"]=="success" and c.sha(journal)==terminal["events_sha256"]
    batches = {e["index"]:e for e in events if e["type"]=="batch"}
    times = {e["index"]:e for e in events if e["type"]=="unit"}
    pairs = []
    for i in range(0,len(manifest["plan"]),2):
        spec = manifest["plan"][i]
        valid = good and i in batches and i+1 in batches and i//4 in times
        pair = {"unit":i//4+1,"block_id":spec["block_id"],"order":spec["order"],"valid":valid}
        if valid:
            for j in (i,i+1):
                e = batches[j]
                label = "A" if e["kernel_id"]==c.KERNELS["A"] else "B"
                pair[label] = 1000*e["event_total_ms"]/e["completed"]
            pair["difference_us"] = pair["A"]-pair["B"]
            pair["ratio"] = pair["A"]/pair["B"]
        pairs.append(pair)
    valid = [p for p in pairs if p["valid"]]
    for key in ("A","B","difference_us"):
        lo,hi = fences([p[key] for p in valid])
        for p in valid:
            p.setdefault("outliers",[])
            if not lo<=p[key]<=hi:
                p["outliers"].append(key)
    telemetry = [json.loads(l) for l in (path/"telemetry.jsonl").read_text().splitlines()]
    lines = ["# Phase C 配对测量", "", f"运行 {path.name}，状态 {terminal['status']}。"
             f"有效配对 {len(valid)}/{len(pairs)}。每侧为 10 次 launch 的 Event 均摊时间。",
             "", "|Unit|顺序|A (µs)|B (µs)|A−B (µs)|A/B|异常标记|","|---:|---|---:|---:|---:|---:|---|"]
    for p in pairs:
        prefix = f"|{p['unit']}|{p['order']}|"
        lines.append(prefix+(f"{p['A']:.3f}|{p['B']:.3f}|{p['difference_us']:+.3f}|"
                             f"{p['ratio']:.4f}|{', '.join(p['outliers']) or '—'}|"
                             if p["valid"] else "—|—|—|—|无效|"))
    if valid:
        lines += ["", f"A/B 中位时间分别为 {st.median(p['A'] for p in valid):.3f} / "
                  f"{st.median(p['B'] for p in valid):.3f} µs。配对差均值 "
                  f"{st.mean(p['difference_us'] for p in valid):+.3f} µs。"]
    lines += ["","|Unit|开始 UTC（Unix µs）|实际墙钟 µs|遥测查询窗口重叠编号（从 1 开始）|",
              "|---:|---:|---:|---|"]
    for u in range(manifest["unit_count"]):
        if u not in times:
            lines.append(f"|{u+1}|—|—|Unknown|")
            continue
        t = times[u]
        start = t["start_utc_us"]
        overlap = [str(j) for j,v in enumerate(telemetry,1)
                   if v["start_utc_us"]<=start+t["wall_us"] and v["end_utc_us"]>=start]
        lines.append(f"|{u+1}|{start}|{t['wall_us']:.3f}|{', '.join(overlap) or 'Unknown'}|")
    lines += ["","Unit 墙钟由执行器单调时钟测量，含同步和回执输出。查询窗口重叠仅供定位；"
              "温度、Graphics Clock、功耗的原始读数见 telemetry.jsonl，不作插值。"
              "异常按各配置时间及配对差分别应用 1.5×IQR 围栏，全部保留。单进程结果仅作描述。"]
    c.archive.write_new(path/"pairs.json",pairs)
    with (path/"REPORT.md").open("x",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")


def measure(run_id):
    path = c.run_path(run_id)
    if c.archive.read_json(path/"manifest.json").get("purpose")!="multi_unit_comparison":
        raise ValueError("Phase C requires a multi-unit plan")
    stop = threading.Event()
    with (path/"telemetry.jsonl").open("x",encoding="utf-8",newline="\n") as log:
        def collect():
            while not stop.is_set():
                c.archive.append(log,sample())
                stop.wait(0.1)
        c.archive.append(log,sample())
        thread = threading.Thread(target=collect)
        thread.start()
        try:
            c.verify(run_id)
        finally:
            stop.set()
            thread.join()
            c.archive.append(log,sample())
            report(path)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units",type=int,default=8)
    parser.add_argument("--seed",type=int,default=431)
    args = parser.parse_args()
    measure(c.build(args.seed,args.units))
