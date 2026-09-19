"""Collect mem_probe rows in FRESH processes and persist them.

The memory figures in the README were measured once and typed. Nothing wrote
them to disk, so verify_claims.py's mem_probe_summary.json line could never
fire. This regenerates them instead.

ru_maxrss is a process-wide high-water mark, so every configuration must run in
its own process; a single-process sweep reports the maximum of everything that
ran before it.

Note the two columns are different quantities and the README must say so:
  CPU  forward_rss_delta_gb     host RSS the forward pass adds
  MPS  mps_driver_allocated_gb  memory the Metal driver holds for it
"""
import json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable
ROWS = [("cpu", 1, "off"), ("cpu", 2, "off"), ("mps", 1, "mps"), ("mps", 4, "mps")]

out = []
for dev, batch, amp in ROWS:
    p = subprocess.run([PY, str(HERE/"mem_probe.py"), "--device", dev,
                        "--batch", str(batch), "--amp", amp],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"mem_probe failed for {dev}/{batch}/{amp}:\n{p.stderr[-800:]}")
    row = json.loads(p.stdout.strip().splitlines()[-1])
    row["forward_memory_gb"] = (row["mps_driver_allocated_gb"] if dev == "mps"
                                else row["forward_rss_delta_gb"])
    row["forward_memory_is"] = ("mps_driver_allocated_gb" if dev == "mps"
                                else "forward_rss_delta_gb")
    out.append(row)
    print(json.dumps(row))

import torch
cpu1 = next(r for r in out if r["device"] == "cpu" and r["batch"] == 1)
mps1 = next(r for r in out if r["device"] == "mps" and r["batch"] == 1)
rec = {"torch": torch.__version__, "python": sys.version.split()[0],
       "fresh_process_per_row": True, "rows": out,
       "leaner_x_at_batch1": round(cpu1["forward_memory_gb"]/mps1["forward_memory_gb"], 2)}
R = HERE.parent
(R/"results"/"mem_probe_summary.json").write_text(json.dumps(rec, indent=2)+"\n")
print("leaner at batch 1:", rec["leaner_x_at_batch1"], "x")
