"""Benchmark to villa AGENTS.md rule 1.4: baseline, before/after, and
min/median/max rather than a bare mean. Real checkpoint, fixed tiles.
"""
import json, sys, time, pathlib, statistics as st
import torch
R = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "ops"))
from bench_device import load_model, tiles, forward, CKPT, FRAMES, TILE

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 9
BATCH = 1
out = {"reps": REPS, "batch": BATCH, "frames": FRAMES, "tile": TILE,
       "torch": torch.__version__, "ckpt": CKPT.name, "rows": []}
x = tiles(BATCH, seed=0)

plan = [("mps", None, "MPS, autocast off"),
        ("mps", "cpu", "MPS, villa's current amp_device (a no-op)"),
        ("mps", "mps", "MPS, amp_device corrected"),
        ("cpu", None, "CPU, autocast off (today's Apple Silicon path, fp32)")]
for dev, amp, label in plan:
    if dev == "mps" and not torch.backends.mps.is_available():
        continue
    m = load_model(str(CKPT), torch.device(dev), num_frames=FRAMES)
    forward(m, x, dev, amp)                      # warmup, not timed
    ts = []
    for _ in range(REPS if dev == "mps" else max(3, REPS // 3)):
        t0 = time.time(); forward(m, x, dev, amp); ts.append(time.time() - t0)
    row = {"device": dev, "amp_device": amp or "off", "label": label,
           "n": len(ts), "min_s": round(min(ts), 4),
           "median_s": round(st.median(ts), 4), "max_s": round(max(ts), 4),
           "mean_s": round(st.fmean(ts), 4),
           "tiles_per_second_median": round(BATCH / st.median(ts), 3)}
    out["rows"].append(row)
    print(f"  {label:48s} n={row['n']:2d} min {row['min_s']:.3f} "
          f"median {row['median_s']:.3f} max {row['max_s']:.3f} s/tile", flush=True)
    del m
    if dev == "mps":
        torch.mps.empty_cache()

base = next(r for r in out["rows"] if r["device"] == "cpu")
for r in out["rows"]:
    r["speedup_vs_cpu_median"] = round(base["median_s"] / r["median_s"], 2)
out["headline"] = {
    "cpu_median_s": base["median_s"],
    "best_median_s": min(r["median_s"] for r in out["rows"]),
    "speedup": round(base["median_s"] / min(r["median_s"] for r in out["rows"]), 2)}
(R / "results" / "bench_stats.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["headline"]), flush=True)
