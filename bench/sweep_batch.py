"""Batch sweep: what does 64 GB of unified memory buy that a 12 GB card cannot?

Same fixed tiles, same weights, varying batch. Records peak RSS so the memory
claim is measured rather than asserted.
"""
import json, sys, time, pathlib, resource
import torch
R = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "ops"))
from bench_device import load_model, tiles, timed, CKPT, FRAMES, TILE

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes

out = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "frames": FRAMES, "tile": TILE,
       "rows": []}
plan = [("mps", b, "mps", 3) for b in (1, 2, 4, 8, 16)] + \
       [("mps", b, None, 3) for b in (1, 8)] + \
       [("cpu", b, None, 2) for b in (1, 2)]
loaded = {}
for dev, b, amp, reps in plan:
    if dev == "mps" and not torch.backends.mps.is_available():
        continue
    if dev not in loaded:
        loaded[dev] = load_model(str(CKPT), torch.device(dev), num_frames=FRAMES)
    m = loaded[dev]
    x = tiles(b, seed=0)
    try:
        dt, y = timed(m, x, dev, amp, reps)
        row = {"device": dev, "batch": b, "autocast": amp or "off",
               "seconds_per_batch": round(dt, 4),
               "tiles_per_second": round(b / dt, 3),
               "peak_rss_gb": round(rss_gb(), 2)}
        if dev == "mps":
            row["mps_allocated_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 2)
    except Exception as e:
        row = {"device": dev, "batch": b, "autocast": amp or "off",
               "error": f"{type(e).__name__}: {e}"[:200], "peak_rss_gb": round(rss_gb(), 2)}
    out["rows"].append(row)
    print(f"  {row.get('device'):4s} b={row.get('batch'):<3d} amp={row.get('autocast'):4s} "
          f"{row.get('tiles_per_second', row.get('error'))} tiles/s  rss={row.get('peak_rss_gb')}GB")

cpu1 = next((r for r in out["rows"] if r["device"] == "cpu" and r["batch"] == 1
             and "tiles_per_second" in r), None)
if cpu1:
    best = max((r for r in out["rows"] if r["device"] == "mps" and "tiles_per_second" in r),
               key=lambda r: r["tiles_per_second"])
    out["headline"] = {
        "cpu_b1_tiles_per_second": cpu1["tiles_per_second"],
        "best_mps": {k: best[k] for k in ("batch", "autocast", "tiles_per_second")},
        "speedup": round(best["tiles_per_second"] / cpu1["tiles_per_second"], 2)}
(R / "results" / "sweep_batch.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out.get("headline", {}), indent=2))
