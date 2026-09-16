"""One config per PROCESS, because ru_maxrss is a process-wide high-water mark
and a sweep in one process reports the max of everything that ran before it.
"""
import json, sys, time, pathlib, resource, argparse
import torch
R = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "ops"))
from bench_device import load_model, tiles, forward, CKPT, FRAMES

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--amp", default="off")
a = ap.parse_args()

base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
m = load_model(str(CKPT), torch.device(a.device), num_frames=FRAMES)
after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
x = tiles(a.batch, seed=0)
t0 = time.time()
y = forward(m, x, a.device, None if a.amp == "off" else a.amp)
dt = time.time() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
row = {"device": a.device, "batch": a.batch, "amp": a.amp,
       "rss_start_gb": round(base, 2), "rss_after_load_gb": round(after_load, 2),
       "rss_peak_gb": round(peak, 2),
       "forward_rss_delta_gb": round(peak - after_load, 2),
       "seconds": round(dt, 3), "out_mean": round(float(y.mean()), 6)}
if a.device == "mps":
    row["mps_driver_allocated_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
    row["mps_current_allocated_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 2)
print(json.dumps(row))
