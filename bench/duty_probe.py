"""Find the highest MPS duty cycle that keeps the fans in the idle band.

He is asleep. The recorded rule for this machine is that placement, not
temperature, drives the fan curve, and that a measurement beats an assumption.
So: run real forwards at several duty cycles and read the fan each time.
"""
import json, subprocess, sys, time, pathlib, shutil
import torch
R = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R/"ops"))
from bench_device import load_model, tiles, forward, CKPT, FRAMES

ISTATS = shutil.which("istats") or "/usr/local/bin/istats"
def fan():
    try:
        out = subprocess.run([ISTATS, "fan"], capture_output=True, text=True, timeout=15).stdout
        vals = [int(w) for l in out.splitlines() if "RPM" in l
                for w in l.replace("RPM", " ").split() if w.isdigit()]
        return max(vals) if vals else None
    except Exception:
        return None

m = load_model(str(CKPT), torch.device("mps"), num_frames=FRAMES)
x = tiles(1, seed=0)
forward(m, x, "mps", "mps")                      # warm

res = {"idle_fan_before": fan(), "rows": []}
print("idle fan before:", res["idle_fan_before"], flush=True)
SECONDS = 75                                     # long enough for the fan curve to respond
for duty in (1.0, 0.5, 0.33, 0.25):
    t_end = time.time() + SECONDS
    n = 0
    while time.time() < t_end:
        t0 = time.time()
        forward(m, x, "mps", "mps")
        busy = time.time() - t0
        n += 1
        if duty < 1.0:
            time.sleep(busy * (1.0/duty - 1.0))
    f = fan()
    tps = n / SECONDS
    res["rows"].append({"duty": duty, "tiles_per_second": round(tps, 3), "fan_rpm": f})
    print(f"  duty {duty:4.2f}  {tps:5.3f} tiles/s  fan {f} RPM", flush=True)
    time.sleep(20)                               # let it settle before the next step
res["idle_fan_after_settle"] = fan()
(R/"results"/"duty_probe.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res["rows"]), flush=True)
