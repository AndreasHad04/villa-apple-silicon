"""ARM H3: does fp16 autocast help hecate on MPS, where bf16 was 1.54x SLOWER?

hecate.py offers only fp32 and bf16 and refuses bf16 off CUDA. On this M1 Max, bf16 on MPS
measured 1.54x SLOWER than fp32 (results/hecate_bf16.json). This runs the SAME pipeline
twice, changing only the precision (3 lines), with min/median/max per villa AGENTS.md 1.4.
"""
import importlib.util, json, time, sys
import numpy as np, torch

HERE = "/Users/andreashad04/money/vesuv"
CKPT = f"{HERE}/hecate/hecate_2.4um.pth"
VOL  = f"{HERE}/hecate/surface_1024.npy"
REPS = 3

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

def run(mod, precision, vol, reps):
    model = mod.load_model(CKPT, device="mps")
    h, w = vol.shape[1], vol.shape[2]
    out = np.zeros((h, w), dtype=np.uint8)
    mod.predict(model, vol, out, precision=precision)          # warmup, untimed
    ts = []
    for _ in range(reps):
        out = np.zeros((h, w), dtype=np.uint8)
        t0 = time.time(); mod.predict(model, vol, out, precision=precision)
        torch.mps.synchronize(); ts.append(round(time.time() - t0, 3))
        print(f"  {precision} rep {ts[-1]}s", flush=True)
    return ts, out

vol = np.load(VOL)
print("volume", vol.shape, vol.dtype, flush=True)
base = load_mod("hec_base", f"{HERE}/hecate/hecate.py")
var  = load_mod("hec_fp16", f"{HERE}/hecate/hecate_fp16mps.py")

res = {"volume_shape": list(vol.shape), "reps": REPS}

# negative control FIRST: the unmodified module must REFUSE fp16 (it has no such mode)
try:
    m = base.load_model(CKPT, device="mps")
    o = np.zeros((vol.shape[1], vol.shape[2]), dtype=np.uint8)
    base.predict(m, vol, o, precision="fp16")
    res["upstream_refuses_fp16_on_mps"] = False
except (ValueError, TypeError) as e:
    res["upstream_refuses_fp16_on_mps"] = True
    res["upstream_refusal_message"] = str(e)
print("control: upstream refuses fp16 on mps =", res["upstream_refuses_fp16_on_mps"], flush=True)

t32, o32 = run(base, "fp32", vol, REPS)
tbf, obf = run(var,  "fp16", vol, REPS)

def stats(t): return {"min": min(t), "median": float(np.median(t)), "max": max(t), "all": t}
d = np.abs(o32.astype(np.int16) - obf.astype(np.int16))
res.update({
    "fp32_mps": stats(t32), "fp16_mps": stats(tbf),
    "speedup_median": round(float(np.median(t32)) / float(np.median(tbf)), 3),
    "fp16_vs_fp32": {
        "max_abs_diff_255": int(d.max()),
        "mean_abs_diff_255": float(d.mean()),
        "pixels_differing": int((d > 0).sum()),
        "total_pixels": int(d.size),
        "frac_differing": float((d > 0).mean()),
        "pixels_crossing_0.5": int(((o32 >= 128) != (obf >= 128)).sum()),
    },
})
json.dump(res, open(f"{HERE}/results/hecate_fp16.json", "w"), indent=1)
print(json.dumps(res, indent=1), flush=True)
