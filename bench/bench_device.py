"""ARM A: does the organisers' own ink model run on Apple Silicon, and is it right?

Loads scrollprize/ink_canonical_2um through villa's OWN load_model (never a
reimplementation, control C3) and runs identical fixed tiles on CPU and MPS.

Reports three things and refuses to conflate them:
  CORRECTNESS  max|cpu - mps| on the same input, plus a NEGATIVE control on a
               different input so the agreement test is known to discriminate
  SPEED        tiles/sec on each device
  AUTOCAST     whether the amp_device line villa uses ("cuda" or else "cpu")
               behaves differently from the correct device_type on MPS
"""
import argparse, json, os, sys, time, pathlib
import numpy as np, torch

R = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "villa" / "ink-detection" / "optimized_inference"))
CKPT = R / "models" / "r152_3ddec_v2_l5_epoch13.ckpt"

from model_resnet3d_3d_decoder import load_model   # their code, unmodified

FRAMES, TILE = 62, 256          # README: resnet3d-152-3d-decoder wants these


def tiles(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((n, 1, FRAMES, TILE, TILE), generator=g, dtype=torch.float32)


def forward(model, x, device, amp_device=None, sync=True):
    x = x.to(device)
    with torch.inference_mode():
        if amp_device is None:
            y = model.forward(x)
        else:
            with torch.autocast(device_type=amp_device, enabled=True):
                y = model.forward(x)
    if sync and device == "mps":
        torch.mps.synchronize()
    return y.float().cpu()


def timed(model, x, device, amp_device, reps):
    forward(model, x[:1], device, amp_device)          # warm up, never timed
    t0 = time.time()
    for _ in range(reps):
        y = forward(model, x, device, amp_device)
    dt = (time.time() - t0) / reps
    return dt, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default=str(R / "results" / "bench_device.json"))
    a = ap.parse_args()

    res = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "torch": torch.__version__,
           "batch": a.batch, "reps": a.reps, "frames": FRAMES, "tile": TILE,
           "ckpt": CKPT.name}
    x = tiles(a.batch, seed=0)
    x_other = tiles(a.batch, seed=1)           # for the negative control

    for dev in ("cpu", "mps"):
        if dev == "mps" and not torch.backends.mps.is_available():
            res[dev] = {"ok": False, "err": "mps unavailable"}
            continue
        try:
            t0 = time.time()
            m = load_model(str(CKPT), torch.device(dev), num_frames=FRAMES)
            load_s = time.time() - t0
            dt, y = timed(m, x, dev, None, a.reps)
            res[dev] = {"ok": True, "load_seconds": round(load_s, 2),
                        "seconds_per_batch": round(dt, 3),
                        "tiles_per_second": round(a.batch / dt, 3),
                        "out_shape": list(y.shape),
                        "out_mean": float(y.mean()), "out_std": float(y.std())}
            res[dev + "_y"] = y
            if dev == "mps":
                # villa's line picks "cpu" whenever the device is not cuda.
                for tag, ampdev in (("villa_amp_cpu", "cpu"), ("correct_amp_mps", "mps")):
                    try:
                        dta, ya = timed(m, x, dev, ampdev, a.reps)
                        res[tag] = {"ok": True, "seconds_per_batch": round(dta, 3),
                                    "tiles_per_second": round(a.batch / dta, 3),
                                    "max_abs_diff_vs_fp32": float((ya - y).abs().max())}
                    except Exception as e:
                        res[tag] = {"ok": False, "err": f"{type(e).__name__}: {e}"[:300]}
                # negative control needs the model still loaded
                res["_y_other_mps"] = forward(m, x_other, dev, None)
            del m
            if dev == "mps":
                torch.mps.empty_cache()
        except Exception as e:
            res[dev] = {"ok": False, "err": f"{type(e).__name__}: {e}"[:400]}

    # CORRECTNESS, with the negative control that makes it mean something
    if res.get("cpu", {}).get("ok") and res.get("mps", {}).get("ok"):
        yc, ym, yo = res.pop("cpu_y"), res.pop("mps_y"), res.pop("_y_other_mps")
        same = float((yc - ym).abs().max())
        diff = float((yc - yo).abs().max())
        res["correctness"] = {
            "max_abs_diff_same_input": same,
            "max_abs_diff_DIFFERENT_input_negative_control": diff,
            "control_discriminates": diff > 10 * max(same, 1e-9),
            "note": ("the negative control must be far larger, or an agreement "
                     "of ~0 would prove nothing but that the model ignores input"),
        }
        res["speedup_mps_over_cpu"] = round(
            res["cpu"]["seconds_per_batch"] / res["mps"]["seconds_per_batch"], 2)
    for k in ("cpu_y", "mps_y", "_y_other_mps"):
        res.pop(k, None)

    pathlib.Path(a.out).parent.mkdir(exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
