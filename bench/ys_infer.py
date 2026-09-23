"""ARM Y, stage 3: villa's own ink inference on every prepared window.

Imports ops/xs_infer.py, which applies ARM X's only change (the MPS device) at
import, and reuses its checkpoint list and TIFF check, so ARM Y runs exactly the
ARM X inference path. fp32, both directions, one TIFF per (segment, window,
checkpoint, direction), written to a per-process .partial path and renamed.

    env/bin/python3 ops/ys_infer.py --windows prod|all --ckpts primary|all [seg ...]
"""
import json, os, pathlib, sys, time
import numpy as np, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import xs_infer as XI  # noqa: E402  (applies the MPS patch to villa's infer)

SEGS = ["w00", "ag144", "ag174"]


def opt(argv, name, default):
    if name in argv:
        i = argv.index(name); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


def main(argv):
    wsel = opt(argv, "--windows", "all"); csel = opt(argv, "--ckpts", "primary")
    segs = argv or SEGS
    log = ROOT / "results" / "ys" / "infer_runs.jsonl"
    villa_sha = os.popen(f"git -C {ROOT / 'villa'} rev-parse HEAD").read().strip()
    for seg in segs:
        sd = ROOT / "data" / "ys" / seg; prep = json.load(open(sd / "prep.json"))
        prod = f"w{prep['production_window'][0]:02d}"
        wins = [prod] if wsel == "prod" else [prod] + [w for w in sorted(prep["windows"]) if w != prod]
        for w in wins:
            ct = sd / w / "ct.zarr"
            import zarr; D, H, W = zarr.open_array(str(ct), mode="r").shape
            assert D == 21, (ct, D)
            for ck in XI.checkpoints(csel):
                for d in ("forward", "reverse"):
                    out = ROOT / "results" / "ys" / "pred" / seg / w / f"{ck.parent.name}_{ck.stem}_{d}.tif"
                    if XI.valid_tiff(out, (H, W)): continue
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_name(f"{out.stem}.partial{os.getpid()}.tif")
                    if tmp.exists(): tmp.unlink()
                    t = time.time()
                    XI.INF.main([str(ct), str(ck), str(tmp), "--direction", d, "--no-compile", "--num-workers", "0", "--batch-size", "32"])
                    dt = time.time() - t
                    assert XI.valid_tiff(tmp, (H, W)), f"no valid output at {tmp}"
                    os.replace(tmp, out)
                    rec = dict(seg=seg, window=w, ckpt=f"{ck.parent.name}/{ck.name}", direction=d, secs=round(dt, 1), shape=[H, W],
                               device="mps" if torch.backends.mps.is_available() else "cpu", villa=villa_sha, torch=torch.__version__,
                               when=time.strftime("%Y-%m-%dT%H:%M:%S"))
                    with open(log, "a") as f: f.write(json.dumps(rec) + "\n")
                    print(f"DONE {seg} {w} {rec['ckpt']} {d} {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
