"""ARM F, stage 2: primary-checkpoint inference on every ARM F condition, ARM X's path
(ops/xs_infer.py applies the MPS device patch at import). Both directions.

    env/bin/python3 ops/fs_infer.py [--ckpts primary|all] [--conds N0,NSH,...] [seg ...]
"""
import json, os, pathlib, sys, time
import numpy as np, torch, zarr
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import xs_infer as XI  # noqa: E402

SEGS = ["w00", "ag144", "ag174"]
ORDER = ["N0", "NSH", "P0", "PDEG", "NINV", "NS", "NH", "NSDH"]


def ct_path(seg, cond):
    if cond == "N0": return ROOT / "data" / "zs" / seg / "w04" / "ct.zarr"
    if cond == "P0": return ROOT / "data" / "ys" / seg / "w13" / "ct.zarr"
    return ROOT / "data" / "fs" / seg / cond / "ct.zarr"


def opt(argv, name, default):
    if name in argv:
        i = argv.index(name); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


def main(argv):
    csel = opt(argv, "--ckpts", "primary"); conds = opt(argv, "--conds", ",".join(ORDER)).split(",")
    segs = argv or SEGS; log = ROOT / "results" / "fs" / "infer_runs.jsonl"
    villa_sha = os.popen(f"git -C {ROOT / 'villa'} rev-parse HEAD").read().strip()
    for cond in conds:
        for seg in segs:
            ct = ct_path(seg, cond); D, H, W = zarr.open_array(str(ct), mode="r").shape; assert D == 21, (ct, D)
            for ck in XI.checkpoints(csel):
                for d in ("forward", "reverse"):
                    out = ROOT / "results" / "fs" / "pred" / seg / cond / f"{ck.parent.name}_{ck.stem}_{d}.tif"
                    if XI.valid_tiff(out, (H, W)): continue
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_name(f"{out.stem}.partial{os.getpid()}.tif")
                    if tmp.exists(): tmp.unlink()
                    t = time.time()
                    XI.INF.main([str(ct), str(ck), str(tmp), "--direction", d, "--no-compile", "--num-workers", "0", "--batch-size", "32"])
                    dt = time.time() - t
                    assert XI.valid_tiff(tmp, (H, W)), f"no valid output at {tmp}"
                    os.replace(tmp, out)
                    rec = dict(seg=seg, cond=cond, ckpt=f"{ck.parent.name}/{ck.name}", direction=d, secs=round(dt, 1), shape=[H, W],
                               device="mps" if torch.backends.mps.is_available() else "cpu", villa=villa_sha, torch=torch.__version__,
                               when=time.strftime("%Y-%m-%dT%H:%M:%S"))
                    with open(log, "a") as f: f.write(json.dumps(rec) + "\n")
                    print(f"DONE {cond} {seg} {rec['ckpt']} {d} {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
