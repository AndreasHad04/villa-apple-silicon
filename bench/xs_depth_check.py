"""ARM X, EXPLORATORY (not pre-registered): is the held-out AUC depressed because
the labelled sheet sits off the centre of the window? Shifts the 17-plane window
by k pooled planes in the direction the label-free rule picked, and scores each.
Outputs go to results/xs/depthcheck/, NOT pred/, so the scorer never mistakes a
shifted window for a checkpoint.

    env/bin/python3 ops/xs_depth_check.py <segment> <ckpt rel path> <direction>
"""
import json, pathlib, sys
import numpy as np, tifffile
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.argv, (seg, ck, d) = [sys.argv[0]], sys.argv[1:4]
sys.path.insert(0, str(ROOT / "ops")); import xs_infer as XI, xs_score as X  # noqa: E402
sd = ROOT / "data" / "xs" / seg; meta = json.load(open(sd / "meta.json"))
lab, sup = np.load(sd / "label.npy"), np.load(sd / "support.npy")
depth = meta["ct_shape"][0]; out = {}
base = (6, 27) if depth == 32 else (0, 21)   # the pre-registered windows; the model centre-crops 17 of 21
for k in (-4, -2, 0, 2, 4):
    a, b = base[0] + k, base[1] + k
    if a < 0 or b > depth: continue
    f = ROOT / "results" / "xs" / "depthcheck" / seg / f"{pathlib.Path(ck).parent.name}_{pathlib.Path(ck).stem}_{d}_k{k:+d}.tif"
    same = ROOT / "results" / "xs" / "pred" / seg / f"{pathlib.Path(ck).parent.name}_{pathlib.Path(ck).stem}_{d}.tif"
    if k == 0 and same.exists():
        f = same  # the k=0 window IS the scored prediction; reuse it rather than recompute
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True); tmp = f.with_name(f.stem + ".partial.tif")
        XI.INF.main([str(sd / "ct.zarr"), str(ROOT / "models" / "ink_9um" / ck), str(tmp), "--layer-start", str(a), "--layer-end", str(b),
                     "--direction", d, "--no-compile", "--num-workers", "0", "--batch-size", "32"])
        tmp.replace(f)
    pos, neg = X.block_hists(tifffile.imread(f), lab, sup); out[k] = X.auc_from(pos.sum(0), neg.sum(0))
res = dict(segment=seg, ckpt=ck, direction=d, window_base=base, auc_by_shift=out, exploratory=True)
p = ROOT / "results" / "xs" / "depthcheck" / f"{seg}__{pathlib.Path(ck).parent.name}_{pathlib.Path(ck).stem}_{d}.json"
p.write_text(json.dumps(res, indent=1)); print(seg, d, {k: round(v, 4) for k, v in out.items()})
