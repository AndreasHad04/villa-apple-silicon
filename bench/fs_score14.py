"""ARM F, descriptive: all 14 ink_9um checkpoints on NDH (depth sharpening + intensity map) against ARM Z's
stored N0 rows for the same checkpoints, PHerc0841, label-free direction. Exploratory (amendment 2 and 3).

    env/bin/python3 ops/fs_score14.py
"""
import json, pathlib, sys
import numpy as np, tifffile
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "ops"))
import ys_score as YS  # noqa: E402
SEGS = ["w00", "ag144", "ag174"]
Z = json.load(open(ROOT / "results" / "zs" / "armz_scores.json"))["segments"]
out = dict(rows={})
for s in SEGS:
    label, support = np.load(ROOT / "data" / "zs" / s / "label.npy"), np.load(ROOT / "data" / "zs" / s / "support.npy"); rows = {}
    for f in sorted((ROOT / "results" / "fs" / "pred" / s / "NDH").glob("*.tif")):
        if ".partial" in f.name: continue
        key, d = f.stem.rsplit("_", 1); rows.setdefault(key, {})[d] = YS.metrics(tifffile.imread(f), label, support)
    for k, r in rows.items():
        YS.finish(r)
        if "auc_chosen" in r and k in Z[s]["rows"]:
            out["rows"].setdefault(k, {})[s] = dict(n0=Z[s]["rows"][k]["auc_chosen"], ndh=r["auc_chosen"], dir_n0=Z[s]["rows"][k]["chosen"], dir_ndh=r["chosen"])
full = {k: v for k, v in out["rows"].items() if len(v) == 3}
d = [v[s]["ndh"] - v[s]["n0"] for v in full.values() for s in SEGS]
out["summary"] = dict(n_checkpoints=len(full), pairs=len(d), pairs_improved=int(sum(x > 0 for x in d)), median_change=float(np.median(d)) if d else None,
                      per_ckpt_median={k: dict(n0=float(np.median([v[s]["n0"] for s in SEGS])), ndh=float(np.median([v[s]["ndh"] for s in SEGS]))) for k, v in full.items()})
json.dump(out, open(ROOT / "results" / "fs" / "armf_ndh14.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out["summary"].items() if k != "per_ckpt_median"}))
for k, v in sorted(out["summary"]["per_ckpt_median"].items()): print(f"  {k:40s} N0 {v['n0']:.4f}  NDH {v['ndh']:.4f}  {v['ndh'] - v['n0']:+.4f}")
