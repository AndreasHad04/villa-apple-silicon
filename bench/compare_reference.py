"""THE control that matters: does the Apple Silicon run reproduce the
organisers' OWN published prediction for the same pixels?

Agreement alone proves nothing, so this also runs a NEGATIVE control: the same
comparison against a DIFFERENT region of their own map. If the shifted
comparison scores as well as the aligned one, the metric is measuring the
marginal statistics of the map and not alignment at all.
"""
import json, pathlib, sys
import numpy as np, tifffile

R = pathlib.Path(__file__).resolve().parent.parent
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--y0", type=int, default=20000)
_ap.add_argument("--x0", type=int, default=40000); _ap.add_argument("--size", type=int, default=1024)
_ap.add_argument("--tag", default="smoke"); _a = _ap.parse_args()
Y0, X0, SZ = _a.y0, _a.x0, _a.size
mine = np.load(R / "results" / f"{_a.tag}_mps_{Y0}_{X0}_{SZ}.npy").astype(np.float64)

store = tifffile.imread(R / "reference" / "their_pred_2399um.tif", aszarr=True)
import zarr
full = zarr.open(store, mode="r")
theirs = np.asarray(full[Y0:Y0+SZ, X0:X0+SZ]).astype(np.float64) / 255.0

def stats(a, b):
    am, bm = a.ravel(), b.ravel()
    if am.std() < 1e-12 or bm.std() < 1e-12:
        return {"pearson": float("nan"), "note": "a constant field has no correlation"}
    r = float(np.corrcoef(am, bm)[0, 1])
    # rank correlation, robust to any monotone recalibration between the two
    ar = np.argsort(np.argsort(am)); br = np.argsort(np.argsort(bm))
    rs = float(np.corrcoef(ar, br)[0, 1])
    return {"pearson": round(r, 4), "spearman": round(rs, 4),
            "mine_mean": round(float(am.mean()), 4), "theirs_mean": round(float(bm.mean()), 4),
            "mine_std": round(float(am.std()), 4), "theirs_std": round(float(bm.std()), 4)}

res = {"crop": [Y0, X0, SZ], "aligned": stats(mine, theirs)}

# NEGATIVE CONTROLS: same map, wrong place, and a shuffle.
for dy, dx, tag in ((SZ, 0, "shift_down_1_crop"), (0, SZ, "shift_right_1_crop"),
                    (4096, 4096, "far_region")):
    other = np.asarray(full[Y0+dy:Y0+dy+SZ, X0+dx:X0+dx+SZ]).astype(np.float64) / 255.0
    if other.shape == mine.shape:
        res[tag] = stats(mine, other)
rng = np.random.default_rng(0)
sh = theirs.ravel().copy(); rng.shuffle(sh)
res["shuffled_theirs"] = stats(mine, sh.reshape(theirs.shape))

al = res["aligned"].get("pearson", float("nan"))
neg = [res[k]["pearson"] for k in res if k not in ("crop", "aligned")
       and isinstance(res[k], dict) and "pearson" in res[k] and np.isfinite(res[k]["pearson"])]
res["verdict"] = {
    "aligned_pearson": al,
    "best_negative_control": round(max(neg), 4) if neg else None,
    "aligned_beats_every_control": bool(neg) and all(al > n for n in neg),
    "note": ("aligned must beat every control. If it does not, the run is not "
             "reproducing their map at these pixels and the port is NOT verified."),
}
(R / "results" / f"compare_reference_{_a.tag}.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
