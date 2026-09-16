"""Which depth window did villa use for the published prediction?

The window is documented nowhere: not in the model card, not in the
checkpoint config.json, and the inference README only says "such as
START_LAYER=1, END_LAYER=63". If the agreement with the published map peaks
sharply at one offset, that offset recovers their setting.

Controls, because a peak alone is not evidence:
  - the SHUFFLE floor: every offset is also scored against a shuffled copy of
    the reference, which must sit at zero
  - the SHAPE test: a real recovery must be PEAKED. If every offset scores
    about the same, the metric is measuring marginal statistics, not alignment.
"""
import json, pathlib, sys
import numpy as np, tifffile, zarr

R = pathlib.Path(__file__).resolve().parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else "prof"
rec = json.loads((R/"results"/f"inkdiag_{tag}.json").read_text())
Y0, X0, SZ = rec["crop"]

full = zarr.open(tifffile.imread(R/"reference"/"their_pred_2399um.tif", aszarr=True), mode="r")
ref = np.asarray(full[Y0:Y0+SZ, X0:X0+SZ]).astype(np.float64)/255.0
rng = np.random.default_rng(0)
ref_sh = ref.ravel().copy(); rng.shuffle(ref_sh); ref_sh = ref_sh.reshape(ref.shape)

def r(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12: return float("nan")
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

rows = []
for e in rec["depth_profile"]:
    f = R/"results"/f"{tag}_off{e['offset']}.npy"
    if not f.exists(): continue
    p = np.load(f).astype(np.float64)
    rows.append({"offset": e["offset"], "z_window": e["z_window"],
                 "pearson_vs_reference": round(r(p, ref), 4),
                 "pearson_vs_shuffled_reference": round(r(p, ref_sh), 4),
                 "separation": e["separation"], "mean": e["mean"]})
rows.sort(key=lambda x: x["offset"])
best = max(rows, key=lambda x: x["pearson_vs_reference"]) if rows else None
vals = [x["pearson_vs_reference"] for x in rows]
out = {"crop": [Y0, X0, SZ], "n_offsets": len(rows), "rows": rows}
if rows:
    out["verdict"] = {
        "best_offset": best["offset"], "best_z_window": best["z_window"],
        "best_pearson": best["pearson_vs_reference"],
        "worst_pearson": round(min(vals), 4),
        "spread": round(max(vals)-min(vals), 4),
        "max_abs_shuffle_floor": round(max(abs(x["pearson_vs_shuffled_reference"]) for x in rows), 4),
        "is_peaked": bool(max(vals)-min(vals) > 0.15),
        "note": ("a recovery needs a PEAK. A flat profile means the metric is "
                 "reading marginal statistics, not depth alignment."),
    }
(R/"results"/f"offset_vs_reference_{tag}.json").write_text(json.dumps(out, indent=2))
for x in rows:
    print(f"  offset {x['offset']:+3d} z{x['z_window']}  r_ref {x['pearson_vs_reference']:+.4f}  "
          f"r_shuf {x['pearson_vs_shuffled_reference']:+.4f}  sep {x['separation']:.4f}")
if rows: print(json.dumps(out["verdict"], indent=2))
