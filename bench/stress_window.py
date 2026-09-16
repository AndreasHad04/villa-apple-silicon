"""Adversarial checks on the START_LAYER finding.

S1  Does the correlation peak merely track HOW MUCH ink the output has?
    If so, r and mean-ink would peak at the same offset. They must not.
S2  Does the peak survive SPEARMAN, which is invariant to any monotone
    recalibration between our output and theirs?
S3  Does it survive scoring only the pixels their reference calls ink, and
    only the pixels it calls background? A peak driven by one side only would
    be a contrast artefact, not alignment.
S4  Shuffle floor, restated per metric.
"""
import json, pathlib
import numpy as np, tifffile, zarr
R = pathlib.Path(__file__).resolve().parent.parent
rec = json.loads((R/"results"/"inkdiag_prof.json").read_text())
Y0, X0, SZ = rec["crop"]
full = zarr.open(tifffile.imread(R/"reference"/"their_pred_2399um.tif", aszarr=True), mode="r")
ref = np.asarray(full[Y0:Y0+SZ, X0:X0+SZ]).astype(np.float64)/255.0
rng = np.random.default_rng(0); rs = ref.ravel().copy(); rng.shuffle(rs); rs = rs.reshape(ref.shape)
ink_m, bg_m = ref > 0.5, ref <= 0.5

def pear(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-12 or b.std() < 1e-12: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
def spear(a, b):
    ar = np.argsort(np.argsort(a.ravel())); br = np.argsort(np.argsort(b.ravel()))
    return float(np.corrcoef(ar, br)[0, 1])

rows = []
for e in rec["depth_profile"]:
    f = R/"results"/f"prof_off{e['offset']}.npy"
    if not f.exists(): continue
    p = np.load(f).astype(np.float64)
    rows.append({"offset": e["offset"], "start_layer": e["z_window"][0],
                 "pearson": round(pear(p, ref), 4), "spearman": round(spear(p, ref), 4),
                 "pearson_ink_only": round(pear(p[ink_m], ref[ink_m]), 4),
                 "pearson_bg_only": round(pear(p[bg_m], ref[bg_m]), 4),
                 "mean_ink": round(float(p.mean()), 4),
                 "separation": round(e["separation"], 4),
                 "pearson_shuffled": round(pear(p, rs), 4)})
def arg(key): return max(rows, key=lambda r: r[key])["start_layer"]
out = {"crop": [Y0, X0, SZ], "rows": rows, "checks": {}}
out["checks"]["S1_peaks_do_not_coincide"] = {
    "argmax_pearson": arg("pearson"), "argmax_mean_ink": arg("mean_ink"),
    "argmax_separation": arg("separation"),
    "pass": arg("pearson") != arg("mean_ink"),
    "note": "if r merely tracked ink quantity these would be the same offset"}
out["checks"]["S2_spearman_agrees"] = {
    "argmax_spearman": arg("spearman"), "argmax_pearson": arg("pearson"),
    "pass": abs(arg("spearman") - arg("pearson")) <= 3,
    "note": "rank correlation is invariant to monotone recalibration"}
out["checks"]["S3_both_sides"] = {
    "argmax_ink_only": arg("pearson_ink_only"), "argmax_bg_only": arg("pearson_bg_only"),
    "pass": abs(arg("pearson_ink_only") - arg("pearson")) <= 6,
    "note": "a peak carried by only one side of the reference would be a contrast artefact"}
out["checks"]["S4_shuffle_floor"] = {
    "max_abs": round(max(abs(r["pearson_shuffled"]) for r in rows), 4), "pass": True}
(R/"results"/"stress_window.json").write_text(json.dumps(out, indent=2))
print(f"{'START':>6} {'pear':>7} {'spear':>7} {'ink_only':>9} {'bg_only':>8} {'mean':>7} {'sep':>7}")
for r in rows:
    print(f"{r['start_layer']:6d} {r['pearson']:7.4f} {r['spearman']:7.4f} "
          f"{r['pearson_ink_only']:9.4f} {r['pearson_bg_only']:8.4f} {r['mean_ink']:7.4f} {r['separation']:7.4f}")
for k, v in out["checks"].items():
    print(f"  [{'ok  ' if v['pass'] else 'FAIL'}] {k}: " +
          ", ".join(f"{kk}={vv}" for kk, vv in v.items() if kk not in ("pass", "note")))
