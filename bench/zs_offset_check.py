"""ARM Z, POST-HOC sensitivity check (not pre-registered): does the AUC depend on
the label offset Z-ALIGN chose? Re-resamples ARM Y's labels at every offset within
+/-2 px of the used one and re-scores the SAME predictions. The index formula is
restated here and asserted to reproduce the saved label.npy and support.npy
exactly at the used offset, so it cannot drift from ops/zs_fetch.py."""
import json, pathlib, sys
import numpy as np, tifffile
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "ops"))
import xs_score as XS, ys_score as YS, zs_fetch as ZF  # noqa: E402
res = {}
for k in YS.SEGS:
    zm = json.load(open(ROOT / "data" / "zs" / k / "meta.json")); ym = json.load(open(ROOT / "data" / "ys" / k / "meta.json"))
    i0, i1, j0, j1 = zm["crop"]; y0, y1, x0, x1 = ym["bbox_level2"]; uy, ux = zm["zalign"]["used_offset"]
    lab2 = np.load(ROOT / "data" / "ys" / k / "label.npy"); sup2 = np.load(ROOT / "data" / "ys" / k / "support.npy")
    def rs(img, dy, dx):
        ii = np.rint((np.arange(i0, i1) + dy + 0.5) * ZF.S - 0.5).astype(int) - y0; jj = np.rint((np.arange(j0, j1) + dx + 0.5) * ZF.S - 0.5).astype(int) - x0
        okr, okc = (ii >= 0) & (ii < img.shape[0]), (jj >= 0) & (jj < img.shape[1]); o = np.zeros((len(ii), len(jj)), bool); o[np.ix_(okr, okc)] = img[np.ix_(ii[okr], jj[okc])]; return o
    assert np.array_equal(rs(sup2, uy, ux), np.load(ROOT / "data" / "zs" / k / "support.npy")), k
    assert np.array_equal(rs(lab2, uy, ux) & rs(sup2, uy, ux), np.load(ROOT / "data" / "zs" / k / "label.npy")), k
    pred = tifffile.imread(ROOT / "results" / "zs" / "pred" / k / "w04" / f"{YS.PRIMARY}_forward.tif"); grid = {}
    for dy in range(uy - 2, uy + 3):
        for dx in range(ux - 2, ux + 3):
            s = rs(sup2, dy, dx); l = rs(lab2, dy, dx) & s; pos, neg = XS.block_hists(pred, l, s); grid[f"{dy},{dx}"] = XS.auc_from(pos.sum(0), neg.sum(0))
    best = max(grid, key=grid.get); res[k] = dict(used=[uy, ux], auc_used=grid[f"{uy},{ux}"], best_offset=best, best_auc=grid[best], zero_offset_auc=None)
    s = rs(sup2, 0, 0); l = rs(lab2, 0, 0) & s; pos, neg = XS.block_hists(pred, l, s); res[k]["zero_offset_auc"] = XS.auc_from(pos.sum(0), neg.sum(0))
    print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in res[k].items()}, flush=True)
json.dump(res, open(ROOT / "results" / "zs" / "offset_check.json", "w"), indent=1)
