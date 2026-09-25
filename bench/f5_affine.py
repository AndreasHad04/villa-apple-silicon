"""ARM F, amendment 3 sensitivity (not an endpoint): PHerc0139 labels resampled onto the native canvas with a per-axis
scale-and-offset fitted to the four quadrant alignment peaks, instead of one global shift. Re-scores every F5 condition.

    env/bin/python3 ops/f5_affine.py
"""
import json, pathlib, sys
import numpy as np, tifffile
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "ops"))
import ys_score as YS  # noqa: E402
F5 = ROOT / "data" / "f5" / "data"; S = 9.362 / 9.596; CONDS = ("N0", "NSH", "NSDH", "NDH", "ND")
out = dict(segments={})
for name in ("w030", "w045", "w043"):
    zm = json.load(open(F5 / "zs" / name / "meta.json")); ym = json.load(open(F5 / "ys" / name / "meta.json"))
    i0, i1, j0, j1 = zm["crop"]; y0, _, x0, _ = ym["bbox_level2"]; q = zm["zalign"]["quadrant_peaks_diagnostic"]
    h, w = i1 - i0, j1 - j0
    # per-axis model: dy depends on the row, dx on the column; slopes from top vs bottom and left vs right quadrant peaks
    dy_top, dy_bot = np.mean([q["tl"][0], q["tr"][0]]), np.mean([q["bl"][0], q["br"][0]])
    dx_left, dx_right = np.mean([q["tl"][1], q["bl"][1]]), np.mean([q["tr"][1], q["br"][1]])
    ky, kx = (dy_bot - dy_top) / (h / 2), (dx_right - dx_left) / (w / 2)
    oy, ox = (dy_top + dy_bot) / 2, (dx_left + dx_right) / 2
    r = np.arange(h); c = np.arange(w)
    dy = oy + ky * (r - (h - 1) / 2); dx = ox + kx * (c - (w - 1) / 2)
    lab2, sup2 = np.load(F5 / "ys" / name / "label.npy"), np.load(F5 / "ys" / name / "support.npy")
    ii = np.rint((np.arange(i0, i1) + dy + 0.5) * S - 0.5).astype(int) - y0
    jj = np.rint((np.arange(j0, j1) + dx + 0.5) * S - 0.5).astype(int) - x0
    okr, okc = (ii >= 0) & (ii < lab2.shape[0]), (jj >= 0) & (jj < lab2.shape[1])
    lab = np.zeros((h, w), bool); sup = np.zeros((h, w), bool)
    lab[np.ix_(okr, okc)] = lab2[np.ix_(ii[okr], jj[okc])]; sup[np.ix_(okr, okc)] = sup2[np.ix_(ii[okr], jj[okc])]
    lab &= sup
    base = np.load(F5 / "zs" / name / "support.npy")
    rec = dict(model=dict(oy=float(oy), ky_px_per_px=float(ky), ox=float(ox), kx_px_per_px=float(kx), edge_shift_range_y=[float(dy[0]), float(dy[-1])],
                          edge_shift_range_x=[float(dx[0]), float(dx[-1])], global_shift_used_by_F5=zm["zalign"]["used_offset"]),
               n_support=int(sup.sum()), n_support_global=int(base.sum()), auc={})
    for cond in CONDS:
        rows = {}
        for d in ("forward", "reverse"):
            f = ROOT / "results" / "fs" / "f5pred" / name / cond / f"{YS.PRIMARY}_{d}.tif"
            if f.exists(): rows[d] = YS.metrics(tifffile.imread(f), lab, sup)
        if len(rows) == 2:
            YS.finish(rows); rec["auc"][cond] = rows["auc_chosen"]
    out["segments"][name] = rec
    print(name, "affine dy", [round(float(dy[0]), 1), round(float(dy[-1]), 1)], "dx", [round(float(dx[0]), 1), round(float(dx[-1]), 1)],
          {k: round(v, 4) for k, v in rec["auc"].items()}, flush=True)
a = lambda n, c: out["segments"][n]["auc"].get(c)
for key, cond in (("F5", "NSH"), ("F5b", "NSDH"), ("F5c", "NDH")):
    g = [a(n, cond) - a(n, "N0") for n in ("w030", "w045") if a(n, cond) is not None]
    out[key] = dict(condition=cond, median_gain=float(np.median(g)), per_segment={n: a(n, cond) - a(n, "N0") for n in ("w030", "w045")})
    print(key, cond, "affine-label median gain", round(out[key]["median_gain"], 4), {k: round(v, 4) for k, v in out[key]["per_segment"].items()})
json.dump(out, open(ROOT / "results" / "fs" / "f5_affine.json", "w"), indent=1, default=float)
