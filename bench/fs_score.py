"""ARM F, stage 3: score every condition with ARM Y's metrics and label-free direction rule,
and report the endpoints fixed in results/fs/PREREGISTRATION_ARMF.md. Reference numbers are
READ from ARM Y's and ARM Z's JSON, never typed.

    env/bin/python3 ops/fs_score.py
"""
import json, os, pathlib, sys
import numpy as np, tifffile
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import xs_score as XS, ys_score as YS  # noqa: E402
PRIMARY = YS.PRIMARY; SEGS = ["w00", "ag144", "ag174"]
NCONDS = ["N0", "NS", "NH", "NSH", "NSDH", "NINV", "ND", "NDH"]; PCONDS = ["P0", "PDEG", "PDEGZ"]  # ND, NDH, PDEGZ: exploratory (amendment 2)
Z = json.load(open(ROOT / "results" / "zs" / "armz_scores.json"))["segments"]
Y = json.load(open(ROOT / "results" / "ys" / "army_scores.json"))


def lab(seg, cond):
    d = ROOT / "data" / ("ys" if cond.startswith("P") else "zs") / seg
    return np.load(d / "label.npy"), np.load(d / "support.npy")


out = dict(segments={})
for s in SEGS:
    rec = {}
    for cond in NCONDS + PCONDS:
        label, support = lab(s, cond); rows = {}
        for f in sorted((ROOT / "results" / "fs" / "pred" / s / cond).glob("*.tif")):
            if ".partial" in f.name: continue
            key, d = f.stem.rsplit("_", 1); pred = tifffile.imread(f); assert pred.shape == label.shape, (f, pred.shape, label.shape)
            rows.setdefault(key, {})[d] = YS.metrics(pred, label, support)
        for r in rows.values(): YS.finish(r)
        if PRIMARY in rows and "chosen" in rows[PRIMARY]: rec[cond] = rows
    out["segments"][s] = rec


def auc(s, c, ck=PRIMARY):
    r = out["segments"][s].get(c, {}).get(ck); return r["auc_chosen"] if r and "auc_chosen" in r else None


# C0: N0 and P0 reproduce the stored ARM Z and ARM Y primary AUCs
ysw = {s: json.load(open(ROOT / "data" / "ys" / s / "prep.json"))["production_window"][0] for s in SEGS}
c0 = {}
for s in SEGS:
    zst = Z[s]["rows"][PRIMARY]; yst = Y["segments"][s]["windows"][f"w{ysw[s]:02d}"][PRIMARY]
    c0[s] = dict(N0=[auc(s, "N0"), zst["auc_chosen"]], P0=[auc(s, "P0"), yst["auc_chosen"]])
out["C0"] = dict(per_segment=c0, pass_=all(v[0] is not None and abs(v[0] - v[1]) <= 1e-9 for d in c0.values() for v in d.values()))
done = [s for s in SEGS if all(auc(s, c) is not None for c in ("N0", "NSH", "P0"))]
if len(done) == 3:
    d1 = {s: auc(s, "NSH") - auc(s, "N0") for s in SEGS}; f1 = float(np.median(list(d1.values())))
    v1 = ("the family gap is partly repairable with a fixed label-free transform" if f1 >= 0.020 else
          "not repairable this way" if f1 <= 0.005 else "small, as measured")
    frac = {s: (auc(s, "NSH") - auc(s, "N0")) / (auc(s, "P0") - auc(s, "N0")) for s in SEGS}
    out["F1"] = dict(median_gain=f1, per_segment_gain=d1, fraction_of_gap_closed=frac, verdict=v1,
                     auc=dict(N0={s: auc(s, "N0") for s in SEGS}, NSH={s: auc(s, "NSH") for s in SEGS}, P0={s: auc(s, "P0") for s in SEGS}))
if all(auc(s, c) is not None for s in SEGS for c in ("P0", "PDEG", "N0")):
    gap = float(np.median([auc(s, "P0") for s in SEGS]) - np.median([auc(s, "N0") for s in SEGS]))
    f2 = float(np.median([auc(s, "P0") - auc(s, "PDEG") for s in SEGS]))
    out["F2"] = dict(median_drop=f2, half_gap=gap / 2, median_gap=gap, per_segment={s: auc(s, "PDEG") for s in SEGS},
                     verdict="sufficient to explain at least half the gap" if f2 >= gap / 2 else "explains less than half the gap")
if all(auc(s, c) is not None for s in SEGS for c in ("N0", "NINV")):
    dd = {s: auc(s, "NINV") - auc(s, "N0") for s in SEGS}
    out["C_DOSE"] = dict(per_segment=dd, pass_=all(v <= 0.005 for v in dd.values()))
out["F3"] = {c: {s: (auc(s, c) - auc(s, "N0")) if auc(s, c) is not None and auc(s, "N0") is not None else None for s in SEGS} for c in ("NS", "NH", "NSDH")}
dirs = {s: {c: out["segments"][s][c][PRIMARY]["chosen"] for c in out["segments"][s]} for s in SEGS}
out["directions_chosen"] = dirs
best_n = max((c for c in NCONDS if c != "N0" and all(auc(s, c) is not None for s in SEGS)), key=lambda c: np.median([auc(s, c) for s in SEGS]), default=None)
if best_n:
    out["best_N_condition"] = best_n
    out["C2_shuffle_best"] = {s: XS.shuffle_floor(tifffile.imread(ROOT / "results" / "fs" / "pred" / s / best_n / f"{PRIMARY}_{out['segments'][s][best_n][PRIMARY]['chosen']}.tif"), *lab(s, best_n)) for s in SEGS}
tmp = ROOT / "results" / "fs" / f"armf_scores.json.partial{os.getpid()}"
json.dump(out, open(tmp, "w"), indent=1, default=float); os.replace(tmp, ROOT / "results" / "fs" / "armf_scores.json")
print("C0", json.dumps(out["C0"], default=float))
for k in ("F1", "F2", "C_DOSE", "F3", "directions_chosen", "best_N_condition"):
    if k in out: print(k, json.dumps(out[k], default=float))
print("table (chosen-direction AUC, primary checkpoint)")
for c in NCONDS + PCONDS:
    print(f"  {c:5s}", "  ".join(f"{s} {auc(s, c):.4f}" if auc(s, c) is not None else f"{s} -" for s in SEGS))
