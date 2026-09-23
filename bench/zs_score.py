"""ARM Z, stage 3: score the eligible 9.366 um predictions with ARM Y's scorer and
report the endpoint fixed in results/zs/PREREGISTRATION_ARMZ.md. Reference numbers
are READ from ARM X's and ARM Y's JSON, never typed.

    env/bin/python3 ops/zs_score.py
"""
import json, os, pathlib, sys
import numpy as np, tifffile
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import xs_score as XS, ys_score as YS  # noqa: E402
PRIMARY = YS.PRIMARY; BAND = 0.03
out = dict(segments={})
for k in YS.SEGS:
    sd = ROOT / "data" / "zs" / k; meta = json.load(open(sd / "meta.json"))
    if not (sd / "prep.json").exists(): out["segments"][k] = dict(meta=meta, excluded=True); continue
    label = np.load(sd / "label.npy"); support = np.load(sd / "support.npy"); prep = json.load(open(sd / "prep.json"))
    w = f"w{prep['production_window'][0]:02d}"; rows = {}
    for f in sorted((ROOT / "results" / "zs" / "pred" / k / w).glob("*.tif")):
        if ".partial" in f.name: continue
        key, d = f.stem.rsplit("_", 1); pred = tifffile.imread(f); assert pred.shape == label.shape
        rows.setdefault(key, {})[d] = YS.metrics(pred, label, support)
    pdir = ROOT / "results" / "zs" / "pred" / k / w
    for d in ("forward", "reverse"):  # ARM X's pre-registered secondary, when both finals exist
        fa, fb = pdir / f"hybrid_3d2d-seed42_step-075000_{d}.tif", pdir / f"hybrid_3d2d-seed43_step-075000_{d}.tif"
        if fa.exists() and fb.exists():
            rows.setdefault("ensemble_seed42+43_final", {})[d] = YS.metrics(((tifffile.imread(fa).astype(np.uint16) + tifffile.imread(fb)) // 2).astype(np.uint8), label, support)
    for r in rows.values(): YS.finish(r)
    rec = dict(meta=meta, window=w, ink_rate=float(label[support].mean()), rows=rows)
    if PRIMARY in rows and rows[PRIMARY].get("chosen"):
        rec["C2_shuffle"] = XS.shuffle_floor(tifffile.imread(ROOT / "results" / "zs" / "pred" / k / w / f"{PRIMARY}_{rows[PRIMARY]['chosen']}.tif"), label, support)
    out["segments"][k] = rec
Y = json.load(open(ROOT / "results" / "ys" / "army_scores.json")); ymed = Y["Y2"]["published_median"]; xmed = Y["Y2"]["render_median"]
ok = [k for k, v in out["segments"].items() if not v.get("excluded") and PRIMARY in v["rows"]]
if len(ok) == 3:
    med = float(np.median([out["segments"][k]["rows"][PRIMARY]["auc_chosen"] for k in ok]))
    v = ("the loss is the scan: the eligible 1.2 m scan reads worse than the 0.22 m kind" if med <= xmed + BAND else
         "the eligible scan reads as well as the training kind: ARM X's loss was the render" if med >= ymed - BAND else "between the two, as measured")
    out["Z1"] = dict(eligible_median=med, armY_published_median=ymed, armX_render_median=xmed, verdict=v,
                     per_segment={k: dict(fwd=out["segments"][k]["rows"][PRIMARY]["forward"]["auc"], rev=out["segments"][k]["rows"][PRIMARY]["reverse"]["auc"],
                                          chosen=out["segments"][k]["rows"][PRIMARY]["chosen"], label_preferred=out["segments"][k]["rows"][PRIMARY]["label_preferred"]) for k in ok})
ck = sorted({c for k in ok for c in out["segments"][k]["rows"] if c.startswith("hybrid")})
if len(ok) == 3 and len(ck) == 14 and all(c in out["segments"][k]["rows"] and "chosen" in out["segments"][k]["rows"][c] for k in ok for c in ck):
    from scipy.stats import spearmanr
    X = json.load(open(ROOT / "results" / "xs" / "armx_scores.json"))["segments"]
    emed = {c: float(np.median([out["segments"][k]["rows"][c]["auc_chosen"] for k in ok])) for c in ck + ["ensemble_seed42+43_final"] if all(c in out["segments"][k]["rows"] for k in ok)}
    c1 = {c: X["c1_p0139_w016"]["rows"][c]["auc_chosen"] for c in ck}
    rmed = {c: float(np.median([X["p0841_" + k]["rows"][c]["auc_chosen"] for k in ok])) for c in ck}
    sp = lambda a, b: float(spearmanr([a[c] for c in ck], [b[c] for c in ck])[0])
    A = dict(eligible_median=emed, c1=c1, render_median=rmed, Z2_spearman_c1_vs_eligible=sp(c1, emed), Z3_spearman_render_vs_eligible=sp(rmed, emed),
             Z4_best_single=max(ck, key=lambda c: emed[c]), Z4_best_single_median=max(emed[c] for c in ck), Z4_ensemble_finals=emed.get("ensemble_seed42+43_final"))
    if "ALLCKPT" in Y: A["Z3_spearman_2403_vs_eligible"] = sp(Y["ALLCKPT"]["published_median"], emed)
    c1o = {c: X["c1_p0139_w016"]["rows"][c]["auc_oracle"] for c in ck}  # POST-HOC, as in ys_score: correct-direction C1
    A["Z2_spearman_c1oracle_vs_eligible"] = sp(c1o, emed)
    A["spearman_step_vs_eligible"] = {s: float(spearmanr(range(7), [emed[f"hybrid_3d2d-{s}_step-{x}"] for x in ("010000", "020000", "030000", "040000", "050000", "060000", "075000")])[0]) for s in ("seed42", "seed43")}
    rows_all = [out["segments"][k]["rows"][c] for k in ok for c in ck]
    A["direction_rule_kept_better"] = [sum(r["chosen"] == r["label_preferred"] for r in rows_all), len(rows_all)]
    out["ALLCKPT"] = A; print(json.dumps({k: v for k, v in A.items() if not isinstance(v, dict)}, indent=1, default=float))
tmp = ROOT / "results" / "zs" / f"armz_scores.json.partial{os.getpid()}"
json.dump(out, open(tmp, "w"), indent=1, default=float); os.replace(tmp, ROOT / "results" / "zs" / "armz_scores.json")
print(json.dumps(out.get("Z1"), indent=1, default=float))
for k in ok: print(k, "C2 median %.4f range %.4f to %.4f" % (np.median(out["segments"][k]["C2_shuffle"]), min(out["segments"][k]["C2_shuffle"]), max(out["segments"][k]["C2_shuffle"])), "support", out["segments"][k]["meta"]["n_support"], "ink_rate %.3f" % out["segments"][k]["ink_rate"])
