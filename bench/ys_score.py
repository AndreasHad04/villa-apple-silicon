"""ARM Y, stage 4: score every window against the published 20260918 labels.

AUC, block bootstrap, label-free separation and the C2 shuffle floor are ARM X's
own functions from ops/xs_score.py. The direction rule is restated here and is
checked against every ARM X row's stored choice (CHOOSE_EQ), so it cannot drift
from the rule that produced the published ARM X numbers. Endpoints are the ones
fixed in results/ys/PREREGISTRATION_ARMY.md.

    env/bin/python3 ops/ys_score.py
"""
import json, os, pathlib, sys
import numpy as np, tifffile
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import xs_score as XS  # noqa: E402

SEGS = {"w00": "p0841_w00", "ag144": "p0841_ag144", "ag174": "p0841_ag174"}
PRIMARY = "hybrid_3d2d-seed42_step-075000"
DECISIVE, C3_BAR, Y3_BAR = 0.05, 0.75, 0.03


def choose(sf, sr):
    """ARM X's pre-registered rule: larger separation wins; nan is never a silent win."""
    if np.isnan(sf) and np.isnan(sr): return None
    if np.isnan(sf): return "reverse"
    if np.isnan(sr): return "forward"
    return "forward" if sf >= sr else "reverse"


def metrics(pred, label, support):
    pos, neg = XS.block_hists(pred, label, support)
    p = pred[support].astype(np.float64); y = label[support]
    return dict(auc=XS.auc_from(pos.sum(0), neg.sum(0)), ci95=XS.bootstrap(pos, neg, n=300), sep=XS.separation(pred, support),
                label_sep_uint8=float(p[y].mean() - p[~y].mean()), frac_gt_half=float((pred[support] > 127).mean()))


def finish(r):
    if "forward" in r and "reverse" in r:
        r["chosen"] = choose(r["forward"]["sep"], r["reverse"]["sep"])
        r["auc_chosen"] = r[r["chosen"]]["auc"] if r["chosen"] else float("nan")
        r["label_preferred"] = "forward" if r["forward"]["auc"] >= r["reverse"]["auc"] else "reverse"
        r["gap_rev_minus_fwd"] = r["reverse"]["auc"] - r["forward"]["auc"]


def score(seg):
    sd = ROOT / "data" / "ys" / seg; meta = json.load(open(sd / "meta.json")); prep = json.load(open(sd / "prep.json"))
    label = np.load(sd / "label.npy"); support = np.load(sd / "support.npy"); org = np.load(sd / "orgpred.npy")
    out = dict(seg=seg, meta=meta, prep=prep, ink_rate=float(label[support].mean()), windows={})
    c3 = metrics(org, label, support); out["C3"] = dict(auc=c3["auc"], ci95=c3["ci95"], label_sep_uint8=c3["label_sep_uint8"], pass_=bool(c3["auc"] >= C3_BAR))
    pd = ROOT / "results" / "ys" / "pred" / seg
    for wd in sorted(p for p in pd.iterdir() if p.is_dir()) if pd.exists() else []:
        rows = {}
        for f in sorted(wd.glob("*.tif")):
            if ".partial" in f.name: continue
            key, d = f.stem.rsplit("_", 1)
            pred = tifffile.imread(f); assert pred.shape == label.shape, (f, pred.shape, label.shape)
            rows.setdefault(key, {})[d] = metrics(pred, label, support)
        for d in ("forward", "reverse"):  # ARM X's pre-registered secondary, when both finals exist
            fa, fb = wd / f"hybrid_3d2d-seed42_step-075000_{d}.tif", wd / f"hybrid_3d2d-seed43_step-075000_{d}.tif"
            if fa.exists() and fb.exists():
                pred = ((tifffile.imread(fa).astype(np.uint16) + tifffile.imread(fb)) // 2).astype(np.uint8)
                rows.setdefault("ensemble_seed42+43_final", {})[d] = metrics(pred, label, support)
        for r in rows.values(): finish(r)
        out["windows"][wd.name] = rows
    prod = f"w{prep['production_window'][0]:02d}"
    r = out["windows"].get(prod, {}).get(PRIMARY)
    if r and r.get("chosen"):
        pred = tifffile.imread(pd / prod / f"{PRIMARY}_{r['chosen']}.tif")
        out["C2_shuffle"] = XS.shuffle_floor(pred, label, support)
    return out


def main():
    X = json.load(open(ROOT / "results" / "xs" / "armx_scores.json"))["segments"]
    # CHOOSE_EQ: our rule must reproduce every stored ARM X choice
    n = bad = 0
    for s in X.values():
        for r in s["rows"].values():
            if "forward" in r and "reverse" in r:
                n += 1; bad += choose(r["forward"]["sep"], r["reverse"]["sep"]) != r.get("chosen")
    S = {k: score(k) for k in SEGS if (ROOT / "data" / "ys" / k / "prep.json").exists()}
    res = dict(CHOOSE_EQ=dict(rows=n, disagreements=bad, pass_=bad == 0), segments=S)
    prods = {k: f"w{v['prep']['production_window'][0]:02d}" for k, v in S.items()}
    # reference numbers are READ from ARM X's JSON, never typed
    c1 = json.load(open(ROOT / "results" / "xs" / "armx_scores.json"))["verdict"]["C1_auc_chosen_direction"]
    xs_med = float(np.median([X[v]["rows"][PRIMARY]["auc_chosen"] for v in SEGS.values()]))
    ok = [k for k in S if S[k]["C3"]["pass_"] and PRIMARY in S[k]["windows"].get(prods[k], {})]
    y1 = {}
    for k in ok:
        ry = S[k]["windows"][prods[k]][PRIMARY]; rx = X[SEGS[k]]["rows"][PRIMARY]
        rx_pref = "forward" if rx["forward"]["auc"] >= rx["reverse"]["auc"] else "reverse"
        y1[k] = dict(published_fwd=ry["forward"]["auc"], published_rev=ry["reverse"]["auc"], published_pref=ry["label_preferred"],
                     published_decisive=abs(ry["gap_rev_minus_fwd"]) >= DECISIVE, render_fwd=rx["forward"]["auc"], render_rev=rx["reverse"]["auc"],
                     render_pref=rx_pref, render_decisive=abs(rx["reverse"]["auc"] - rx["forward"]["auc"]) >= DECISIVE)
    both = [k for k in y1 if y1[k]["published_decisive"] and y1[k]["render_decisive"]]
    if any(y1[k]["published_pref"] != y1[k]["render_pref"] for k in both): v1 = "SUPPORTED: direction is a property of the array"
    elif len(both) == 3 and all(y1[k]["published_pref"] == y1[k]["render_pref"] for k in both): v1 = "REFUTED: same direction on both arrays"
    else: v1 = "UNDECIDED"
    res["Y1"] = dict(per_segment=y1, verdict=v1, segments_scored=ok)
    if len(ok) == 3:
        med = float(np.median([S[k]["windows"][prods[k]][PRIMARY]["auc_chosen"] for k in ok]))
        v2 = ("input-array effect: at or above C1" if med >= c1 else "array does not matter: within 0.03 of the renders" if abs(med - xs_med) <= 0.03 else "as measured")
        res["Y2"] = dict(published_median=med, render_median=xs_med, c1_in_distribution=c1, diff_vs_render=med - xs_med, verdict=v2)
    y3 = {}
    for k in ok:
        W = {w: rows[PRIMARY] for w, rows in S[k]["windows"].items() if PRIMARY in rows and "chosen" in rows[PRIMARY]}
        base = W[prods[k]]["auc_chosen"]; best = max(W, key=lambda w: W[w]["auc_chosen"])
        ob = max(((w, d) for w in W for d in ("forward", "reverse")), key=lambda t: W[t[0]][t[1]]["auc"])
        y3[k] = dict(n_windows=len(W), production=prods[k], production_auc=base, best_window=best, best_auc=W[best]["auc_chosen"],
                     gain=W[best]["auc_chosen"] - base, oracle_best=list(ob), oracle_auc=W[ob[0]][ob[1]]["auc"])
    if len(y3) == 3 and all(v["n_windows"] == 7 for v in y3.values()):
        res["Y3"] = dict(per_segment=y3, verdict="slice matters on this array" if sum(v["gain"] >= Y3_BAR for v in y3.values()) >= 2 else "slice does not matter on this array")
    else:
        res["Y3_partial"] = y3
    # unit 3: every checkpoint at the production window
    ck = sorted({c for k in ok for c in S[k]["windows"].get(prods[k], {}) if c.startswith("hybrid")})
    if ok and all(all(c in S[k]["windows"][prods[k]] for k in ok) for c in ck) and len(ck) == 14:
        med = {c: float(np.median([S[k]["windows"][prods[k]][c]["auc_chosen"] for k in ok])) for c in ck}
        c1s = {c: X["c1_p0139_w016"]["rows"][c]["auc_chosen"] for c in ck}
        rx = {c: float(np.median([X[SEGS[k]]["rows"][c]["auc_chosen"] for k in ok])) for c in ck}
        c1o = {c: X["c1_p0139_w016"]["rows"][c]["auc_oracle"] for c in ck}  # POST-HOC: the label-free rule picked the wrong direction on 2 C1 rows
        steps = {s: [f"hybrid_3d2d-{s}_step-{x}" for x in ("010000", "020000", "030000", "040000", "050000", "060000", "075000")] for s in ("seed42", "seed43")}
        res["ALLCKPT"] = dict(published_median=med, render_median=rx, c1=c1s, c1_oracle=c1o,
                              c1_rule_wrong=[c for c in ck if X["c1_p0139_w016"]["rows"][c]["auc_chosen"] != c1o[c]],
                              spearman_c1_vs_published=float(spearmanr([c1s[c] for c in ck], [med[c] for c in ck])[0]),
                              spearman_c1oracle_vs_published=float(spearmanr([c1o[c] for c in ck], [med[c] for c in ck])[0]),
                              spearman_render_vs_published=float(spearmanr([rx[c] for c in ck], [med[c] for c in ck])[0]),
                              spearman_step_vs_published={s: float(spearmanr(range(7), [med[k] for k in v])[0]) for s, v in steps.items()},
                              best_single=max(ck, key=lambda c: med[c]), best_single_median=max(med.values()))
        rows_all = [S[k]["windows"][prods[k]][c] for k in ok for c in ck]
        res["ALLCKPT"]["direction_rule_kept_better"] = [sum(r["chosen"] == r["label_preferred"] for r in rows_all), len(rows_all)]
    tmp = ROOT / "results" / "ys" / f"army_scores.json.partial{os.getpid()}"
    json.dump(res, open(tmp, "w"), indent=1, default=float); os.replace(tmp, ROOT / "results" / "ys" / "army_scores.json")
    print(json.dumps({k: res[k] for k in res if k != "segments"}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
