"""ARM X, stage 3: score every prediction against the organisers' labels.

Pixel ROC AUC inside the support mask, computed EXACTLY from 256-bin histograms
(predictions are uint8, so binning loses nothing), per 64 px block so a block
bootstrap costs microseconds. Pre-registered rules live in PREREGISTRATION.md,
ARM X; this file only implements them, and says so where it decides anything.
"""
import json, pathlib, sys
import numpy as np, tifffile

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCK = 64
PRIMARY_SEGS = ["p0841_w00", "p0841_ag144", "p0841_ag174", "p0500p2a"]
PRIMARY_CKPT = "hybrid_3d2d-seed42_step-075000"


def block_hists(pred, label, support):
    """(nblocks, 256) positive and negative histograms over the support pixels."""
    H, W = label.shape
    by, bx = np.arange(H) // BLOCK, np.arange(W) // BLOCK
    bid = (by[:, None] * (bx.max() + 1) + bx[None, :])[support]
    v = pred[support].astype(np.int64); y = label[support]
    nb = int((by.max() + 1) * (bx.max() + 1))
    pos = np.bincount(bid[y] * 256 + v[y], minlength=nb * 256).reshape(nb, 256)
    neg = np.bincount(bid[~y] * 256 + v[~y], minlength=nb * 256).reshape(nb, 256)
    keep = (pos.sum(1) + neg.sum(1)) > 0
    return pos[keep], neg[keep]


def auc_from(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(equal), exact for integer scores."""
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0: return float("nan")
    below = np.concatenate([[0], np.cumsum(neg)[:-1]])
    return float((pos * (below + 0.5 * neg)).sum() / (npos * nneg))


def separation(pred, support):
    """Label free: mean(p | p > 0.5) minus mean(p | p <= 0.5), p in [0, 1]."""
    p = pred[support].astype(np.float64) / 255.0
    hi, lo = p[p > 0.5], p[p <= 0.5]
    return float(hi.mean() - lo.mean()) if hi.size and lo.size else float("nan")


def bootstrap(pos, neg, n=1000, seed=0):
    rng = np.random.default_rng(seed); k = len(pos)
    vals = [auc_from(pos[i].sum(0), neg[i].sum(0)) for i in (rng.integers(0, k, k) for _ in range(n))]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def shuffle_floor(pred, label, support, seeds=20, min_px=32):
    """C2: every supported pixel stays scored, but its LABEL is drawn from the
    supported pixels of a DIFFERENT 64 px block (a derangement over blocks with at
    least min_px supported pixels), keeping each donor block's ink rate. A first
    version scored only pixels supported in BOTH a block and its donor; on sparse
    validation masks that left so few pixels that the floor swung 0.09 to 0.92."""
    H, W = label.shape
    bid = (np.arange(H)[:, None] // BLOCK) * ((W - 1) // BLOCK + 1) + (np.arange(W)[None, :] // BLOCK)
    b, y, v = bid[support], label[support], pred[support].astype(np.int64)
    ids, counts = np.unique(b, return_counts=True); ok = ids[counts >= min_px]
    keep = np.isin(b, ok); b, y, v = b[keep], y[keep], v[keep]
    order = np.argsort(b, kind="stable"); b, y, v = b[order], y[order], v[order]
    starts = np.searchsorted(b, ok); ends = np.searchsorted(b, ok, side="right")
    out = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        while True:
            perm = rng.permutation(len(ok))
            if not (perm == np.arange(len(ok))).any(): break
        ys = np.empty_like(y)
        for k in range(len(ok)):
            d0, d1 = starts[perm[k]], ends[perm[k]]
            ys[starts[k]:ends[k]] = y[d0 + rng.integers(0, d1 - d0, ends[k] - starts[k])]
        out.append(auc_from(np.bincount(v[ys], minlength=256), np.bincount(v[~ys], minlength=256)))
    return out


def score_segment(seg):
    sd = ROOT / "data" / "xs" / seg; meta = json.load(open(sd / "meta.json"))
    label = np.load(sd / "label.npy"); support = np.load(sd / "support.npy")
    rows = {}
    for f in sorted((ROOT / "results" / "xs" / "pred" / seg).glob("*.tif")):
        if ".partial" in f.name: continue
        key, d = f.stem.rsplit("_", 1)
        pred = tifffile.imread(f)
        assert pred.shape == label.shape, (f, pred.shape, label.shape)
        pos, neg = block_hists(pred, label, support)
        r = rows.setdefault(key, {})
        r[d] = dict(auc=auc_from(pos.sum(0), neg.sum(0)), ci95=bootstrap(pos, neg, n=300), sep=separation(pred, support),
                    frac_gt_half=float((pred[support] > 127).mean()))
    # pre-registered secondary: the seed42 and seed43 finals averaged, per direction
    pd = ROOT / "results" / "xs" / "pred" / seg
    for d in ("forward", "reverse"):
        fa, fb = pd / f"hybrid_3d2d-seed42_step-075000_{d}.tif", pd / f"hybrid_3d2d-seed43_step-075000_{d}.tif"
        if fa.exists() and fb.exists():
            pred = ((tifffile.imread(fa).astype(np.uint16) + tifffile.imread(fb)) // 2).astype(np.uint8)
            pos, neg = block_hists(pred, label, support)
            rows.setdefault("ensemble_seed42+43_final", {})[d] = dict(auc=auc_from(pos.sum(0), neg.sum(0)), ci95=bootstrap(pos, neg, n=300),
                                                                        sep=separation(pred, support), frac_gt_half=float((pred[support] > 127).mean()))
    for key, r in rows.items():
        if "forward" in r and "reverse" in r:
            sf, sr = r["forward"]["sep"], r["reverse"]["sep"]
            # pre-registered: larger separation wins. A nan separation (no pixel on one
            # side of 0.5) is UNDECIDED, never a silent win for either direction.
            if np.isnan(sf) and np.isnan(sr): r["chosen"] = None
            elif np.isnan(sf): r["chosen"] = "reverse"
            elif np.isnan(sr): r["chosen"] = "forward"
            else: r["chosen"] = "forward" if sf >= sr else "reverse"
            r["auc_chosen"] = r[r["chosen"]]["auc"] if r["chosen"] else float("nan")
            r["auc_oracle"] = max(r["forward"]["auc"], r["reverse"]["auc"])
    out = dict(seg=seg, meta=dict((k, meta.get(k)) for k in ("kind", "source", "native_um", "pooled_um", "n_support", "n_ink")),
               ink_rate=float(label[support].mean()), rows=rows)
    if PRIMARY_CKPT in rows and rows[PRIMARY_CKPT].get("chosen"):
        pred = tifffile.imread(ROOT / "results" / "xs" / "pred" / seg / f"{PRIMARY_CKPT}_{rows[PRIMARY_CKPT]['chosen']}.tif")
        out["c2_shuffle"] = shuffle_floor(pred, label, support)
    return out


def main(segs):
    segs = segs or sorted(p.name for p in (ROOT / "results" / "xs" / "pred").iterdir() if p.is_dir())
    res = {s: score_segment(s) for s in segs}
    verdict = {}
    c1 = res.get("c1_p0139_w016", {}).get("rows", {}).get(PRIMARY_CKPT, {})
    # C1 uses the SAME label-free direction rule as the primary endpoint; the oracle is printed beside it
    c1_auc = c1.get("auc_chosen", float("nan")) if c1 else float("nan")
    verdict["C1_auc_chosen_direction"] = c1_auc; verdict["C1_direction"] = c1.get("chosen")
    verdict["C1_auc_oracle"] = c1.get("auc_oracle", float("nan"))
    verdict["C1_pass"] = bool(c1_auc >= 0.85)  # nan >= 0.85 is False: a missing C1 cannot pass
    prim = [res[s]["rows"][PRIMARY_CKPT]["auc_chosen"] for s in PRIMARY_SEGS if s in res and PRIMARY_CKPT in res[s]["rows"] and "auc_chosen" in res[s]["rows"][PRIMARY_CKPT]]
    verdict["primary_aucs"] = prim
    if len(prim) == len(PRIMARY_SEGS) and not any(np.isnan(prim)) and verdict["C1_pass"]:
        med = float(np.median(prim))
        verdict["primary_median"] = med
        verdict["verdict"] = ("TRANSFERS" if med >= 0.80 and min(prim) >= 0.70 else
                              "DEGRADES" if med >= 0.65 else "FAILS")
    else:
        verdict["verdict"] = "UNDECIDED: C1 failed, or a primary segment is missing or undecided"
    c2 = [x for s in res.values() for x in s.get("c2_shuffle", [])]
    verdict["C2_worst_abs_dev"] = float(np.nanmax(np.abs(np.array(c2) - 0.5))) if c2 else float("nan")
    verdict["C2_pass"] = bool(verdict["C2_worst_abs_dev"] < 0.03)  # the pre-registered bar, reported as is
    verdict["C2_per_segment"] = {s: dict(median=float(np.median(r["c2_shuffle"])), lo=float(min(r["c2_shuffle"])),
                                         hi=float(max(r["c2_shuffle"]))) for s, r in res.items() if r.get("c2_shuffle")}
    # diagnostic added AFTER C1 failed, labelled as such: the pipeline against kadenpool's published 0.912
    d = res.get("c1_p0139_w016", {}).get("rows", {}).get("hybrid_3d2d-seed43_step-060000", {})
    if d: verdict["C1_diagnostic_seed43_step060000_forward"] = d.get("forward", {}).get("auc")
    dst = ROOT / "results" / "xs" / "armx_scores.json"; tmp = dst.with_name(f"armx_scores.{__import__('os').getpid()}.tmp")
    tmp.write_text(json.dumps(dict(verdict=verdict, segments=res), indent=1)); tmp.replace(dst)  # atomic: two scorers may overlap
    print(json.dumps(verdict, indent=1))
    for s, r in res.items():
        p = r["rows"].get(PRIMARY_CKPT, {})
        print(f"{s:16s} ink {r['ink_rate']:.3f}  primary fwd {p.get('forward', {}).get('auc', float('nan')):.4f} "
              f"rev {p.get('reverse', {}).get('auc', float('nan')):.4f} chosen {p.get('chosen')} -> {p.get('auc_chosen', float('nan')):.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
