"""ARM FLINJ phase 1 metrics.

Primary (registered): counter contrast on D = P(injected) - P(same crop, no injection),
    C = (mean D on strokes - mean D on counters) / (mean D on strokes - mean D on far background).
Calibration (registered): the model's AUC for injected stroke pixels against pixels farther than 2 cap heights
from any glyph. It is computed on P(injected), the model's output, so that it is comparable with the real
held-out AUC it is matched to (K1 in the report); the AUC of D is recorded beside it, descriptive only.
AUCs are exact for integer scores, through ARM X's own auc_from (ops/xs_score.py), not a copy of it.

A3 (amendment 1) floors are applied in score(). NOT registered (listed in the FLINJ report):
  K1 the calibration AUC is on P(injected), not on D.
  K2 C is undefined (nan) when mean D on strokes is not above mean D on far background.
"""
import pathlib
import sys
import numpy as np
from common import REG, ROOT
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # ops/ or bench/
import xs_score as XS  # noqa: E402


def contrast(D, geom):
    m = {k: (float(D[geom[k]].mean()) if geom[k].any() else float("nan")) for k in ("strokes", "counters", "far")}
    den = m["strokes"] - m["far"]
    C = (m["strokes"] - m["counters"]) / den if den > 0 else float("nan")
    return dict(C=C, mean_strokes=m["strokes"], mean_counters=m["counters"], mean_far=m["far"],
                n_strokes=int(geom["strokes"].sum()), n_counters=int(geom["counters"].sum()), n_far=int(geom["far"].sum()))


def auc_int(pos, neg):
    """Exact AUC, P(pos > neg) + 0.5 P(equal), for integer-valued scores of any range."""
    pos, neg = np.asarray(pos, np.int64), np.asarray(neg, np.int64)
    if pos.size == 0 or neg.size == 0: return float("nan")
    lo = min(pos.min(), neg.min()); n = int(max(pos.max(), neg.max()) - lo + 1)
    return XS.auc_from(np.bincount(pos - lo, minlength=n), np.bincount(neg - lo, minlength=n))


def score(P_inj, P0, geom, unit=255.0):
    """P_inj, P0: uint8 predictions (unit 255), or integer sums of n of them (unit 255 n). D in the same units.
    A3 (amendment 1): C is UNDEFINED unless (mean D strokes - mean D far) >= 0.02 in probability AND the
    stroke-versus-far AUC on P(injected) >= 0.55; C_a3 is None then, and counts as neither STROKES nor BLOBS."""
    D = P_inj.astype(np.int32) - P0.astype(np.int32)
    out = contrast(D, geom)
    out.update(auc_P=auc_int(P_inj[geom["strokes"]], P_inj[geom["far"]]),
               auc_D=auc_int(D[geom["strokes"]], D[geom["far"]]),
               max_abs_D=int(np.abs(D).max()), mean_abs_D_far=float(np.abs(D[geom["far"]]).mean()) if geom["far"].any() else float("nan"))
    out["d_prob"] = (out["mean_strokes"] - out["mean_far"]) / unit
    out["a3_defined"] = bool(np.isfinite(out["d_prob"]) and out["d_prob"] >= REG["d_floor_prob"] and out["auc_P"] >= REG["auc_floor"] and np.isfinite(out["C"]))
    out["C_a3"] = out["C"] if out["a3_defined"] else None
    return out


def mask_contrast(img, geom):
    """C of an image standing in for D (model-free checks M1, M2)."""
    return contrast(np.asarray(img, np.float64), geom)


def first_crossing(xs, ys, target):
    """Registered a* rule: the first grid interval (increasing a) where y crosses target, linear in log a.
    None when the grid does not bracket the target (no extrapolation). nan points are skipped."""
    pts = [(float(x), float(y)) for x, y in sorted(zip(xs, ys)) if np.isfinite(y)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 == target: return x0
        if (y0 - target) * (y1 - target) < 0:
            l0, l1 = np.log(x0), np.log(x1)
            return float(np.exp(l0 + (target - y0) * (l1 - l0) / (y1 - y0)))
    if pts and pts[-1][1] == target: return pts[-1][0]
    return None


def score_pair(P_a, P_b, geom, unit=255.0):
    """Amendment 2, B3: D = P_a - P_b (ink transplant minus non-ink transplant; for C3, C3 minus host). C on D as
    registered. C is UNDEFINED unless (mean D strokes - mean D far) / unit >= 0.02 AND the AUC of P_a against P_b over
    stroke pixels >= 0.55. max_abs_D_far must be 0: both inputs are the untouched host there."""
    D = P_a.astype(np.int64) - P_b.astype(np.int64); out = contrast(D, geom); st = geom["strokes"]
    out.update(auc_pair=auc_int(P_a[st], P_b[st]), d_prob=(out["mean_strokes"] - out["mean_far"]) / unit,
               max_abs_D_far=int(np.abs(D[geom["far"]]).max()) if geom["far"].any() else None)
    out["b3_defined"] = bool(np.isfinite(out["C"]) and out["d_prob"] >= REG["b3_d_floor"] and out["auc_pair"] >= REG["b3_auc_floor"])
    out["C_b3"] = out["C"] if out["b3_defined"] else None
    return out
