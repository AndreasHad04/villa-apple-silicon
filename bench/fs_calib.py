"""For villa #1898 (khj1222): the Gaussian blur that makes the pooled 2.403 um production input match the eligible
9.366 um one, label free. Depth sigma (layers) matched on mean adjacent-layer correlation; in-plane sigma (px) matched on
the high-frequency power fraction; then the depth sigma re-matched after the in-plane blur. Same boxes as ops/fs_make.py."""
import json, pathlib, sys
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "ops"))
import fs_make as M  # noqa: E402


def adj(v):
    v = v.reshape(v.shape[0], -1); return float(np.mean([np.corrcoef(v[k], v[k + 1])[0, 1] for k in range(v.shape[0] - 1)]))


def hf(v, px):  # fraction of in-plane power above 0.6 of the common Nyquist, averaged over layers
    g = M.radial_psd(v, px); return float(np.nansum(g[int(0.6 * len(g)):]))


def solve(f, target, lo, hi):
    xs = np.linspace(lo, hi, 81); ys = np.array([f(x) for x in xs]); i = int(np.argmin(np.abs(ys - target)))
    return float(xs[i]), float(ys[i])


out = {}
for s in M.SEGS:
    n, p, sn, sp = M.load(s); f = M.fit_pair(s); r0, c0, L = f["box"]; ii, jj = M.pooled_index(s)
    nb = n[:, r0:r0 + L, c0:c0 + L].astype(np.float64); Lp = int(round(L * M.PXN / M.PXP))
    pb = p[:, ii[r0]:ii[r0] + Lp, jj[c0]:jj[c0] + Lp].astype(np.float64)
    tn_adj, tn_hf = adj(nb), hf(nb, M.PXN)
    sz0, got0 = solve(lambda z: adj(gaussian_filter1d(pb, z, axis=0, mode="reflect") if z > 0 else pb), tn_adj, 0, 2.0)
    sxy, got_hf = solve(lambda x: hf(gaussian_filter(pb, (0, x, x), mode="reflect") if x > 0 else pb, M.PXP), tn_hf, 0, 2.0)
    pxy = gaussian_filter(pb, (0, sxy, sxy), mode="reflect") if sxy > 0 else pb
    sz1, got1 = solve(lambda z: adj(gaussian_filter1d(pxy, z, axis=0, mode="reflect") if z > 0 else pxy), tn_adj, 0, 2.0)
    out[s] = dict(native_adj=tn_adj, pooled_adj=adj(pb), depth_sigma_alone=sz0, inplane_sigma=sxy, depth_sigma_after_inplane=sz1,
                  native_hf=tn_hf, pooled_hf=hf(pb, M.PXP), matched_adj=got1)
    print(s, {k: round(v, 3) for k, v in out[s].items()}, flush=True)
json.dump(out, open(ROOT / "results" / "fs" / "blur_calibration.json", "w"), indent=1)
