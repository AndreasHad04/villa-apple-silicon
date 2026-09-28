"""ARM FLINJ phase 1, Injection 3: I' = clip(I + a x G(y,x) x R(z,y,x), 0, 255) as uint8, R tiling donor
residual patches drawn at random with one fixed seed per host, written as a zarr in the layout every earlier
arm's inference read.

Modes: 'real' (the library as tiled), 'shuffle' (C1: the 21 planes of each tiled patch permuted, one permutation
per tile), 'flat' (C2: each tiled patch replaced by its own mean over depth and space), 'profile' (amendment 1, S6
and C3: the donor library's mean depth profile, no in-plane texture). The library is the smoothed one (A2 primary)
or the raw one (S5), chosen by the caller.

NOT registered, fixed here before any injected input existed (listed in the FLINJ report):
  J1 R tiles a 32 px grid anchored at the crop's (0, 0); each cell draws one patch, uniformly with replacement.
     The draw and the C1 permutations depend only on the host seed, so every amplitude, condition and stroke
     width of one host uses the SAME patches in the same cells.
  J2 the float result is rounded to nearest (np.rint) before the uint8 cast; truncation would bias every
     touched pixel by -0.5 even where a x G x R is zero.
  J3 C2's per-patch mean includes the patch's r = 0 non-ink pixels (R4), so the flat bump adds the same total
     residual per cell as the real one.
  J4 PHerc0009B: host pixels with x < W // 2 take R from the library built on the right half, the others from
     the left half (R5), decided per pixel.
"""
import numpy as np
from scipy.signal import fftconvolve
from common import REG

P = REG["patch_px"]


def draws(shape_hw, n_lib, seed, stream):
    rng = np.random.default_rng([seed, 2, stream])
    return rng.integers(0, n_lib, ((shape_hw[0] + P - 1) // P, (shape_hw[1] + P - 1) // P))


def perms(shape_hw, depth, seed):
    rng = np.random.default_rng([seed, 3])
    return np.argsort(rng.random(((shape_hw[0] + P - 1) // P, (shape_hw[1] + P - 1) // P, depth)), axis=-1)


def field(lib, idx, cond, perm, y0, y1, x0, x1):
    """R over rows [y0, y1) and columns [x0, x1) of the crop, float32 (depth, y1 - y0, x1 - x0)."""
    cy0, cy1, cx0, cx1 = y0 // P, (y1 - 1) // P + 1, x0 // P, (x1 - 1) // P + 1
    pat = lib[idx[cy0:cy1, cx0:cx1]]  # (ncy, ncx, depth, P, P)
    if cond == "shuffle":
        pat = np.take_along_axis(pat, perm[cy0:cy1, cx0:cx1][:, :, :, None, None], axis=2)
    elif cond == "flat":
        pat = np.broadcast_to(pat.mean(axis=(2, 3, 4), keepdims=True), pat.shape)
    else:
        assert cond == "real", cond
    ncy, ncx, D = pat.shape[:3]
    R = np.ascontiguousarray(pat.transpose(2, 0, 3, 1, 4)).reshape(D, ncy * P, ncx * P)
    return R[:, y0 - cy0 * P:y1 - cy0 * P, x0 - cx0 * P:x1 - cx0 * P]


def _R(libs, mode, seed, shape, box):
    """R over the box for a tiled library (mode real | shuffle | flat) or a depth profile (mode profile)."""
    D, H, W = shape; y0, y1, x0, x1 = box
    if mode == "profile":
        mk = lambda m: np.broadcast_to(np.asarray(m, np.float32)[:, None, None], (D, y1 - y0, x1 - x0))
    else:
        pm = perms((H, W), D, seed)
        mk = lambda lib, st: field(lib, draws((H, W), len(lib), seed, st), mode, pm, y0, y1, x0, x1)
    if "all" in libs:
        return mk(libs["all"]) if mode == "profile" else mk(libs["all"], 0)
    RL = mk(libs["left"]) if mode == "profile" else mk(libs["left"], 1)
    RR = mk(libs["right"]) if mode == "profile" else mk(libs["right"], 2)
    return np.where((np.arange(x0, x1) < W // 2)[None, None, :], RL, RR)


def inject(I, G, a, mode, libs, seed, strokes):
    """The injected uint8 volume (a copy) and dict(box, clip_frac, mean_change): A5's fraction of stroke voxels
    (stroke pixels x all planes) whose rounded value fell outside [0, 255], and the mean change on stroke voxels."""
    out = I.copy()
    ys, xs = np.nonzero(G > 0)
    box = [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]; y0, y1, x0, x1 = box
    raw = np.rint(I[:, y0:y1, x0:x1].astype(np.float32) + np.float32(a) * G[y0:y1, x0:x1][None] * _R(libs, mode, seed, I.shape, box))
    out[:, y0:y1, x0:x1] = np.clip(raw, 0, 255).astype(np.uint8)
    st = strokes[y0:y1, x0:x1]; assert st.sum() == strokes.sum(), "strokes outside the soft mask box"
    rs = raw[:, st]
    info = dict(box=box, clip_frac=float(((rs < 0) | (rs > 255)).mean()),
                mean_change=float((out[:, y0:y1, x0:x1][:, st].astype(np.float64) - I[:, y0:y1, x0:x1][:, st]).mean()))
    return out, info


def c3_scale(I, G, profiles, strokes, target=REG["c3_mean_change"]):
    """A5 C3: the scale s of (donor mean depth profile x soft glyph mask) giving a mean change of `target` grey levels
    on stroke voxels, found by bisection on the clipped, rounded result. The clipped fraction is reported, not forced."""
    D, H, W = I.shape; ys, xs = np.nonzero(strokes)
    Is = I[:, ys, xs].astype(np.float64); Gs = G[ys, xs].astype(np.float64)
    if "all" in profiles: M = np.repeat(np.asarray(profiles["all"], np.float64)[:, None], len(ys), 1)
    else: M = np.where((xs < W // 2)[None, :], np.asarray(profiles["left"], np.float64)[:, None], np.asarray(profiles["right"], np.float64)[:, None])
    change = lambda s: float((np.clip(np.rint(Is + s * Gs * M), 0, 255) - Is).mean())
    lo, hi = 0.0, 1.0
    while change(hi) < target:
        hi *= 2.0; assert hi < 1e6, "profile cannot reach the C3 target"
    for _ in range(60):
        mid = (lo + hi) / 2.0; lo, hi = (mid, hi) if change(mid) < target else (lo, mid)
    s = (lo + hi) / 2.0; r = np.rint(Is + s * Gs * M)
    return dict(scale=s, mean_change=change(s), clip_frac=float(((r < 0) | (r > 255)).mean()))


def host_background(I, bgmask, box, margin=REG["annulus_px"][1]):
    """b_host(z, y, x) over rows/cols of box: per plane, the mean over bgmask pixels in the 16 to 48 px annulus (B1).
    Computed on the box plus a 48 px margin, which equals the whole-crop computation. Returns (b, defined)."""
    from residuals import annulus
    D, H, W = I.shape; y0, y1, x0, x1 = box
    Y0, Y1, X0, X1 = max(y0 - margin, 0), min(y1 + margin, H), max(x0 - margin, 0), min(x1 + margin, W)
    K = annulus(); m = bgmask[Y0:Y1, X0:X1].astype(np.float64); den = np.rint(fftconvolve(m, K, mode="same"))
    sl = (slice(y0 - Y0, y1 - Y0), slice(x0 - X0, x1 - X0))
    b = np.stack([(fftconvolve(I[z, Y0:Y1, X0:X1].astype(np.float64) * m, K, mode="same") / np.maximum(den, 1.0))[sl] for z in range(D)]).astype(np.float32)
    return b, den[sl] > 0.5


def transplant(I, bgmask, binary, libs, seed, mode, stream):
    """B1 / B2: under the HARD glyph mask the host column is replaced by b_host + (donor I - donor b), donor patches
    tiled on the crop's 32 px grid (J1); mode real | shuffle (C1) | flat (C2). stream separates the ink and non-ink
    draws. Refuses (AssertionError) where b_host is undefined under the glyph."""
    ys, xs = np.nonzero(binary); box = [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]; y0, y1, x0, x1 = box
    bh, bdef = host_background(I, bgmask, box); st = binary[y0:y1, x0:x1]
    assert bdef[st].all(), f"b_host undefined at {int((~bdef[st]).sum())} of {int(st.sum())} glyph pixels"
    D, H, W = I.shape; pm = perms((H, W), D, seed)
    if "all" in libs:
        R = field(libs["all"], draws((H, W), len(libs["all"]), seed, stream), mode, pm, y0, y1, x0, x1)
    else:
        RL = field(libs["left"], draws((H, W), len(libs["left"]), seed, stream + 1), mode, pm, y0, y1, x0, x1)
        RR = field(libs["right"], draws((H, W), len(libs["right"]), seed, stream + 2), mode, pm, y0, y1, x0, x1)
        R = np.where((np.arange(x0, x1) < W // 2)[None, None, :], RL, RR)
    raw = np.rint(bh + R)[:, st]
    out = I.copy(); blk = out[:, y0:y1, x0:x1]; old = blk[:, st].astype(np.float64); blk[:, st] = np.clip(raw, 0, 255).astype(np.uint8)
    return out, dict(box=box, clip_frac=float(((raw < 0) | (raw > 255)).mean()), mean_change=float((blk[:, st] - old).mean()))
