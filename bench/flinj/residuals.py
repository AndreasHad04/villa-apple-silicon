"""ARM FLINJ phase 1, Injection 1: the donor ink-residual library, real ink only.

Registered: for every labelled ink pixel inside the support of a DONOR crop, r(z,y,x) = I(z,y,x) - b(z,y,x),
b = mean over non-ink support pixels in an annulus of radius 16 to 48 px at the same plane. 32 x 32 patches are
cut where at least 90% of pixels are labelled ink. Donors never equal the host: PHerc0841 hosts take the other
two PHerc0841 crops, PHerc0500P2 takes all three, PHerc0009B is split into left and right halves, each half
hosting injections built from the other half.

NOT registered, fixed here before any injected input existed (listed in the FLINJ report):
  R1 annulus: pixel-centre distance 16 <= d <= 48 px.
  R2 patches are the cells of a non-overlapping 32 px grid anchored at the donor crop's (0, 0).
  R3 a patch is kept only if every labelled ink pixel in it has a defined b (a non-empty annulus of non-ink
     support). Deep inside wide traced strokes the annulus holds no non-ink pixel, so r does not exist there.
  R4 the (at most 10%) non-ink pixels of a kept patch carry r = 0: the residual is defined on ink only.
  R5 a PHerc0009B half is treated as its own donor crop (its annulus never reaches into the other half);
     the split is at column W // 2 of the crop.
  R6 the PHerc0841 donors of a host are pooled into one library; patches are drawn uniformly from the pool.
"""
import json, sys
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from common import DATA, HOSTS, REG, atomic_json, atomic_npz, load_host


def annulus(r0=REG["annulus_px"][0], r1=REG["annulus_px"][1]):
    y, x = np.mgrid[-r1:r1 + 1, -r1:r1 + 1]; d = np.hypot(y, x)
    return ((d >= r0) & (d <= r1)).astype(np.float64)


def cut(I, ink, support):
    """Residual patches (n, 21, 32, 32) float32 and counts, for one donor crop (arrays already restricted)."""
    P = REG["patch_px"]; K = annulus(); non = (support & ~ink).astype(np.float64)
    den = np.rint(fftconvolve(non, K, mode="same"))
    defined = den > 0.5
    H, W = ink.shape; ny, nx = H // P, W // P
    cells_ink = ink[:ny * P, :nx * P].reshape(ny, P, nx, P).mean((1, 3))
    cells_undef = (ink & ~defined)[:ny * P, :nx * P].reshape(ny, P, nx, P).any((1, 3))
    cand = cells_ink >= REG["patch_min_ink"]; keep = cand & ~cells_undef
    ky, kx = np.nonzero(keep)
    lib = np.zeros((len(ky), I.shape[0], P, P), np.float32)
    inkp = np.stack([ink[cy * P:(cy + 1) * P, cx * P:(cx + 1) * P] for cy, cx in zip(ky, kx)]) if len(ky) else np.zeros((0, P, P), bool)
    for z in range(I.shape[0]):
        Iz = I[z].astype(np.float64)
        b = fftconvolve(Iz * non, K, mode="same") / np.where(defined, den, 1.0)
        r = np.where(ink & defined, Iz - b, 0.0)
        for n, (cy, cx) in enumerate(zip(ky, kx)):
            lib[n, z] = r[cy * P:(cy + 1) * P, cx * P:(cx + 1) * P]
    stats = dict(ink_px=int(ink.sum()), ink_px_undefined_b=int((ink & ~defined).sum()), cells=int(ny * nx),
                 cells_ge90_ink=int(cand.sum()), cells_dropped_undefined_b=int((cand & cells_undef).sum()), patches=int(len(ky)),
                 mean_r=float(lib.mean()) if len(ky) else None,
                 mean_r_on_ink_by_plane=[float(lib[:, z][lib[:, z] != 0].mean()) if len(ky) else None for z in range(I.shape[0])])
    return lib, inkp, stats


def donor(key):
    """key: a PHerc0841 crop name, or p0009b_L / p0009b_R (R5). Cached under data/flinj/lib/."""
    f = DATA / "lib" / f"{key}.npz"
    if f.exists():
        z = np.load(f)
        if "ink" in z.files: return z["lib"], z["ink"], json.load(open(f.with_suffix(".json")))
    h = key.split("_")[0]; I, ink, support, _ = load_host(h)
    if key.endswith("_L") or key.endswith("_R"):
        mid = ink.shape[1] // 2; sl = slice(0, mid) if key.endswith("_L") else slice(mid, None)
        I, ink, support = I[:, :, sl], ink[:, sl], support[:, sl]
    lib, inkp, stats = cut(I, ink, support)
    stats.update(key=key, shape=list(ink.shape))
    atomic_npz(f, lib=lib, ink=inkp); atomic_json(f.with_suffix(".json"), stats)
    return lib, inkp, stats


def _pooled(h):
    """[(lib, ink)] per donor-half: {'all': ...} pooled, or {'left': ..., 'right': ...} for PHerc0009B, where 'left' is
    what host pixels with x < W // 2 use (built from the RIGHT half) and 'right' the reverse (R5, J4)."""
    d = HOSTS[h]["donors"]
    if d == "halves":
        return dict(left=donor("p0009b_R")[:2], right=donor("p0009b_L")[:2])
    assert h not in d
    parts = [donor(k)[:2] for k in d]
    return dict(all=(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])))


def smooth(lib, sigma=REG["smooth_sigma_px"]):
    """A2: each 32 x 32 patch smoothed in plane by a Gaussian (depth untouched). Per patch, scipy mode 'reflect'."""
    return ndi.gaussian_filter(lib, sigma=(0, 0, sigma, sigma), mode="reflect", truncate=4.0).astype(np.float32)


def host_libraries(h, kind="smooth"):
    """kind 'smooth' (A2 primary) or 'raw' (the registered residual, S5)."""
    assert kind in ("smooth", "raw"), kind
    return {k: (smooth(lib) if kind == "smooth" else lib) for k, (lib, _) in _pooled(h).items()}


def host_profiles(h):
    """A2 S6 and A5 C3: the donor library's mean depth profile, the mean of r over its INK pixels at each plane."""
    return {k: (lib * ink[:, None]).sum(axis=(0, 2, 3)) / ink.sum() for k, (lib, ink) in _pooled(h).items()}


def transplant_library(key, kind):
    """Amendment 2. kind 'ink' (B1): 32 px grid cells wholly inside support, >= 90% labelled ink; kind 'non' (B2): cells
    wholly inside support with every pixel farther than 48 px from any label. Both keep only cells where b is defined at
    EVERY pixel, and carry r = I - b at every pixel: the donor column as it is, spatially coherent. b per amendment 3, C1:
    the annulus mean over valid, not-labelled-ink pixels, support or not. Cached."""
    assert kind in ("ink", "non"), kind
    f = DATA / "lib" / f"{key}_{kind}_c1.npz"
    if f.exists(): return np.load(f)["lib"], json.load(open(f.with_suffix(".json")))
    h = key.split("_")[0]; I, ink, support, valid = load_host(h)
    if key.endswith("_L") or key.endswith("_R"):
        mid = ink.shape[1] // 2; sl = slice(0, mid) if key.endswith("_L") else slice(mid, None)
        I, ink, support, valid = I[:, :, sl], ink[:, sl], support[:, sl], valid[:, sl]
    P = REG["patch_px"]; K = annulus(); non = (valid & ~ink).astype(np.float64)  # amendment 3, C1: valid, not labelled ink
    den = np.rint(fftconvolve(non, K, mode="same")); bdef = den > 0.5
    H, W = ink.shape; ny, nx = H // P, W // P
    cell = lambda m: m[:ny * P, :nx * P].reshape(ny, P, nx, P)
    ok = cell(support).all((1, 3)) & cell(bdef).all((1, 3))
    if kind == "ink": keep = ok & (cell(ink).mean((1, 3)) >= REG["patch_min_ink"])
    else: keep = ok & cell(ndi.distance_transform_edt(~ink) > REG["b2_label_dist_px"]).all((1, 3))
    ky, kx = np.nonzero(keep); lib = np.zeros((len(ky), I.shape[0], P, P), np.float32)
    for z in range(I.shape[0]):
        Iz = I[z].astype(np.float64); r = Iz - fftconvolve(Iz * non, K, mode="same") / np.where(bdef, den, 1.0)
        for n, (cy, cx) in enumerate(zip(ky, kx)): lib[n, z] = r[cy * P:(cy + 1) * P, cx * P:(cx + 1) * P]
    stats = dict(key=key, kind=kind, patches=int(len(ky)), mean_r=float(lib.mean()) if len(ky) else None, std_r=float(lib.std()) if len(ky) else None)
    atomic_npz(f, lib=lib); atomic_json(f.with_suffix(".json"), stats)
    return lib, stats


def host_transplant_libraries(h, kind):
    """{'all': lib} pooled, or {'left', 'right'} for PHerc0009B (left = used at x < W // 2, built from the right half)."""
    d = HOSTS[h]["donors"]
    if d == "halves":
        return dict(left=transplant_library("p0009b_R", kind)[0], right=transplant_library("p0009b_L", kind)[0])
    assert h not in d
    return dict(all=np.concatenate([transplant_library(k, kind)[0] for k in d]))


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["w00", "ag144", "ag174", "p0009b_L", "p0009b_R"]):
        lib, _, s = donor(k); print(k, {a: b for a, b in s.items() if a != "mean_r_on_ink_by_plane"}, flush=True)
