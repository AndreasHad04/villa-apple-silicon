"""ARM F, stage 1: build the transformed production-window inputs, fitted WITHOUT labels,
leave one segment out. Pre-registration: results/fs/PREREGISTRATION_ARMF.md.

    env/bin/python3 ops/fs_make.py            # all conditions, all segments, plus C-ID
Writes data/fs/<seg>/<COND>/ct.zarr (21,H,W) uint8 and results/fs/transforms.json.
"""
import json, pathlib, shutil, sys, tempfile
import numpy as np, zarr
from scipy import fft as sfft
from scipy.ndimage import gaussian_filter1d

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
from fs_profile import pooled_index  # noqa: E402

SEGS = ["w00", "ag144", "ag174"]
PXN, PXP = 9.366, 9.612          # in-plane um per pixel, native and pooled inputs
DZN, DZP = 9.366, 9.612          # depth um per layer (pooled = zmean4 of 2.403 um)
FB = np.linspace(0.0, 1 / (2 * PXP), 27)   # radial bins, cycles per um, common range
CAP = 3.0
PAD = 64


def load(seg):
    n = np.asarray(zarr.open_array(str(ROOT / "data" / "zs" / seg / "w04" / "ct.zarr"), mode="r")[:])
    p = np.asarray(zarr.open_array(str(ROOT / "data" / "ys" / seg / "w13" / "ct.zarr"), mode="r")[:])
    sn = np.load(ROOT / "data" / "zs" / seg / "support.npy"); sp = np.load(ROOT / "data" / "ys" / seg / "support.npy")
    return n, p, sn, sp


def box(sup, L):
    ys, xs = np.where(sup); cy, cx = int(np.median(ys)), int(np.median(xs))
    for side in (L, 768, 512, 384, 256):
        r0 = min(max(0, cy - side // 2), sup.shape[0] - side); c0 = min(max(0, cx - side // 2), sup.shape[1] - side)
        if r0 >= 0 and c0 >= 0 and sup[r0:r0 + side, c0:c0 + side].mean() >= 0.98: return r0, c0, side
    raise RuntimeError("no in-support box")


def radial_psd(vol, px):
    """Mean over layers of the 2D power spectrum (Hann window), radially binned in cycles/um, normalised to unit total."""
    D, h, w = vol.shape; win = np.outer(np.hanning(h), np.hanning(w))
    P = np.zeros((h, w))
    for k in range(D):
        a = vol[k].astype(np.float64); a = (a - a.mean()) * win; P += np.abs(np.fft.fft2(a)) ** 2
    fy = np.fft.fftfreq(h, d=px); fx = np.fft.fftfreq(w, d=px); fr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    out = np.array([P[(fr >= FB[i]) & (fr < FB[i + 1])].mean() for i in range(len(FB) - 1)])
    out[0] = np.nan  # DC bin excluded from the shape
    return out / np.nansum(out)


def z_psd(vol, sup, dz):
    v = vol[:, sup].astype(np.float64); v = v - v.mean(0, keepdims=True)
    P = (np.abs(np.fft.rfft(v * np.hanning(v.shape[0])[:, None], axis=0)) ** 2).mean(1)
    f = np.fft.rfftfreq(v.shape[0], d=dz); P[0] = np.nan
    return f, P / np.nansum(P)


def fit_pair(seg):
    """Spectral shapes of one segment's native and pooled inputs over the same physical box."""
    n, p, sn, sp = load(seg); ii, jj = pooled_index(seg)
    r0, c0, L = box(n.min(0) > 0, 1024)  # rendered papyrus, not label support: spectra need area, not labels
    pr0, pc0 = ii[r0], jj[c0]; Lp = int(round(L * PXN / PXP))
    assert pr0 >= 0 and pc0 >= 0 and pr0 + Lp <= p.shape[1] and pc0 + Lp <= p.shape[2], "pooled box out of range"
    gn = radial_psd(n[:, r0:r0 + L, c0:c0 + L], PXN); gp = radial_psd(p[:, pr0:pr0 + Lp, pc0:pc0 + Lp], PXP)
    fzn, zn = z_psd(n, sn, DZN); fzp, zp = z_psd(p, sp, DZP)
    return dict(box=[int(r0), int(c0), int(L)], psd_native=gn, psd_pooled=gp, fz_native=fzn, z_native=zn, fz_pooled=fzp, z_pooled=zp)


def gain_curve(fits, invert=False):
    lg = np.nanmean([0.5 * np.log(f["psd_pooled"] / f["psd_native"]) for f in fits], axis=0)
    lg[0] = 0.0; lg = gaussian_filter1d(lg, 1.0, mode="nearest"); lg[0] = 0.0
    g = np.clip(np.exp(-lg if invert else lg), 1 / CAP, CAP); g[0] = 1.0
    return g


def z_gain_curve(fits, invert=False):
    f = fits[0]["fz_native"]
    lg = np.nanmean([0.5 * np.log(np.interp(f, x["fz_pooled"][1:], x["z_pooled"][1:]) / x["z_native"]) for x in fits], axis=0)
    lg[0] = 0.0; g = np.clip(np.exp(-lg if invert else lg), 1 / CAP, CAP); g[0] = 1.0
    return f, g


def apply_inplane(vol, g, px):
    """Per layer radial filter g (on FB bin centres, cycles/um), reflect padded; returns float64."""
    D, h, w = vol.shape; H, W = h + 2 * PAD, w + 2 * PAD
    fy = np.fft.fftfreq(H, d=px); fx = np.fft.rfftfreq(W, d=px); fr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    ctr = 0.5 * (FB[:-1] + FB[1:]); ctr[0] = 0.0
    G = np.interp(fr, ctr, g, right=g[-1])
    out = np.empty(vol.shape, np.float64)
    for k in range(D):
        a = np.pad(vol[k].astype(np.float64), PAD, mode="reflect")
        out[k] = sfft.irfft2(sfft.rfft2(a, workers=-1) * G, s=(H, W), workers=-1)[PAD:PAD + h, PAD:PAD + w]
    return out


def apply_depth(vol, fz, gz, dz):
    D = vol.shape[0]; v = np.concatenate([vol[::-1], vol, vol[::-1]], axis=0)  # reflect to 3D
    f = np.fft.rfftfreq(3 * D, d=dz); G = np.interp(f, fz, gz, right=gz[-1])
    return np.fft.irfft(np.fft.rfft(v, axis=0) * G[:, None, None], n=3 * D, axis=0)[D:2 * D]


def quantile_map(src_vals, tgt_vals):
    q = np.linspace(0, 100, 1001)
    return np.percentile(src_vals, q), np.percentile(tgt_vals, q)


def to_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def write(seg, cond, ct):
    dest = ROOT / "data" / "fs" / seg / cond; dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=cond + ".", dir=dest.parent))
    zarr.open_array(str(tmp / "ct.zarr"), mode="w", shape=ct.shape, chunks=(ct.shape[0], 128, 128), dtype=np.uint8, zarr_format=2)[:] = ct
    if dest.exists(): shutil.rmtree(dest)
    tmp.rename(dest)


def native_chain(n, cond, gxy, fz, gz, ginv):
    x = n.astype(np.float64)
    if cond in ("NS", "NSH", "NSDH"): x = apply_inplane(n, gxy, PXN)
    if cond in ("NSDH", "ND", "NDH"): x = apply_depth(x, fz, gz, DZN)
    if cond == "NINV": x = apply_inplane(n, ginv, PXN)
    return x


CONDS = ["NS", "NH", "NSH", "NSDH", "NINV", "PDEG"]


def main():
    fits = {s: fit_pair(s) for s in SEGS}
    data = {s: load(s) for s in SEGS}
    rec = dict(prereg_sha256=open(ROOT / "results" / "fs" / "prereg.sha256").read().split()[0], segments={})
    # C-ID: unit gain through the same in-plane path must reproduce N0 byte for byte
    n0 = data["w00"][0]; ident = to_uint8(apply_inplane(n0, np.ones(len(FB) - 1), PXN))
    rec["C_ID"] = dict(pass_=bool(np.array_equal(ident, n0)), max_abs_diff=int(np.abs(ident.astype(int) - n0).max()))
    print("C-ID", rec["C_ID"], flush=True); assert rec["C_ID"]["pass_"], "C-ID failed"
    for s in SEGS:
        others = [o for o in SEGS if o != s]; F = [fits[o] for o in others]
        gxy = gain_curve(F); ginv = gain_curve(F, invert=True); fz, gz = z_gain_curve(F)
        n, p, sn, sp = data[s]
        conds = {}
        for cond in [c for c in CONDS if c.startswith("N")]:
            x = native_chain(n, cond, gxy, fz, gz, ginv)
            if cond in ("NH", "NSH", "NSDH", "NDH"):
                src = np.concatenate([native_chain(data[o][0], cond, gxy, fz, gz, ginv)[:, data[o][2]].ravel() for o in others])
                tgt = np.concatenate([data[o][1][:, data[o][3]].ravel() for o in others])
                qs, qt = quantile_map(src, tgt); x = np.interp(x, qs, qt)
            ct = to_uint8(x); write(s, cond, ct)
            conds[cond] = dict(mean=float(ct[:, sn].mean()), sd=float(ct[:, sn].std()))
        if "PDEGZ" in CONDS:  # exploratory, amendment 2: in plane AND depth degraded, then mapped onto native values
            fzi, gzi = z_gain_curve(F, invert=True)
            deg = lambda v: apply_depth(apply_inplane(v, ginv, PXP), fzi, gzi, DZP)
            src = np.concatenate([deg(data[o][1])[:, data[o][3]].ravel() for o in others])
            tgt = np.concatenate([data[o][0][:, data[o][2]].ravel() for o in others])
            qs, qt = quantile_map(src, tgt); ct = to_uint8(np.interp(deg(p), qs, qt)); write(s, "PDEGZ", ct)
            conds["PDEGZ"] = dict(mean=float(ct[:, sp].mean()), sd=float(ct[:, sp].std()))
        if "PDEG" not in CONDS:
            rec["segments"][s] = dict(fitted_on=others, conds=conds); continue
        # PDEG: the pooled input made native-like (inverse in-plane gain at the pooled pixel size, then mapped onto native values)
        x = apply_inplane(p, ginv, PXP)
        src = np.concatenate([apply_inplane(data[o][1], ginv, PXP)[:, data[o][3]].ravel() for o in others])
        tgt = np.concatenate([data[o][0][:, data[o][2]].ravel() for o in others])
        qs, qt = quantile_map(src, tgt); ct = to_uint8(np.interp(x, qs, qt)); write(s, "PDEG", ct)
        conds["PDEG"] = dict(mean=float(ct[:, sp].mean()), sd=float(ct[:, sp].std()))
        rec["segments"][s] = dict(fitted_on=others, g_inplane=gxy.tolist(), g_inverse=ginv.tolist(), fz=fz.tolist(), g_depth=gz.tolist(),
                                  box=fits[s]["box"], conds=conds)
        print(s, "fitted on", others, "G in plane", np.round(gxy, 2).tolist(), "G depth", np.round(gz, 2).tolist(), flush=True)
    rec["fb_cycles_per_um"] = FB.tolist()
    name = "transforms.json" if CONDS == ["NS", "NH", "NSH", "NSDH", "NINV", "PDEG"] else "transforms_" + "_".join(CONDS) + ".json"
    json.dump(rec, open(ROOT / "results" / "fs" / name, "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1: CONDS = sys.argv[1].split(",")
    main()
