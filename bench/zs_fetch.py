"""ARM Z, stage 1: the ELIGIBLE native 9.366 um surface volumes of the ARM Y segments.

Pre-registration: results/zs/PREREGISTRATION_ARMZ.md. Labels are ARM Y's level-2
labels (2.403 um grid, 9.612 um per pixel) resampled nearest-neighbour onto the
9.366 um canvas, after the pre-registered Z-ALIGN check against ARM Y's own
production-window input.

    env/bin/python3 ops/zs_fetch.py [w00 ag144 ag174]
"""
import json, pathlib, shutil, sys, tempfile
import numpy as np, zarr
from scipy.ndimage import gaussian_filter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops")); sys.path.insert(0, str(ROOT / "villa" / "vesuvius" / "src"))
import ys_fetch as YF  # noqa: E402  (same S3 access and retry)
from vesuvius.ink_detection.preprocessing import prepare_9um_isotropic_input as P9  # noqa: E402

VOL9 = "9.366um-1.2m-113keV-volume-20250821151531"
S = 9.366 / 9.612  # level-2 pixels per 9.366 um pixel
SEARCH, PEAK_TOL, PEAK_RATIO, M = 8, 2, 3.0, 8


def hp(img):
    x = img.astype(np.float64); x = x - gaussian_filter(x, 8)
    return (x - x.mean()) / (x.std() + 1e-9)


def ncc_grid(A, B, valid):
    """NCC(dy, dx) = corr(A[i, j], B[i + dy, j + dx]) over pixels valid in both."""
    out = np.full((2 * SEARCH + 1, 2 * SEARCH + 1), np.nan); H, W = A.shape
    for a, dy in enumerate(range(-SEARCH, SEARCH + 1)):
        for b, dx in enumerate(range(-SEARCH, SEARCH + 1)):
            ys, ye, xs, xe = max(0, -dy), min(H, H - dy), max(0, -dx), min(W, W - dx)
            a_ = A[ys:ye, xs:xe]; b_ = B[ys + dy:ye + dy, xs + dx:xe + dx]
            v = valid[ys:ye, xs:xe] & valid[ys + dy:ye + dy, xs + dx:xe + dx]
            if v.sum() > 1000: out[a, b] = float(np.corrcoef(a_[v], b_[v])[0, 1])
    return out


def main(segs):
    for name in segs:
        dest = ROOT / "data" / "zs" / name
        if (dest / "meta.json").exists(): print("have", name, flush=True); continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        ym = json.load(open(ROOT / "data" / "ys" / name / "meta.json")); seg = ym["seg"]
        y0, y1, x0, x1 = ym["bbox_level2"]
        lab2 = np.load(ROOT / "data" / "ys" / name / "label.npy"); sup2 = np.load(ROOT / "data" / "ys" / name / "support.npy")
        m2 = np.asarray(zarr.open_array(str(ROOT / "data" / "ys" / name / "w13" / "ct.zarr"), mode="r")[:]).astype(np.float64).mean(0)
        a = YF.open_s3(seg + "/surface-volumes/" + VOL9 + ".zarr", "0", 2); D, H9, W9 = a.shape
        z0, z1 = P9.centered_slice(D, 21)
        i0 = max(0, int(np.floor(y0 / S)) - M); i1 = min(H9, int(np.ceil(y1 / S)) + M)
        j0 = max(0, int(np.floor(x0 / S)) - M); j1 = min(W9, int(np.ceil(x1 / S)) + M)
        vol = YF.retry(lambda: np.asarray(a[:, i0:i1, j0:j1]), "vol9")
        # nearest level-2 index of every 9.366 um pixel centre, in ARM Y's crop coordinates
        def resample(img, dy=0, dx=0, fill=0):
            """img (ARM Y crop, level-2 grid) onto the 9.366 um crop; (dy, dx) in 9.366 um pixels."""
            ii = np.rint((np.arange(i0, i1) + dy + 0.5) * S - 0.5).astype(int) - y0
            jj = np.rint((np.arange(j0, j1) + dx + 0.5) * S - 0.5).astype(int) - x0
            okr, okc = (ii >= 0) & (ii < img.shape[0]), (jj >= 0) & (jj < img.shape[1])
            out = np.full((len(ii), len(jj)), fill, dtype=img.dtype)
            out[np.ix_(okr, okc)] = img[np.ix_(ii[okr], jj[okc])]
            return out, np.outer(okr, okc)

        A = vol[z0:z1].astype(np.float64).mean(0); B, inb = resample(m2, fill=0.0)
        g = ncc_grid(hp(A), hp(B), inb)
        pk = np.unravel_index(np.nanargmax(g), g.shape); dy, dx = pk[0] - SEARCH, pk[1] - SEARCH
        peak, med = float(g[pk]), float(np.nanmedian(np.abs(g)))
        if abs(dy) <= PEAK_TOL and abs(dx) <= PEAK_TOL: use, rule = (0, 0), "peak within 2 px: zero offset"
        elif peak >= PEAK_RATIO * med: use, rule = (int(dy), int(dx)), "peak shift applied"
        else: use, rule = None, "EXCLUDED: no identifiable peak"
        quads = {}
        h, w = A.shape
        for qn, (ys_, xs_) in dict(tl=(slice(0, h // 2), slice(0, w // 2)), tr=(slice(0, h // 2), slice(w // 2, w)),
                                   bl=(slice(h // 2, h), slice(0, w // 2)), br=(slice(h // 2, h), slice(w // 2, w))).items():
            gq = ncc_grid(hp(A[ys_, xs_]), hp(B[ys_, xs_]), inb[ys_, xs_])
            if np.isfinite(gq).any():
                q = np.unravel_index(np.nanargmax(gq), gq.shape); quads[qn] = [int(q[0] - SEARCH), int(q[1] - SEARCH), round(float(gq[q]), 4)]
        zal = dict(peak_shift=[int(dy), int(dx)], peak_ncc=peak, median_abs_ncc=med, ratio=peak / med if med else None,
                   ncc_at_zero=float(g[SEARCH, SEARCH]), rule=rule, used_offset=use, quadrant_peaks_diagnostic=quads)
        print(f"{name}: 9.366 um {a.shape}, window [{z0},{z1}), crop rows [{i0},{i1}) cols [{j0},{j1}), Z-ALIGN {zal}", flush=True)
        tmp = pathlib.Path(tempfile.mkdtemp(prefix=name + ".", dir=dest.parent))
        meta = dict(seg=seg, source=f"s3://{YF.BUCKET}{seg}/surface-volumes/{VOL9}.zarr level 0", shape=list(a.shape), window=[z0, z1],
                    crop=[i0, i1, j0, j1], scale_level2_per_px=S, labels_from="data/ys/" + name, zalign=zal)
        if use is not None:
            lab, _ = resample(lab2, *use, fill=False); sup, _ = resample(sup2, *use, fill=False)
            (tmp / f"w{z0:02d}").mkdir()
            zarr.open_array(str(tmp / f"w{z0:02d}" / "ct.zarr"), mode="w", shape=(z1 - z0,) + A.shape, chunks=(z1 - z0, 128, 128),
                            dtype=np.uint8, zarr_format=2)[:] = vol[z0:z1]
            np.save(tmp / "label.npy", lab & sup); np.save(tmp / "support.npy", sup)
            meta.update(n_support=int(sup.sum()), n_ink=int((lab & sup).sum()))
            json.dump(dict(depth=D, production_window=[z0, z1], windows={f"w{z0:02d}": dict(source_planes=[z0, z1])}), open(tmp / "prep.json", "w"), indent=1)
        json.dump(meta, open(tmp / "meta.json", "w"), indent=1, default=float)
        if dest.exists(): shutil.rmtree(dest)
        tmp.rename(dest)


if __name__ == "__main__":
    main(sys.argv[1:] or ["w00", "ag144", "ag174"])
