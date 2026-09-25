"""ARM F, stage 0 (label free): how does the eligible 9.366 um render of a PHerc0841
segment differ from the 2.403 um render of the same segment?

No labels and no model output are read here. For each ARM Z segment:
  1. fetch all 28 layers of the 9.366 um surface volume over ARM Z's crop
     (ARM Z stored only its 21-layer window), atomically, to data/fs/<seg>/native28.zarr;
  2. per 128 px tile inside the label support, correlate the native depth profile with
     the 2.403 um depth profile of the same pixels (ARM Z's in-plane offset applied),
     the 2.403 um profile box-averaged to 9.366 um layers and shifted by s um,
     s in [-60, 60] at 0.5 um: the best s is that tile's depth offset;
  3. global descriptors of the two production-window inputs: robust-normalised
     histogram quantiles, radial power spectrum of the window mean, and an
     adjacent-layer noise proxy.

    env/bin/python3 ops/fs_profile.py [w00 ag144 ag174]

Writes results/fs/profile_<seg>.json.
"""
import json, pathlib, shutil, sys, tempfile
import numpy as np, zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import ys_fetch as YF  # noqa: E402

VOL9 = "9.366um-1.2m-113keV-volume-20250821151531"
S = 9.366 / 9.612
T = 128
DZN, DZP = 9.366, 2.403
SHIFTS = np.arange(-60.0, 60.0001, 0.5)


def native28(name):
    dest = ROOT / "data" / "fs" / name / "native28.zarr"
    if dest.exists():
        return np.asarray(zarr.open_array(str(dest), mode="r")[:])
    zm = json.load(open(ROOT / "data" / "zs" / name / "meta.json"))
    i0, i1, j0, j1 = zm["crop"]
    a = YF.open_s3(zm["seg"] + "/surface-volumes/" + VOL9 + ".zarr", "0", 2)
    vol = YF.retry(lambda: np.asarray(a[:, i0:i1, j0:j1]), "native28")
    assert vol.shape[0] == 28, vol.shape
    w = np.asarray(zarr.open_array(str(ROOT / "data" / "zs" / name / "w04" / "ct.zarr"), mode="r")[:])
    assert np.array_equal(vol[4:25], w), "re-fetched window differs from ARM Z's stored input"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="n28.", dir=dest.parent))
    zarr.open_array(str(tmp / "a.zarr"), mode="w", shape=vol.shape, chunks=(28, 128, 128), dtype=np.uint8, zarr_format=2)[:] = vol
    (tmp / "a.zarr").rename(dest); shutil.rmtree(tmp, ignore_errors=True)
    return vol


def pooled_index(name):
    """Level-2 (ARM Y crop) row/col index of every native crop pixel, ARM Z's offset applied."""
    zm = json.load(open(ROOT / "data" / "zs" / name / "meta.json")); ym = json.load(open(ROOT / "data" / "ys" / name / "meta.json"))
    i0, i1, j0, j1 = zm["crop"]; y0, _, x0, _ = ym["bbox_level2"]; dy, dx = zm["zalign"]["used_offset"]
    ii = np.rint((np.arange(i0, i1) + dy + 0.5) * S - 0.5).astype(int) - y0
    jj = np.rint((np.arange(j0, j1) + dx + 0.5) * S - 0.5).astype(int) - x0
    return ii, jj


def boxed(profile_p, s):
    """2.403 um profile (109 planes, centre 54) box-averaged over each native layer (28, centre 13.5), shifted by s um."""
    zp = (np.arange(profile_p.size) - (profile_p.size - 1) / 2) * DZP
    out = np.empty(28)
    for k in range(28):
        lo = (k - 14.0) * DZN + s; hi = lo + DZN
        sel = (zp >= lo) & (zp < hi)
        out[k] = profile_p[sel].mean() if sel.any() else np.nan
    return out


def best_shift(prof_n, prof_p):
    best = (np.nan, -2.0)
    for s in SHIFTS:
        b = boxed(prof_p, s); ok = np.isfinite(b)
        if ok.sum() < 20: continue
        r = float(np.corrcoef(prof_n[ok], b[ok])[0, 1])
        if r > best[1]: best = (float(s), r)
    return best


def robust(x):
    x = x.astype(np.float64); lo, hi = np.percentile(x, [1, 99]); x = np.clip(x, lo, hi)
    med = np.median(x); mad = 1.4826 * np.median(np.abs(x - med))
    return (x - med) / (mad if mad > 1e-6 else 1.0)


def radial_power(img):
    img = img - img.mean(); F = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    h, w = img.shape; yy, xx = np.indices((h, w)); fy = (yy - h // 2) / h; fx = (xx - w // 2) / w
    fr = np.sqrt(fy ** 2 + fx ** 2); bins = np.linspace(0, 0.5, 26)
    return [float(F[(fr >= bins[i]) & (fr < bins[i + 1])].mean()) for i in range(25)]


def main(names):
    out_dir = ROOT / "results" / "fs"; out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        nat = native28(name)
        raw = np.asarray(zarr.open_array(str(ROOT / "data" / "ys" / name / "raw.zarr"), mode="r")[:])
        sup = np.load(ROOT / "data" / "zs" / name / "support.npy")
        ii, jj = pooled_index(name)
        H, W = nat.shape[1:]
        tiles = []
        for r0 in range(0, H - T + 1, T):
            for c0 in range(0, W - T + 1, T):
                rs, cs = ii[r0:r0 + T], jj[c0:c0 + T]
                okr = (rs >= 0) & (rs < raw.shape[1]); okc = (cs >= 0) & (cs < raw.shape[2])
                m = sup[r0:r0 + T, c0:c0 + T] & np.outer(okr, okc)
                if m.mean() < 0.5: continue
                pn = nat[:, r0:r0 + T, c0:c0 + T][:, m].mean(1)
                sub = raw[:, rs[okr]][:, :, cs[okc]]
                mm = m[np.ix_(okr, okc)]
                pp = sub[:, mm].mean(1)
                s, r = best_shift(pn, pp)
                tiles.append(dict(r0=r0, c0=c0, frac=round(float(m.mean()), 3), shift_um=s, r=round(r, 4),
                                  peak_layer_native=int(np.argmax(pn)), peak_plane_pooled=int(np.argmax(pp))))
        sh = np.array([t["shift_um"] for t in tiles]); rr = np.array([t["r"] for t in tiles])
        good = rr >= 0.8
        wn = nat[4:25].astype(np.float64)
        wp = np.asarray(zarr.open_array(str(ROOT / "data" / "ys" / name / "w13" / "ct.zarr"), mode="r")[:]).astype(np.float64)
        # same pixels: pooled production input resampled onto the native crop
        okr = (ii >= 0) & (ii < wp.shape[1]); okc = (jj >= 0) & (jj < wp.shape[2])
        both = sup & np.outer(okr, okc)
        wp_on = np.zeros_like(wn); wp_on[:, np.ix_(okr, okc)[0], np.ix_(okr, okc)[1]] = wp[:, ii[okr]][:, :, jj[okc]]
        qs = [1, 5, 25, 50, 75, 95, 99]
        rn, rp = robust(wn[:, both]), robust(wp_on[:, both])
        def noise(v):  # adjacent-layer difference, robust scale, relative to the volume's own robust scale
            d = np.diff(v[:, both], axis=0); med = np.median(d); return float(1.4826 * np.median(np.abs(d - med)) / (1.4826 * np.median(np.abs(v[:, both] - np.median(v[:, both])))))
        # a central square for spectra, inside support
        ys_, xs_ = np.where(both); cy, cx = int(np.median(ys_)), int(np.median(xs_)); hs = 256
        box = (slice(max(0, cy - hs), cy + hs), slice(max(0, cx - hs), cx + hs))
        rec = dict(seg=name, n_tiles=len(tiles), tiles=tiles,
                   shift_um=dict(median=float(np.median(sh)), p10=float(np.percentile(sh, 10)), p90=float(np.percentile(sh, 90)),
                                 median_good=float(np.median(sh[good])) if good.any() else None, n_good=int(good.sum()),
                                 median_r=float(np.median(rr))),
                   hist_quantiles=dict(q=qs, native=[float(np.percentile(rn, q)) for q in qs], pooled=[float(np.percentile(rp, q)) for q in qs]),
                   raw_quantiles=dict(q=qs, native=[float(np.percentile(wn[:, both], q)) for q in qs], pooled=[float(np.percentile(wp_on[:, both], q)) for q in qs]),
                   noise_rel=dict(native=noise(wn), pooled=noise(wp_on)),
                   radial_power=dict(native=radial_power(wn.mean(0)[box]), pooled=radial_power(wp_on.mean(0)[box])),
                   layer_corr_native=[float(np.corrcoef(wn[k][both], wn[k + 1][both])[0, 1]) for k in range(20)],
                   layer_corr_pooled=[float(np.corrcoef(wp_on[k][both], wp_on[k + 1][both])[0, 1]) for k in range(20)])
        json.dump(rec, open(out_dir / f"profile_{name}.json", "w"), indent=1)
        s = rec["shift_um"]
        print(f"{name}: {len(tiles)} tiles, depth shift um median {s['median']:.1f} (p10 {s['p10']:.1f}, p90 {s['p90']:.1f}), "
              f"good(r>=0.8) {s['n_good']} median {s['median_good']}, median r {s['median_r']:.3f}; noise_rel N {rec['noise_rel']['native']:.3f} "
              f"P {rec['noise_rel']['pooled']:.3f}", flush=True)
        print("   robust quantiles N", [round(v, 2) for v in rec["hist_quantiles"]["native"]], "P", [round(v, 2) for v in rec["hist_quantiles"]["pooled"]], flush=True)
        print("   raw quantiles N", [round(v, 1) for v in rec["raw_quantiles"]["native"]], "P", [round(v, 1) for v in rec["raw_quantiles"]["pooled"]], flush=True)
        pn_, pp_ = np.array(rec["radial_power"]["native"]), np.array(rec["radial_power"]["pooled"])
        print("   power ratio N/P by radial bin (normalised to bin 1)", [round(v, 2) for v in (pn_ / pn_[1]) / (pp_ / pp_[1])], flush=True)
        print("   adjacent-layer corr N", [round(v, 3) for v in rec["layer_corr_native"][::4]], "P", [round(v, 3) for v in rec["layer_corr_pooled"][::4]], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["w00", "ag144", "ag174"])
