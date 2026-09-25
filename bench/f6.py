"""ARM F, F6 (amendment 4): the published depth filter on two more held-out scrolls, PHerc0009B (8.64 um 116 keV)
and PHerc0500P2 (9.362 um 113 keV). N0, NDH, ND exactly as bench/depth_sharpen_9um.py computes them.

    env/bin/python3 ops/f6.py fetch | infer | score
"""
import importlib.util, json, os, pathlib, shutil, sys, tempfile, time
import numpy as np, zarr
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "ops"))
import ys_fetch as YF, zs_fetch as ZF  # noqa: E402
D6 = ROOT / "data" / "f6"
SEGS = {"p0009b": dict(scroll="PHerc0009B", seg="20250919125754-auto_grown_20250919055754487_inp_hr",
                       fine="2.401um-0.35m-77keV-volume-20250820154339", fine_um=2.401, nat="8.64um-1.2m-116keV-volume-20250521125136", nat_um=8.64,
                       lab="ink-labels/2.401um-volume-20250820154339/20260918/"),
        "p0500p2": dict(scroll="PHerc0500P2", seg="20250825181859--1",
                        fine="2.215um-0.4m-111keV-volume-20250526151718", fine_um=2.215, nat="9.362um-1.2m-113keV-volume-20250820143440", nat_um=9.362,
                        lab="ink-labels/2.215um-volume-20250526151718/20260918/")}
_TOOL = ROOT / "public" / "bench" / "depth_sharpen_9um.py"
if not _TOOL.exists(): _TOOL = pathlib.Path(__file__).with_name("depth_sharpen_9um.py")  # the published layout keeps it beside this script
_sp = importlib.util.spec_from_file_location("ds9", _TOOL)
T = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(T)
MARGIN = 64


def write(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = pathlib.Path(tempfile.mkdtemp(prefix=path.name + ".", dir=path.parent))
    zarr.open_array(str(tmp / "a.zarr"), mode="w", shape=arr.shape, chunks=(arr.shape[0], 128, 128), dtype=np.uint8, zarr_format=2)[:] = arr
    if path.exists(): shutil.rmtree(path)
    (tmp / "a.zarr").rename(path); shutil.rmtree(tmp, ignore_errors=True)


def relabel(label, support, box, resample, use, S):
    """Labels are on the FULL level-2 canvas; resample() takes box-relative arrays (as ARM Z's does), so crop first.
    Count check: the resampled support must hold the box's support scaled by the pixel-area ratio, within 20%."""
    y0, y1, x0, x1 = box; lb, sb = label[y0:y1, x0:x1], support[y0:y1, x0:x1]
    lab, _ = resample(lb, *use, fill=False); sp, _ = resample(sb, *use, fill=False)
    want = sb.sum() / (S * S); got = sp.sum()
    assert 0.8 * want <= got <= 1.2 * want, f"resampled support {got} px against {want:.0f} expected: labels misplaced"
    return lab & sp, sp


def fix_labels():
    """Recompute label.npy/support.npy from the level-2 labels without re-reading any volume (the inputs are unaffected)."""
    for name, c in SEGS.items():
        d6 = D6 / name; m = json.load(open(d6 / "meta.json"))
        YF.BUCKET = f"vesuvius-challenge-open-data/{c['scroll']}/segments/"
        ink = YF.retry(lambda: np.asarray(YF.open_s3(c["seg"] + "/" + c["lab"] + "inklabels.zarr", "2", 3)[:]), "ink")
        sup = YF.retry(lambda: np.asarray(YF.open_s3(c["seg"] + "/" + c["lab"] + "supervision.zarr", "2", 3)[:]), "sup")
        support = sup > 0; label = (ink > 0) & support
        ys, xs = np.nonzero(support); H2, W2 = support.shape
        y0, y1 = max(0, ys.min() - MARGIN), min(H2, ys.max() + 1 + MARGIN); x0, x1 = max(0, xs.min() - MARGIN), min(W2, xs.max() + 1 + MARGIN)
        i0, i1, j0, j1 = m["crop"]; S = m["S"]; use = tuple(m["zalign"]["used_offset"])

        def resample(img, dy=0, dx=0, fill=0):
            ii = np.rint((np.arange(i0, i1) + dy + 0.5) * S - 0.5).astype(int) - y0
            jj = np.rint((np.arange(j0, j1) + dx + 0.5) * S - 0.5).astype(int) - x0
            okr, okc = (ii >= 0) & (ii < img.shape[0]), (jj >= 0) & (jj < img.shape[1])
            out = np.full((len(ii), len(jj)), fill, dtype=img.dtype); out[np.ix_(okr, okc)] = img[np.ix_(ii[okr], jj[okc])]
            return out, np.outer(okr, okc)
        lab, sp = relabel(label, support, (y0, y1, x0, x1), resample, use, S)
        np.save(d6 / "label.npy", lab); np.save(d6 / "support.npy", sp)
        m.update(n_support=int(sp.sum()), n_ink=int(lab.sum()), label_box_level2=[int(y0), int(y1), int(x0), int(x1)], labels_fixed="2026-09-26 box-relative crop before resample")
        json.dump(m, open(d6 / "meta.json", "w"), indent=1, default=float)
        print(name, "support", int(sp.sum()), "ink", int(lab.sum()), "rate", round(float(lab.sum() / sp.sum()), 4), flush=True)


def fetch():
    ZF.SEARCH = 32
    for name, c in SEGS.items():
        dest = D6 / name
        if (dest / "meta.json").exists(): print("have", name, flush=True); continue
        YF.BUCKET = f"vesuvius-challenge-open-data/{c['scroll']}/segments/"
        ink = YF.retry(lambda: np.asarray(YF.open_s3(c["seg"] + "/" + c["lab"] + "inklabels.zarr", "2", 3)[:]), "ink")
        sup = YF.retry(lambda: np.asarray(YF.open_s3(c["seg"] + "/" + c["lab"] + "supervision.zarr", "2", 3)[:]), "sup")
        assert len(np.unique(ink)) <= 2 and len(np.unique(sup)) <= 2, "labels not binary at level 2"
        support = sup > 0; label = (ink > 0) & support
        ys, xs = np.nonzero(support); H2, W2 = support.shape
        y0, y1 = max(0, ys.min() - MARGIN), min(H2, ys.max() + 1 + MARGIN); x0, x1 = max(0, xs.min() - MARGIN), min(W2, xs.max() + 1 + MARGIN)
        f = YF.open_s3(c["seg"] + "/surface-volumes/" + c["fine"] + ".zarr", "2", 2); Df = f.shape[0]; z0f = int(np.ceil((Df - 21) / 2))
        assert f.shape[1:] == support.shape, (f.shape, support.shape)
        t = time.time(); m2 = YF.retry(lambda: np.asarray(f[z0f:z0f + 21, y0:y1, x0:x1]), "fine").astype(np.float64).mean(0)
        print(f"{name}: fine L2 {f.shape}, planes [{z0f},{z0f + 21}) box y[{y0},{y1}) x[{x0},{x1}) read {time.time() - t:.0f}s", flush=True)
        S = c["nat_um"] / (4 * c["fine_um"])  # level-2 pixels per native pixel
        a = YF.open_s3(c["seg"] + "/surface-volumes/" + c["nat"] + ".zarr", "0", 2); D, Hn, Wn = a.shape
        z0, z1 = int(np.ceil((D - 21) / 2)), int(np.ceil((D - 21) / 2)) + 21
        i0 = max(0, int(np.floor(y0 / S)) - 32); i1 = min(Hn, int(np.ceil(y1 / S)) + 32)
        j0 = max(0, int(np.floor(x0 / S)) - 32); j1 = min(Wn, int(np.ceil(x1 / S)) + 32)
        t = time.time(); vol = YF.retry(lambda: np.asarray(a[z0:z1, i0:i1, j0:j1]), "native"); print(f"{name}: native {a.shape} window [{z0},{z1}) read {time.time() - t:.0f}s", flush=True)

        def resample(img, dy=0, dx=0, fill=0):
            ii = np.rint((np.arange(i0, i1) + dy + 0.5) * S - 0.5).astype(int) - y0
            jj = np.rint((np.arange(j0, j1) + dx + 0.5) * S - 0.5).astype(int) - x0
            okr, okc = (ii >= 0) & (ii < img.shape[0]), (jj >= 0) & (jj < img.shape[1])
            out = np.full((len(ii), len(jj)), fill, dtype=img.dtype); out[np.ix_(okr, okc)] = img[np.ix_(ii[okr], jj[okc])]
            return out, np.outer(okr, okc)
        A = vol.astype(np.float64).mean(0); B, inb = resample(m2, fill=0.0)
        g = ZF.ncc_grid(ZF.hp(A), ZF.hp(B), inb & (A > 0) & (B > 0))
        pk = np.unravel_index(np.nanargmax(g), g.shape); dy, dx = int(pk[0] - ZF.SEARCH), int(pk[1] - ZF.SEARCH)
        peak, med = float(g[pk]), float(np.nanmedian(np.abs(g)))
        if abs(dy) <= ZF.PEAK_TOL and abs(dx) <= ZF.PEAK_TOL: use, rule = (0, 0), "peak within 2 px: zero offset"
        elif peak >= ZF.PEAK_RATIO * med: use, rule = (dy, dx), "peak shift applied"
        else: use, rule = None, "EXCLUDED: no identifiable peak"
        meta = dict(**c, native_shape=[D, Hn, Wn], window=[z0, z1], crop=[i0, i1, j0, j1], S=S, zalign=dict(peak_shift=[dy, dx], peak_ncc=peak, median_abs_ncc=med,
                    ratio=peak / med if med else None, ncc_at_zero=float(g[ZF.SEARCH, ZF.SEARCH]), rule=rule, used_offset=use))
        print(name, "Z-ALIGN", meta["zalign"], flush=True)
        dest.mkdir(parents=True, exist_ok=True)
        if use is not None:
            lab, sp = relabel(label, support, (y0, y1, x0, x1), resample, use, S)
            np.save(dest / "label.npy", lab & sp); np.save(dest / "support.npy", sp)
            write(dest / "N0" / "ct.zarr", vol)
            write(dest / "NDH" / "ct.zarr", T.sharpen(vol, c["nat_um"], True)); write(dest / "ND" / "ct.zarr", T.sharpen(vol, c["nat_um"], False))
            meta.update(n_support=int(sp.sum()), n_ink=int((lab & sp).sum()))
        json.dump(meta, open(dest / "meta.json", "w"), indent=1, default=float)


def infer():
    import torch, xs_infer as XI
    for cond in ("N0", "NDH", "ND"):
        for name in SEGS:
            ct = D6 / name / cond / "ct.zarr"
            if not ct.exists(): print("skip", name, cond, flush=True); continue
            Dd, H, W = zarr.open_array(str(ct), mode="r").shape; assert Dd == 21
            for ck in XI.checkpoints("primary"):
                for d in ("forward", "reverse"):
                    out = ROOT / "results" / "fs" / "f6pred" / name / cond / f"{ck.parent.name}_{ck.stem}_{d}.tif"
                    if XI.valid_tiff(out, (H, W)): continue
                    out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(f"{out.stem}.partial{os.getpid()}.tif")
                    t = time.time(); XI.INF.main([str(ct), str(ck), str(tmp), "--direction", d, "--no-compile", "--num-workers", "0", "--batch-size", "32"])
                    assert XI.valid_tiff(tmp, (H, W)); os.replace(tmp, out); print(f"DONE F6 {cond} {name} {d} {time.time() - t:.1f}s", flush=True)


def score():
    import tifffile, ys_score as YS
    out = dict(segments={})
    for name in SEGS:
        d6 = D6 / name; rec = dict(meta=json.load(open(d6 / "meta.json")))
        if (d6 / "label.npy").exists():
            label, support = np.load(d6 / "label.npy"), np.load(d6 / "support.npy")
            for cond in ("N0", "NDH", "ND"):
                rows = {}
                for f in sorted((ROOT / "results" / "fs" / "f6pred" / name / cond).glob("*.tif")):
                    if ".partial" in f.name: continue
                    key, dd = f.stem.rsplit("_", 1); rows.setdefault(key, {})[dd] = YS.metrics(tifffile.imread(f), label, support)
                for r in rows.values(): YS.finish(r)
                if YS.PRIMARY in rows and "chosen" in rows[YS.PRIMARY]: rec[cond] = rows[YS.PRIMARY]
        out["segments"][name] = rec
    a = lambda n, c: out["segments"][n].get(c, {}).get("auc_chosen")
    ok = [n for n in SEGS if a(n, "N0") is not None and a(n, "NDH") is not None]
    if ok:
        m = float(np.mean([a(n, "NDH") - a(n, "N0") for n in ok]))
        out["F6"] = dict(mean_gain=m, segments=ok, per_segment={n: a(n, "NDH") - a(n, "N0") for n in ok},
                         verdict="transfers" if m >= 0.020 else "does not transfer" if m <= 0.005 else "small, as measured")
    out["auc"] = {n: {c: a(n, c) for c in ("N0", "NDH", "ND")} for n in SEGS}
    json.dump(out, open(ROOT / "results" / "fs" / "f6_scores.json", "w"), indent=1, default=float)
    print(json.dumps(out.get("F6"), default=float))
    for n in SEGS: print(n, {c: (round(a(n, c), 4) if a(n, c) is not None else None) for c in ("N0", "NDH", "ND")},
                         {c: out["segments"][n].get(c, {}).get("chosen") for c in ("N0", "NDH", "ND")})


if __name__ == "__main__":
    dict(fetch=fetch, infer=infer, score=score, fix_labels=fix_labels)[sys.argv[1]]()
