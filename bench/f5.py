"""ARM F, F5: the PHerc0841-fitted transform applied unchanged to PHerc0139 held-out segments.
Pre-registration: results/fs/PREREGISTRATION_ARMF.md (amendment F5). Reuses ARM Y's fetch,
villa's production recipe (via ys_prep.prepare), ARM Z's fetch/Z-ALIGN, ARM F's transform
and ARM X's inference path, re-pointed at PHerc0139 by module constants, under data/f5/.

    env/bin/python3 ops/f5.py fetch | make | infer | score
"""
import json, os, pathlib, shutil, sys, tempfile, time
import numpy as np, zarr
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "ops"))
F5 = ROOT / "data" / "f5"
SEGS = {"w030": "20250108000005-w030_2025010818", "w045": "20260126000000-w045_2026012619", "w043": "20260112000000-w043_2026011217"}
VOLP = "2.399um-0.22m-78keV-volume-20260102150214"; VOLN = "9.362um-1.2m-113keV-volume-20250728140407"


def patch():
    import ys_fetch as YF, ys_prep as YP, zs_fetch as ZF, s3fs, tempfile as tf, tifffile
    YF.BUCKET = "vesuvius-challenge-open-data/PHerc0139/segments/"; YF.VOL = VOLP
    YF.LAB = "ink-labels/2.399um-volume-20260102150214/20260918/"; YF.SEGS = SEGS; YF.ROOT = F5
    YP.ROOT = F5; ZF.ROOT = F5; ZF.VOL9 = VOLN; ZF.S = 9.362 / 9.596; ZF.SEARCH = 32; ZF.M = 32

    def org_prediction(seg, shape2):  # the 2.399 um canonical prediction, block-mean 4x4 onto level 2
        fs = s3fs.S3FileSystem(anon=True)
        [key] = [p for p in fs.ls(YF.BUCKET + seg + "/ink-detection") if p.endswith(".tif") and "2.399um" in p]
        with tf.TemporaryDirectory() as td:
            local = pathlib.Path(td) / "pred.tif"; YF.retry(lambda: fs.get(key, str(local)), "orgpred"); p = tifffile.imread(local)
        if p.dtype != np.uint8: p = np.rint(np.clip(p.astype(np.float64) / (255.0 if p.max() > 1 else 1.0), 0, 1) * 255).astype(np.uint8)
        H, W = shape2; q = p[: H * 4, : W * 4].astype(np.float32).reshape(H, 4, W, 4).mean(axis=(1, 3))
        return np.rint(q).astype(np.uint8), key.rsplit("/", 1)[-1], "block mean 4x4 onto level 2"
    YF.org_prediction = org_prediction
    return YF, YP, ZF


def fetch(only=None):
    YF, YP, ZF = patch()
    for name in ([only] if only else SEGS):
        YF.fetch(name)
        sd = F5 / "data" / "ys" / name
        if not (sd / "w13" / "ct.zarr").exists():
            raw = np.asarray(zarr.open_array(str(sd / "raw.zarr"), mode="r")[:]); assert raw.shape[0] == 109, raw.shape
            ct, attrs = YP.prepare(raw, sd); assert attrs["source_z_slice"] == [13, 97], attrs
            YP.write_ct(sd / "w13", ct)
            json.dump(dict(depth=109, production_window=[13, 97], windows={"w13": dict(source_planes=[13, 97])}), open(sd / "prep.json", "w"), indent=1)
        ZF.main([name])
        print("F5 fetched", name, json.load(open(F5 / "data" / "zs" / name / "meta.json"))["zalign"]["rule"], flush=True)


def make():
    import fs_make as M
    fits = [M.fit_pair(s) for s in M.SEGS]; g = M.gain_curve(fits); fz, gz = M.z_gain_curve(fits)
    data = {s: M.load(s) for s in M.SEGS}
    chains = {"NSH": lambda v, px, dz: M.apply_inplane(v, g, px),
              "NSDH": lambda v, px, dz: M.apply_depth(M.apply_inplane(v, g, px), fz, gz, dz),
              "NDH": lambda v, px, dz: M.apply_depth(v.astype(np.float64), fz, gz, dz),
              "ND": lambda v, px, dz: M.apply_depth(v.astype(np.float64), fz, gz, dz)}  # ND: exploratory, no intensity map
    tgt = np.concatenate([data[s][1][:, data[s][3]].ravel() for s in M.SEGS])
    rec = dict(g_inplane=g.tolist(), fz=fz.tolist(), g_depth=gz.tolist(), fitted_on=["PHerc0841 " + s for s in M.SEGS], segments={})
    for cond, ch in chains.items():
        if cond != "ND": src = np.concatenate([ch(data[s][0], M.PXN, M.DZN)[:, data[s][2]].ravel() for s in M.SEGS]); qs, qt = M.quantile_map(src, tgt)
        for name in SEGS:
            zd = F5 / "data" / "zs" / name
            if not (zd / "prep.json").exists(): rec["segments"][name] = "excluded by Z-ALIGN"; continue
            n = np.asarray(zarr.open_array(str(zd / "w04" / "ct.zarr"), mode="r")[:])
            x = ch(n, 9.362, 9.362); ct = M.to_uint8(x if cond == "ND" else np.interp(x, qs, qt))
            dest = F5 / "data" / "fs" / name / cond; dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = pathlib.Path(tempfile.mkdtemp(prefix=cond + ".", dir=dest.parent))
            zarr.open_array(str(tmp / "ct.zarr"), mode="w", shape=ct.shape, chunks=(21, 128, 128), dtype=np.uint8, zarr_format=2)[:] = ct
            if dest.exists(): shutil.rmtree(dest)
            tmp.rename(dest); rec["segments"].setdefault(name, []).append(cond) if isinstance(rec["segments"].get(name, []), list) else None
            print("F5", cond, name, ct.shape, flush=True)
    json.dump(rec, open(ROOT / "results" / "fs" / "f5_transform.json", "w"), indent=1)


def ct_path(name, cond):
    return {"N0": F5 / "data" / "zs" / name / "w04" / "ct.zarr", "P0": F5 / "data" / "ys" / name / "w13" / "ct.zarr",
            "NSH": F5 / "data" / "fs" / name / "NSH" / "ct.zarr", "NSDH": F5 / "data" / "fs" / name / "NSDH" / "ct.zarr",
            "NDH": F5 / "data" / "fs" / name / "NDH" / "ct.zarr", "ND": F5 / "data" / "fs" / name / "ND" / "ct.zarr"}[cond]


def infer():
    import torch, xs_infer as XI
    for cond in ("N0", "NDH", "NSDH", "NSH", "P0", "ND"):
        for name in SEGS:
            ct = ct_path(name, cond)
            if not ct.exists(): print("skip", name, cond, flush=True); continue
            D, H, W = zarr.open_array(str(ct), mode="r").shape; assert D == 21
            for ck in XI.checkpoints("primary"):
                for d in ("forward", "reverse"):
                    out = ROOT / "results" / "fs" / "f5pred" / name / cond / f"{ck.parent.name}_{ck.stem}_{d}.tif"
                    if XI.valid_tiff(out, (H, W)): continue
                    out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(f"{out.stem}.partial{os.getpid()}.tif")
                    t = time.time(); XI.INF.main([str(ct), str(ck), str(tmp), "--direction", d, "--no-compile", "--num-workers", "0", "--batch-size", "32"])
                    assert XI.valid_tiff(tmp, (H, W)); os.replace(tmp, out); print(f"DONE F5 {cond} {name} {d} {time.time() - t:.1f}s", flush=True)


def score():
    import tifffile, ys_score as YS
    out = dict(segments={})
    for name in SEGS:
        rec = {}
        for cond in ("N0", "NSH", "NSDH", "NDH", "ND", "P0"):
            d = F5 / "data" / ("ys" if cond == "P0" else "zs") / name
            if not (d / "label.npy").exists(): continue
            label, support = np.load(d / "label.npy"), np.load(d / "support.npy"); rows = {}
            for f in sorted((ROOT / "results" / "fs" / "f5pred" / name / cond).glob("*.tif")):
                if ".partial" in f.name: continue
                key, dd = f.stem.rsplit("_", 1); rows.setdefault(key, {})[dd] = YS.metrics(tifffile.imread(f), label, support)
            for r in rows.values(): YS.finish(r)
            if YS.PRIMARY in rows and "chosen" in rows[YS.PRIMARY]: rec[cond] = rows[YS.PRIMARY]
        out["segments"][name] = rec
    a = lambda n, c: out["segments"][n].get(c, {}).get("auc_chosen")
    clean = [n for n in ("w030", "w045") if a(n, "N0") is not None]
    for key, cond in (("F5", "NSH"), ("F5b", "NSDH"), ("F5c", "NDH")):
        ok = [n for n in clean if a(n, cond) is not None]
        if not ok: continue
        m = float(np.median([a(n, cond) - a(n, "N0") for n in ok]))
        out[key] = dict(condition=cond, median_gain=m, segments=ok, per_segment={n: a(n, cond) - a(n, "N0") for n in ok},
                        verdict="transfers" if m >= 0.020 else "does not transfer" if m <= 0.005 else "small, as measured")
    out["auc"] = {n: {c: a(n, c) for c in ("N0", "NSH", "NSDH", "NDH", "ND", "P0")} for n in SEGS}
    json.dump(out, open(ROOT / "results" / "fs" / "f5_scores.json", "w"), indent=1, default=float)
    for k in ("F5", "F5b", "F5c"): print(k, json.dumps(out.get(k), default=float))
    for n in SEGS: print(n, {c: (round(a(n, c), 4) if a(n, c) is not None else None) for c in ("N0", "NSH", "NSDH", "NDH", "ND", "P0")},
                         {c: out["segments"][n].get(c, {}).get("chosen") for c in ("N0", "NSH", "NSDH", "NDH", "P0")})


if __name__ == "__main__":
    if sys.argv[1] == "fetch1": fetch(sys.argv[2])
    else: dict(fetch=fetch, make=make, infer=infer, score=score)[sys.argv[1]]()
