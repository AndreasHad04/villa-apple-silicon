"""ARM Y, stage 1: the PUBLISHED 2.403 um PHerc0841 surface volumes at level 2.

These are the 109-layer arrays liliandevarieux sliced on villa #1867, as opposed
to the organisers' 65-plane 4.681 um renders ARM X used. Pre-registration:
results/ys/PREREGISTRATION_ARMY.md.

    env/bin/python3 ops/ys_fetch.py [w00 ag144 ag174]

Writes data/ys/<seg>/ raw.zarr (109,H,W uint8, level 2, supervised bbox plus a
margin), label.npy, support.npy, orgpred.npy (the organisers' own published
prediction on the same grid, for C3) and meta.json. Staged in a temp directory
and renamed, so a killed run never leaves a directory that looks finished.
"""
import json, pathlib, shutil, sys, tempfile, time
import numpy as np, s3fs, tifffile, zarr
from zarr.storage import FsspecStore

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUCKET = "vesuvius-challenge-open-data/PHerc0841/segments/"
VOL = "2.403um-0.22m-77keV-volume-20260319124803"
LAB = "ink-labels/2.403um-volume-20260319124803/20260918/"
SEGS = {"w00": "20260220213127-w00",
        "ag144": "20260220214732-auto_grown_20260220144552896",
        "ag174": "20260221022814-auto_grown_20260220174252405"}
MARGIN = 64
zarr.config.set({"async.concurrency": 32})


def open_s3(key, path, fmt):
    st = FsspecStore.from_url("s3://" + BUCKET + key, read_only=True, storage_options={"anon": True})
    return zarr.open_array(store=st, path=path, mode="r", zarr_format=fmt)


def retry(fn, what):
    for k in range(6):
        try:
            return fn()
        except Exception as exc:
            if k == 5: raise
            print(f"retry {what}: {type(exc).__name__} {exc}", flush=True); time.sleep(2 ** k)


def labels(seg):
    ink = retry(lambda: np.asarray(open_s3(seg + "/" + LAB + "inklabels.zarr", "2", 3)[:]), "ink")
    sup = retry(lambda: np.asarray(open_s3(seg + "/" + LAB + "supervision.zarr", "2", 3)[:]), "sup")
    vi, vs = np.unique(ink), np.unique(sup)
    # pre-registered: binary level 2 is used as is; anything else falls back to
    # level-0 pooling, which is not implemented until the data demand it
    assert len(vi) <= 2 and len(vs) <= 2, f"level-2 labels are not binary: ink {vi[:10]} sup {vs[:10]}"
    support = sup > 0
    return (ink > 0) & support, support, dict(ink_values=vi.tolist(), sup_values=vs.tolist())


def org_prediction(seg, shape2):
    fs = s3fs.S3FileSystem(anon=True)
    [key] = [p for p in fs.ls(BUCKET + seg + "/ink-detection") if p.endswith(".tif")]
    with tempfile.TemporaryDirectory() as td:
        local = pathlib.Path(td) / "pred.tif"; retry(lambda: fs.get(key, str(local)), "orgpred")
        p = tifffile.imread(local)
    if p.dtype != np.uint8:
        p = np.rint(np.clip(p.astype(np.float64) / (255.0 if p.max() > 1 else 1.0), 0, 1) * 255).astype(np.uint8)
    H, W = shape2
    if p.shape == (H, W): return p, key.rsplit("/", 1)[-1], "as published"
    assert p.shape[0] // 4 == H and p.shape[1] // 4 == W, (p.shape, shape2)
    q = p[: H * 4, : W * 4].astype(np.float32).reshape(H, 4, W, 4).mean(axis=(1, 3))
    return np.rint(q).astype(np.uint8), key.rsplit("/", 1)[-1], "block mean 4x4 onto level 2"


def fetch(name):
    seg = SEGS[name]; dest = ROOT / "data" / "ys" / name
    if (dest / "meta.json").exists():
        print("have", name, flush=True); return
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    label, support, lv = labels(seg)
    a = open_s3(seg + "/surface-volumes/" + VOL + ".zarr", "2", 2)
    assert a.shape[1:] == support.shape, (a.shape, support.shape)
    H, W = support.shape; ys, xs = np.nonzero(support)
    y0 = max(0, (ys.min() - MARGIN) // 128 * 128); x0 = max(0, (xs.min() - MARGIN) // 128 * 128)
    y1 = min(H, -(-(ys.max() + 1 + MARGIN) // 128) * 128); x1 = min(W, -(-(xs.max() + 1 + MARGIN) // 128) * 128)
    y0, y1, x0, x1 = (int(v) for v in (y0, y1, x0, x1))
    print(f"{name}: level2 {a.shape}, support {int(support.sum())} px, crop y[{y0},{y1}) x[{x0},{x1})", flush=True)
    raw = np.empty((a.shape[0], y1 - y0, x1 - x0), np.uint8); t1 = time.time()
    for r0 in range(y0, y1, 512):
        r1 = min(y1, r0 + 512)
        raw[:, r0 - y0:r1 - y0] = retry(lambda: a[:, r0:r1, x0:x1], f"rows {r0}")
    dt = time.time() - t1
    print(f"{name}: read {raw.nbytes / 1e6:.0f} MB in {dt:.0f} s ({raw.nbytes / 1e6 / max(dt, 1e-9):.1f} MB/s)", flush=True)
    org, org_name, org_how = org_prediction(seg, (H, W))
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=name + ".", dir=dest.parent))
    z = zarr.open_array(str(tmp / "raw.zarr"), mode="w", shape=raw.shape, chunks=(raw.shape[0], 128, 128), dtype=np.uint8, zarr_format=2)
    z[:] = raw
    np.save(tmp / "label.npy", label[y0:y1, x0:x1]); np.save(tmp / "support.npy", support[y0:y1, x0:x1])
    np.save(tmp / "orgpred.npy", org[y0:y1, x0:x1])
    meta = dict(seg=seg, source=f"s3://{BUCKET}{seg}/surface-volumes/{VOL}.zarr level 2", labels=f"s3://{BUCKET}{seg}/{LAB}",
                org_prediction=org_name, org_prediction_regrid=org_how, level2_shape=list(a.shape), bbox_level2=[y0, y1, x0, x1],
                raw_shape=list(raw.shape), label_values=lv, n_support=int(support[y0:y1, x0:x1].sum()),
                n_ink=int(label[y0:y1, x0:x1].sum()), read_s=round(dt, 1), total_s=round(time.time() - t0, 1))
    json.dump(meta, open(tmp / "meta.json", "w"), indent=1)
    if dest.exists(): shutil.rmtree(dest)
    tmp.rename(dest)
    print(f"WROTE {name} raw {raw.shape} support {meta['n_support']} ink {meta['n_ink']} in {meta['total_s']} s", flush=True)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(SEGS)):
        fetch(n)
    print("DONE", flush=True)
