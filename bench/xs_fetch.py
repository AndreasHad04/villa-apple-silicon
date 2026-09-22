"""ARM X, stage 1: fetch and pool the CT under each labelled region.

Two kinds of segment, both written to data/xs/<name>/ as
    ct.zarr      plain zarr v2 uint8 array (D, H, W), the model input
    label.npy    bool (H, W), ink
    support.npy  bool (H, W), pixels that are scored
    meta.json    bbox, spacing, sources, and how each array was derived

  render  the organisers' labelled renders in huggingface.co/buckets/scrollprize/
          datasets (ink/...): 65 planes at native spacing, 2D tif labels. Pooled
          2x2x2 by mean (planes 0..63 -> 32), labels by (>=2 of 4 ink), support
          by (all 4 supervised). Pre-registered in PREREGISTRATION.md, ARM X.
  c1      the positive control: public 2.399 um surface volume at XY pyramid
          level 2, centred 84 of 109 planes, mean of 4 -> 21 slices, exactly the
          ink_9um card's recipe. Labels are the ink_9um dataset's 21-slice zarrs,
          annotated at plane 10; support is the validation mask.

Every output is built in a temp dir and renamed into place, so an interrupted
fetch never leaves a half-written segment that a later stage would trust.
"""
import json, os, pathlib, shutil, sys, tempfile, time
import numpy as np, tifffile, zarr
from zarr.storage import FsspecStore

ROOT = pathlib.Path(__file__).resolve().parents[1]
HF = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
zarr.config.set({"async.concurrency": 8})  # 48 drew HTTP 429 from the HF bucket
sys.path.insert(0, str(ROOT / "villa" / "vesuvius" / "src"))

SEGMENTS = {
    # primary set, spacing from meta.json
    "p0841_w00":   dict(kind="render", dir="ink/841/w00", ct="w00.zarr", um=4.681),
    "p0841_ag144": dict(kind="render", dir="ink/841/auto_grown_20260220144552896", ct="auto_grown_20260220144552896.zarr", um=4.681),
    "p0841_ag174": dict(kind="render", dir="ink/841/auto_grown_20260220174252405", ct="auto_grown_20260220174252405.zarr", um=4.681),
    "p0500p2a":    dict(kind="render", dir="ink/unused/500p2a", ct="500p2a.zarr", um=4.32),
    # secondary, spacing not recorded
    "man5_outer3": dict(kind="render", dir="ink/man5/MAN5_outer_3", ct="MAN5_outer_3.zarr", um=None),
    "p0009b":      dict(kind="render", dir="ink/0009b/auto_grown_20250919055754487_inp_hr_2um", ct="auto_grown_20250919055754487_inp_hr_2um.zarr", um=None,
                        label="auto_grown_20250919055754487_inp_hr_2um_inklabels.tif", mask="auto_grown_20250919055754487_inp_hr_2um_supervision_mask.tif"),
    "p0500p2_m1":  dict(kind="render", dir="ink/0500p2/-1", ct="-1.zarr", um=None),
    # C1 positive control
    "c1_p0139_w016": dict(kind="c1", seg="PHerc0139/segments/20250108000004-w029_2025010827",
                          vol="2.399um-0.22m-78keV-volume-20260102150214.zarr",
                          labels="ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0139-w016/pherc0139-w016"),
}


def open_remote(url, path="0"):
    st = FsspecStore.from_url(url, read_only=True)
    return zarr.open_array(store=st, path=path, mode="r", zarr_format=2)


# HF's anonymous quota, read from its own headers 2026-09-22: ratelimit-policy
# "fixed window";"resolvers";q=3000;w=300. 8/s stays under 10/s with margin.
RATE = 8.0
_T0, _N = time.time(), 0


def pace(n):
    global _N
    _N += n
    ahead = _N / RATE - (time.time() - _T0)
    if ahead > 0: time.sleep(ahead)


def backoff(k, exc):
    wait = 130 if "429" in str(exc) else 15 * 2 ** k  # a 429 means wait out the 300 s window
    print(f"  retry after {type(exc).__name__}: {str(exc)[:90]} (sleep {wait}s)", flush=True); time.sleep(wait)


def read_strips(a, planes, y0, y1, x0, x1, rows=512, hf=True):
    out = np.empty((planes.stop - planes.start, y1 - y0, x1 - x0), np.uint8)
    cy, cx = a.chunks[1], a.chunks[2]
    for s in range(y0, y1, rows):
        e = min(y1, s + rows)
        if hf: pace(((e - 1) // cy - s // cy + 1) * ((x1 - 1) // cx - x0 // cx + 1))
        for k in range(6):  # transient HTTP truncation is a known failure (villa #1666)
            try:
                out[:, s - y0:e - y0] = a[planes, s:e, x0:x1]; break
            except Exception as exc:
                if k == 5: raise
                backoff(k, exc)
        print(f"  rows {e - y0}/{y1 - y0}", flush=True)
    return out


def pool2(v):
    d, h, w = v.shape
    v = v[: d - d % 2, : h - h % 2, : w - w % 2].astype(np.uint16)
    s = v.reshape(d // 2, 2, h // 2, 2, w // 2, 2).sum(axis=(1, 3, 5))
    return ((s + 4) // 8).astype(np.uint8)


def write_segment(name, ct, label, support, meta):
    dest = ROOT / "data" / "xs" / name
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=name + ".", dir=dest.parent))
    z = zarr.open_array(str(tmp / "ct.zarr"), mode="w", shape=ct.shape, chunks=(ct.shape[0], 128, 128), dtype=np.uint8, zarr_format=2)
    z[:] = ct
    np.save(tmp / "label.npy", label); np.save(tmp / "support.npy", support)
    meta.update(ct_shape=list(ct.shape), label_shape=list(label.shape), n_support=int(support.sum()), n_ink=int((label & support).sum()))
    json.dump(meta, open(tmp / "meta.json", "w"), indent=1)
    if dest.exists(): shutil.rmtree(dest)
    tmp.rename(dest)
    print(f"WROTE {name} ct {ct.shape} support {int(support.sum())} ink {int((label & support).sum())}", flush=True)


def fetch_render(name, s, margin=128):
    d = s["dir"]; lab_name = s.get("label") or s["ct"].replace(".zarr", "_inklabels.tif")
    mask_name = s.get("mask") or s["ct"].replace(".zarr", "_supervision_mask.tif")
    L = tifffile.imread(ROOT / "data" / "xs_labels" / d / lab_name); M = tifffile.imread(ROOT / "data" / "xs_labels" / d / mask_name)
    L = L if L.ndim == 2 else L[..., 0]; M = M if M.ndim == 2 else M[..., 0]
    a = open_remote(HF + d + "/" + s["ct"])
    H, W = min(a.shape[1], L.shape[0]), min(a.shape[2], L.shape[1])
    ys, xs = np.nonzero(M[:H, :W] > 0)
    y0 = max(0, (ys.min() - margin) // 128 * 128); x0 = max(0, (xs.min() - margin) // 128 * 128)
    y1 = min(H, -(-(ys.max() + 1 + margin) // 128) * 128); x1 = min(W, -(-(xs.max() + 1 + margin) // 128) * 128)
    y1 -= (y1 - y0) % 2; x1 -= (x1 - x0) % 2
    print(f"{name}: render {a.shape} labels {L.shape} crop y[{y0},{y1}) x[{x0},{x1}) = {(y1-y0)*(x1-x0)*a.shape[0]/1e9:.2f} GB", flush=True)
    t = time.time(); pooled = np.empty((a.shape[0] // 2, (y1 - y0) // 2, (x1 - x0) // 2), np.uint8); nbytes = 0
    for s0 in range(y0, y1, 512):  # pool strip by strip: the whole crop can exceed 10 GB
        s1 = min(y1, s0 + 512); strip = read_strips(a, slice(0, a.shape[0]), s0, s1, x0, x1, rows=512)
        pooled[:, (s0 - y0) // 2:(s1 - y0) // 2] = pool2(strip); nbytes += strip.nbytes
    dt = time.time() - t
    lab = (L[y0:y1, x0:x1] > 0).reshape((y1 - y0) // 2, 2, (x1 - x0) // 2, 2).sum(axis=(1, 3)) >= 2
    sup = (M[y0:y1, x0:x1] > 0).reshape((y1 - y0) // 2, 2, (x1 - x0) // 2, 2).all(axis=(1, 3))
    write_segment(name, pooled, lab, sup, dict(
        kind="render", source=HF + d, native_um=s["um"], pooled_um=(None if s["um"] is None else 2 * s["um"]),
        bbox_native=[int(y0), int(y1), int(x0), int(x1)], render_shape=list(a.shape), label_shape=list(L.shape),
        pooling="2x2x2 mean of planes 0..63; label >=2 of 4 ink; support all 4 supervised",
        labelled_plane_pooled=16, fetch_s=round(dt, 1), fetch_MBps=round(nbytes / 1e6 / dt, 1)))


def fetch_c1(name, s, margin=64):
    lab = open_remote(HF + s["labels"] + "_inklabels.zarr"); val = open_remote(HF + s["labels"] + "_validation_mask.zarr")
    sup_all = open_remote(HF + s["labels"] + "_supervision_mask.zarr")
    # list the validation mask's STORED chunks (tree API), so only its bbox is read:
    # reading plane 10 blind costs one resolver request per chunk, 3135 per array
    import urllib.request, urllib.parse, re as _re
    url = "https://huggingface.co/api/buckets/scrollprize/datasets/tree/" + urllib.parse.quote(s["labels"] + "_validation_mask.zarr/0") + "?recursive=true"
    keys = []
    while url:
        r = urllib.request.urlopen(url, timeout=60); keys += [(x["path"].rsplit("/", 1)[-1], x["size"]) for x in json.load(r) if x["type"] == "file"]
        m = _re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", "")); url = m.group(1) if m else None
    # this zarr STORES every chunk, zeros included (3135 of 3135), so stored is not
    # informative; an all-zero chunk compresses to one fixed small size, anything
    # larger holds validation pixels
    ch = [(k, sz) for k, sz in keys if k.count(".") == 2]; empty = min(sz for _, sz in ch)
    cyx = np.array([[int(k.split(".")[1]), int(k.split(".")[2])] for k, sz in ch if sz > empty])
    cy0, cx0 = cyx.min(0); cy1, cx1 = cyx.max(0) + 1
    print(f"{name}: {len(ch)} stored chunks, {len(cyx)} non-empty (empty size {empty} B), rows {cy0}-{cy1} cols {cx0}-{cx1}", flush=True)
    by0, by1, bx0, bx1 = cy0 * 128, cy1 * 128, cx0 * 128, cx1 * 128
    def plane10(arr):
        for k in range(6):
            try:
                full = np.zeros(arr.shape[1:], bool)
                for r0 in range(by0, by1, 512):  # strips: one burst must stay far below the 3000 quota
                    r1 = min(by1, r0 + 512); pace(-(-(r1 - r0) // 128) * (cx1 - cx0))
                    full[r0:r1, bx0:bx1] = np.asarray(arr[10, r0:r1, bx0:bx1]) > 0
                return full
            except Exception as exc:
                if k == 5: raise
                backoff(k, exc)
    V = plane10(val); Lz = plane10(lab); Sz = plane10(sup_all)
    print(f"{name}: labels read, validation px {int(V.sum())}", flush=True)
    ys, xs = np.nonzero(V)
    a = open_remote(S3 + s["seg"] + "/surface-volumes/" + s["vol"], path="2")
    assert a.shape[1:] == V.shape, (a.shape, V.shape)
    H, W = V.shape
    y0 = max(0, (ys.min() - margin) // 128 * 128); x0 = max(0, (xs.min() - margin) // 128 * 128)
    y1 = min(H, -(-(ys.max() + 1 + margin) // 128) * 128); x1 = min(W, -(-(xs.max() + 1 + margin) // 128) * 128)
    from vesuvius.ink_detection.preprocessing import prepare_9um_isotropic_input as P9
    z0, z1 = P9.centered_slice(a.shape[0], P9.INPUT_Z)
    print(f"{name}: level2 {a.shape} production planes [{z0},{z1}) crop y[{y0},{y1}) x[{x0},{x1})", flush=True)
    t = time.time(); raw = read_strips(a, slice(0, a.shape[0]), y0, y1, x0, x1, hf=False); dt = time.time() - t
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="c1src.", dir=ROOT / "data" / "xs"))
    zarr.open_array(str(scratch / "src.zarr"), mode="w", shape=raw.shape, chunks=(raw.shape[0], 128, 128), dtype=np.uint8, zarr_format=2)[:] = raw
    out = P9.prepare_isotropic_input(str(scratch / "src.zarr"), scratch / "prep.zarr", level="2", workers=4)
    node = zarr.open(str(out), mode="r"); arr = node if hasattr(node, "shape") else node[sorted(node.array_keys())[0]]
    ct = np.asarray(arr[:]); shutil.rmtree(scratch)
    assert ct.shape == (21, y1 - y0, x1 - x0), ct.shape
    write_segment(name, ct, Lz[y0:y1, x0:x1], V[y0:y1, x0:x1], dict(  # validation pixels ARE the held-out set; supervision marks TRAINING pixels and is disjoint from them
        kind="c1", source=S3 + s["seg"] + "/surface-volumes/" + s["vol"] + " level 2", labels=HF + s["labels"],
        pooled_um=9.596, bbox_level2=[int(y0), int(y1), int(x0), int(x1)], planes=[int(z0), int(z1)],
        pooling="villa production prepare_9um_isotropic_input (" + P9.FORMAT_TAG + ") run on the level-2 crop; support = the validation mask at plane 10 (disjoint from supervision by design)",
        n_validation_px=int(V.sum()), n_validation_and_supervised=int((V & Sz).sum()), fetch_s=round(dt, 1)))


if __name__ == "__main__":
    names = sys.argv[1:] or list(SEGMENTS)
    for n in names:
        if (ROOT / "data" / "xs" / n / "meta.json").exists():
            print("have", n, flush=True); continue
        (ROOT / "data" / "xs").mkdir(parents=True, exist_ok=True)
        s = SEGMENTS[n]
        (fetch_c1 if s["kind"] == "c1" else fetch_render)(n, s)
    print("DONE", flush=True)
