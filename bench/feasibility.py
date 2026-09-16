"""Feasibility probe for ARM A (Apple Silicon ink inference).

Answers, from source and never from assumption:
  1. does MPS exist and what dtypes does it refuse
  2. can the surface-volume zarr be read ANONYMOUSLY from the open bucket
  3. what shape / dtype / size is one segment's surface volume
  4. does the organisers' own published prediction exist for the same segment
Writes results/feasibility.json. No model is downloaded here.
"""
import json, os, sys, time, pathlib

R = pathlib.Path(__file__).resolve().parent.parent
OUT = R / "results" / "feasibility.json"
BUCKET = "vesuvius-challenge-open-data"
SEG = "PHerc1667/segments/20240304141531-w013_20240304141531_flatboi"
SURF = f"{SEG}/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr"

res = {"probed": time.strftime("%Y-%m-%dT%H:%M:%S"), "segment": SEG}

import torch
res["torch"] = torch.__version__
res["mps_available"] = bool(torch.backends.mps.is_available())
res["mps_built"] = bool(torch.backends.mps.is_built())

# what MPS refuses. float64 is the expected blocker; measure it here rather
# than assume it.
dt = {}
for name in ("float32", "float16", "bfloat16", "float64"):
    try:
        t = torch.ones(4, 4, dtype=getattr(torch, name), device="mps")
        dt[name] = {"ok": True, "err": None, "sum": float(t.sum().item())}
    except Exception as e:
        dt[name] = {"ok": False, "err": f"{type(e).__name__}: {e}"[:200]}
res["mps_dtypes"] = dt

# autocast on mps: the inference code picks amp_device = cuda or cpu only,
# so whether mps autocast even works decides how that line must be fixed.
try:
    with torch.autocast(device_type="mps", enabled=True):
        a = torch.randn(32, 32, device="mps")
        b = (a @ a)
    res["mps_autocast"] = {"ok": True, "dtype": str(b.dtype)}
except Exception as e:
    res["mps_autocast"] = {"ok": False, "err": f"{type(e).__name__}: {e}"[:200]}

# anonymous zarr read
import s3fs, zarr
t0 = time.time()
fs = s3fs.S3FileSystem(anon=True)
res["anon_s3"] = {}
try:
    # zarr 3 needs an ASYNC fsspec filesystem; a plain S3FileSystem raises
    # "Filesystem needs to support async operations". from_url builds the
    # async one itself, which is the working form.
    store = zarr.storage.FsspecStore.from_url(
        f"s3://{BUCKET}/{SURF}", storage_options={"anon": True})
    g = zarr.open_group(store=store, mode="r")
    arrays = {}
    for k in list(g.array_keys()):
        a = g[k]
        arrays[k] = {"shape": list(a.shape), "dtype": str(a.dtype),
                     "chunks": list(a.chunks),
                     "gb_uncompressed": round(a.nbytes / 1e9, 3)}
    res["anon_s3"] = {"ok": True, "arrays": arrays,
                      "groups": list(g.group_keys()),
                      "open_seconds": round(time.time() - t0, 2)}
except Exception as e:
    res["anon_s3"] = {"ok": False, "err": f"{type(e).__name__}: {e}"[:400]}

# does their published prediction exist for this segment
try:
    listing = fs.ls(f"{BUCKET}/{SEG}/ink-detection", detail=True)
    res["their_prediction"] = [
        {"name": o["name"].split("/")[-1], "mb": round(o["size"] / 1e6, 2)}
        for o in listing if o["type"] == "file"]
except Exception as e:
    res["their_prediction"] = {"err": f"{type(e).__name__}: {e}"[:200]}

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
