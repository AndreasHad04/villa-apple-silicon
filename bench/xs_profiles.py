"""ARM X, EXPLORATORY: mean CT intensity per pooled plane inside each segment's
support (every 8th pixel), saved so the write-up can quote it from a file."""
import json, pathlib
import numpy as np, zarr
R = pathlib.Path(__file__).resolve().parents[1]; out = {}
for sd in sorted((R / "data" / "xs").iterdir()):
    if not (sd / "meta.json").exists(): continue
    a = zarr.open_array(str(sd / "ct.zarr"), mode="r"); m = np.load(sd / "support.npy")[::8, ::8]
    sub = np.asarray(a[:, ::8, ::8]).astype(float)
    out[sd.name] = [round(float(sub[z][m].mean()), 1) for z in range(sub.shape[0])]
(R / "results" / "xs" / "depth_profiles.json").write_text(json.dumps(out, indent=1)); print({k: (len(v), max(v), min(v)) for k, v in out.items()})
