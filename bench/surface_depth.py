"""Model-free: where in the 109 layer surface volume does the sheet actually sit?

If these volumes are built by sampling +/-N along the mesh normal, the writing
surface is at the geometric centre and a 62 layer window should be centred
there. This tests that from the VOXELS alone, with no model involved, so it
cannot be circular with the window recovery.
"""
import json, pathlib, sys
import numpy as np, zarr
R = pathlib.Path(__file__).resolve().parent.parent
B = "vesuvius-challenge-open-data"
ZP = ("PHerc1667/segments/20240304141531-w013_20240304141531_flatboi"
      "/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr")
Y0, X0, SZ = 20480, 16384, 1024
st = zarr.storage.FsspecStore.from_url(f"s3://{B}/{ZP}", storage_options={"anon": True})
a = zarr.open_group(store=st, mode="r")["0"]
v = np.asarray(a[:, Y0:Y0+SZ, X0:X0+SZ]).astype(np.float32)
n = v.shape[0]
rows = []
for z in range(n):
    s = v[z]
    nz = s[s > 0]
    rows.append({"z": z, "mean": float(s.mean()),
                 "nonzero_frac": float((s > 0).mean()),
                 "mean_nonzero": float(nz.mean()) if nz.size else 0.0,
                 "std": float(s.std()),
                 # in-plane gradient energy: sharpest where the sheet is in focus
                 "grad": float(np.abs(np.diff(s, axis=0)).mean() + np.abs(np.diff(s, axis=1)).mean())})
arr = {k: np.array([r[k] for r in rows]) for k in ("mean", "std", "grad", "nonzero_frac")}
out = {"volume_layers": n, "crop": [Y0, X0, SZ], "per_layer": rows, "peaks": {}}
for k in ("mean", "std", "grad"):
    x = arr[k]
    # centre of mass of the profile, and the argmax, both reported
    com = float((np.arange(n)*x).sum()/x.sum()) if x.sum() > 0 else float("nan")
    out["peaks"][k] = {"argmax": int(x.argmax()), "centre_of_mass": round(com, 2),
                       "value_at_argmax": float(x.max())}
out["geometric_centre"] = (n-1)/2
out["centred_62_window"] = [(n-62)//2, (n-62)//2+62]
(R/"results"/"surface_depth.json").write_text(json.dumps(out, indent=2))
print(f"layers {n}, geometric centre {(n-1)/2}")
for k, p in out["peaks"].items():
    print(f"  {k:6s} argmax z={p['argmax']:3d}  centre_of_mass={p['centre_of_mass']:6.2f}")
print("centred 62 window:", out["centred_62_window"])
print("  z  mean   std    grad   nonzero")
for r in rows[::6]:
    print(f"  {r['z']:3d} {r['mean']:6.2f} {r['std']:6.2f} {r['grad']:6.3f} {r['nonzero_frac']:.3f}")
