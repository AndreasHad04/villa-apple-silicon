"""Does the recovered START_LAYER generalise to a DIFFERENT segment?

The window was recovered on PHerc1667/20240304141531. If it is a property of
the model and the volume convention it must hold elsewhere. If it is a
property of that one crop, it will not. Run the same recovery on another
segment with its own published canonical prediction.
"""
import argparse, json, pathlib, sys, time
import numpy as np, tifffile, zarr, s3fs, torch
R = pathlib.Path(__file__).resolve().parent.parent
D = R/"villa"/"ink-detection"/"optimized_inference"
sys.path.insert(0, str(D)); sys.path.insert(0, str(R/"ops"))
from inference import run_inference, CFG
from model_resnet3d_3d_decoder import load_model
from device_utils import select_device
B = "vesuvius-challenge-open-data"

ap = argparse.ArgumentParser()
ap.add_argument("--scroll", required=True)
ap.add_argument("--seg", required=True)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--offsets", default="0,9,18,22,24,26,30,39")
ap.add_argument("--tag", required=True)
a = ap.parse_args()
offs = [int(v) for v in a.offsets.split(",")]

fs = s3fs.S3FileSystem(anon=True)
segp = f"{B}/{a.scroll}/segments/{a.seg}"
cands = [p for p in fs.ls(f"{segp}/surface-volumes")
         if p.endswith(".zarr") and ("2.399um" in p or "2.4um" in p)]
assert cands, f"no 2.4um-class volume under {segp}/surface-volumes"
vol = cands[0]
pred_tif = [p for p in fs.ls(f"{segp}/ink-detection")
            if p.endswith(".tif") and "new_canon_autoresearch_recipe" in p][0]
print("vol ", vol.split("/")[-1], flush=True)
print("pred", pred_tif.split("/")[-1], flush=True)

st = zarr.storage.FsspecStore.from_url(f"s3://{vol}", storage_options={"anon": True})
arr = zarr.open_group(store=st, mode="r")["0"]
nlay, H, W = arr.shape
local = R/"reference"/f"ref_{a.tag}.tif"
if not local.exists():
    t0=time.time(); fs.get(pred_tif, str(local)); print(f"ref downloaded {time.time()-t0:.0f}s", flush=True)
full = zarr.open(tifffile.imread(local, aszarr=True), mode="r")
assert full.shape == (H, W), f"reference {full.shape} != volume {(H,W)}"

# pick an ink-rich window from THEIR map, same rule as before
best=None
for y in range(0, H-a.size, max(a.size, (H-a.size)//8 or 1)):
    for x in range(0, W-a.size, max(a.size, (W-a.size)//8 or 1)):
        s=np.asarray(full[y:y+a.size:8, x:x+a.size:8]).astype(np.float32)/255.0
        if best is None or s.std()>best[0]: best=(float(s.std()), y, x)
_, Y0, X0 = best
print(f"ink-rich window y={Y0} x={X0} ref_std={best[0]:.4f}  (volume {nlay} layers)", flush=True)

zmax = 1+max(offs)+62
stack = np.asarray(arr[0:zmax, Y0:Y0+a.size, X0:X0+a.size])
ref = np.asarray(full[Y0:Y0+a.size, X0:X0+a.size]).astype(np.float64)/255.0
rng=np.random.default_rng(0); rs=ref.ravel().copy(); rng.shuffle(rs); rs=rs.reshape(ref.shape)

CFG.model_type="resnet3d-152-3d-decoder"; CFG.in_chans=62
CFG.tile_size=CFG.size=256; CFG.stride=128; CFG.batch_size=4; CFG.workers=0
CFG.zarr_output_dir=str(R/"results"/f"part_{a.tag}"); pathlib.Path(CFG.zarr_output_dir).mkdir(parents=True,exist_ok=True)
dev=select_device(); model=load_model(str(R/"models"/"r152_3ddec_v2_l5_epoch13.ckpt"),dev,num_frames=62)
rows=[]
for d in offs:
    z0=1+d
    if z0+62>stack.shape[0]: print(f"  offset {d}: out of range"); continue
    layers=np.ascontiguousarray(np.transpose(stack[z0:z0+62],(1,2,0)))
    res=run_inference(layers,model,dev)
    p=np.asarray(zarr.open(res["mask_pred"],mode="r")); c=np.asarray(zarr.open(res["mask_count"],mode="r"))
    bl=np.divide(p,c,out=np.zeros_like(p),where=c>0).astype(np.float64)
    r=float(np.corrcoef(bl.ravel(),ref.ravel())[0,1]); rsh=float(np.corrcoef(bl.ravel(),rs.ravel())[0,1])
    rows.append({"offset":d,"start_layer":z0,"pearson_vs_reference":round(r,4),
                 "pearson_vs_shuffled":round(rsh,4)})
    print(f"  offset {d:+3d} START_LAYER {z0:3d}  r_ref {r:+.4f}  r_shuf {rsh:+.4f}", flush=True)
vals=[x["pearson_vs_reference"] for x in rows]
best_row=max(rows,key=lambda x:x["pearson_vs_reference"])
out={"scroll":a.scroll,"seg":a.seg,"layers":nlay,"crop":[Y0,X0,a.size],"rows":rows,
     "verdict":{"best_start_layer":best_row["start_layer"],"best_pearson":best_row["pearson_vs_reference"],
                "readme_start_layer_1_pearson":next((x["pearson_vs_reference"] for x in rows if x["start_layer"]==1),None),
                "spread":round(max(vals)-min(vals),4),
                "max_abs_shuffle_floor":round(max(abs(x["pearson_vs_shuffled"]) for x in rows),4)}}
(R/"results"/f"window_generalise_{a.tag}.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out["verdict"],indent=2), flush=True)
