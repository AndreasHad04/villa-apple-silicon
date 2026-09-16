"""AGENTS.md 1.3: how much does correcting amp_device change the REAL output?

Runs villa's full pipeline on the same real crop twice on MPS, once with the
current amp_device expression (a no-op) and once corrected, and diffs the
blended prediction. Nothing synthetic.
"""
import json, sys, time, pathlib
import numpy as np, torch, zarr
R = pathlib.Path(__file__).resolve().parent.parent
D = R/"villa"/"ink-detection"/"optimized_inference"
sys.path.insert(0, str(D)); sys.path.insert(0, str(R/"ops"))
import inference as vinf
from inference import run_inference, CFG
from model_resnet3d_3d_decoder import load_model
import device_utils

Y0, X0, SZ = 20480, 16384, 1024
BUCKET = "vesuvius-challenge-open-data"
ZP = ("PHerc1667/segments/20240304141531-w013_20240304141531_flatboi"
      "/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr")
store = zarr.storage.FsspecStore.from_url(f"s3://{BUCKET}/{ZP}", storage_options={"anon": True})
arr = zarr.open_group(store=store, mode="r")["0"]
layers = np.ascontiguousarray(np.transpose(np.asarray(arr[1:63, Y0:Y0+SZ, X0:X0+SZ]), (1, 2, 0)))
print("read", layers.shape, flush=True)

CFG.model_type = "resnet3d-152-3d-decoder"; CFG.in_chans = 62
CFG.tile_size = CFG.size = 256; CFG.stride = 128; CFG.batch_size = 4; CFG.workers = 0
CFG.zarr_output_dir = str(R/"results"/"partitions_amp"); pathlib.Path(CFG.zarr_output_dir).mkdir(parents=True, exist_ok=True)
dev = torch.device("mps")
model = load_model(str(R/"models"/"r152_3ddec_v2_l5_epoch13.ckpt"), dev, num_frames=62)

real = device_utils.amp_device_type
outs = {}
for tag, fn in (("villa_current_amp", lambda d: "cuda" if d.type == "cuda" else "cpu"),
                ("corrected_amp", real)):
    vinf.amp_device_type = fn
    t0 = time.time(); res = run_inference(layers, model, dev); dt = time.time()-t0
    pred = np.asarray(zarr.open(res["mask_pred"], mode="r"))
    cnt = np.asarray(zarr.open(res["mask_count"], mode="r"))
    outs[tag] = {"blend": np.divide(pred, cnt, out=np.zeros_like(pred), where=cnt > 0),
                 "seconds": round(dt, 2)}
    print(f"  {tag}: {dt:.1f}s", flush=True)
vinf.amp_device_type = real

a, b = outs["villa_current_amp"]["blend"], outs["corrected_amp"]["blend"]
d = np.abs(a-b)
rec = {"crop": [Y0, X0, SZ], "tiles": ((SZ-256)//128+1)**2,
       "villa_current_amp_seconds": outs["villa_current_amp"]["seconds"],
       "corrected_amp_seconds": outs["corrected_amp"]["seconds"],
       "speedup_end_to_end": round(outs["villa_current_amp"]["seconds"]/outs["corrected_amp"]["seconds"], 3),
       "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
       "p99_abs_diff": float(np.percentile(d, 99)),
       "pearson_between_the_two": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
       "frac_pixels_crossing_0.5_decision": float(((a > 0.5) != (b > 0.5)).mean())}
(R/"results"/"amp_endtoend.json").write_text(json.dumps(rec, indent=2))
print(json.dumps(rec, indent=2), flush=True)
