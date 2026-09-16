"""Run villa's ink detection on an Apple Silicon Mac, locally, with NO AWS
credentials and NO container.

villa's entrypoint.py assumes /workspace and a credentialed boto3 client, so
an ordinary laptop user cannot run it at all. This reads a crop straight from
the public open-data bucket anonymously and hands it to villa's OWN
run_inference, unmodified.
"""
import argparse, json, sys, time, pathlib
import numpy as np, torch

R = pathlib.Path(__file__).resolve().parent.parent
D = R / "villa" / "ink-detection" / "optimized_inference"
sys.path.insert(0, str(D)); sys.path.insert(0, str(R / "ops"))

import inference as vinf
from inference import run_inference, CFG
from device_utils import select_device
from model_resnet3d_3d_decoder import load_model

BUCKET = "vesuvius-challenge-open-data"

ap = argparse.ArgumentParser()
ap.add_argument("--zarr", default=("PHerc1667/segments/20240304141531-w013_20240304141531_flatboi"
                                   "/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr"))
ap.add_argument("--level", default="0")
ap.add_argument("--y0", type=int, default=20000)
ap.add_argument("--x0", type=int, default=40000)
ap.add_argument("--size", type=int, default=2048)
ap.add_argument("--start-layer", type=int, default=1)
ap.add_argument("--end-layer", type=int, default=63)
ap.add_argument("--tile", type=int, default=256)
ap.add_argument("--stride", type=int, default=128)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--device", default=None)
ap.add_argument("--tag", default="crop")
a = ap.parse_args()

dev = select_device(a.device)
print(f"device: {dev}")

import zarr
t0 = time.time()
store = zarr.storage.FsspecStore.from_url(f"s3://{BUCKET}/{a.zarr}", storage_options={"anon": True})
g = zarr.open_group(store=store, mode="r")
arr = g[a.level]
y1, x1 = a.y0 + a.size, a.x0 + a.size
assert y1 <= arr.shape[1] and x1 <= arr.shape[2], f"crop outside {arr.shape}"
vol = np.asarray(arr[a.start_layer:a.end_layer, a.y0:y1, a.x0:x1])   # (C,H,W)
read_s = time.time() - t0
layers = np.ascontiguousarray(np.transpose(vol, (1, 2, 0)))           # (H,W,C)
print(f"read {layers.shape} {layers.dtype} in {read_s:.1f}s, "
      f"{layers.nbytes/1e9:.2f} GB, nonzero {float((layers!=0).mean()):.3f}")

CFG.model_type = "resnet3d-152-3d-decoder"
CFG.in_chans = a.end_layer - a.start_layer
CFG.tile_size = a.tile
CFG.size = a.tile
CFG.stride = a.stride
CFG.batch_size = a.batch
CFG.workers = 0          # spawn workers re-import torch; pointless for a crop
CFG.zarr_output_dir = str(R / "results" / "partitions")
pathlib.Path(CFG.zarr_output_dir).mkdir(parents=True, exist_ok=True)

model = load_model(str(R / "models" / "r152_3ddec_v2_l5_epoch13.ckpt"), dev,
                   num_frames=CFG.in_chans)
t0 = time.time()
out = run_inference(layers, model, dev)
infer_s = time.time() - t0

pred = np.asarray(zarr.open(out["mask_pred"], mode="r"))
cnt = np.asarray(zarr.open(out["mask_count"], mode="r"))
blended = np.divide(pred, cnt, out=np.zeros_like(pred), where=cnt > 0)

ntiles = ((a.size - a.tile) // a.stride + 1) ** 2
rec = {"device": str(dev), "crop": [a.y0, a.x0, a.size], "level": a.level,
       "layers": [a.start_layer, a.end_layer], "tile": a.tile, "stride": a.stride,
       "batch": a.batch, "read_seconds": round(read_s, 2),
       "inference_seconds": round(infer_s, 2), "tiles_nominal": ntiles,
       "tiles_per_second": round(ntiles / infer_s, 3),
       "pred_shape": list(blended.shape),
       "pred_min": float(blended.min()), "pred_max": float(blended.max()),
       "pred_mean": float(blended.mean()),
       "frac_above_0.5": float((blended > 0.5).mean())}
name = f"{a.tag}_{dev.type}_{a.y0}_{a.x0}_{a.size}"
np.save(R / "results" / f"{name}.npy", blended.astype(np.float32))
(R / "results" / f"{name}.json").write_text(json.dumps(rec, indent=2))
print(json.dumps(rec, indent=2))
