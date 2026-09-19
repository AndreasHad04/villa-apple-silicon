"""inkdiag: which failure mode am I fighting?

The organisers state the open problem directly: when a model shows no ink, the
right conclusion is neither "the scan failed" nor "the model failed", and six
explanations remain open. This measures two things that discriminate between
them, on real data, with adversarial nulls.

D3, DEPTH-LOCALISATION PROFILE.
  The model reads a 62 layer window around the predicted surface. Slide that
  window through depth and record the prediction at each offset. A signal that
  really lives on the writing surface must be PEAKED in offset. A flat profile
  means the model is reading something depth independent, which is the
  "not exploiting the right features" failure rather than a scan failure.
  A peak at a NONZERO offset means the surface is mislocalised, by that many
  voxels, which is directly actionable.

D2, SUBSTRATE NULLS (the hallucination controls).
  Destroy ink structure while preserving stated statistics, and see what
  survives. Ordered from most to least conservative:
    depth_shuffle  permutes the 62 layers. Every voxel value and every
                   per-column histogram is preserved EXACTLY; only depth
                   ordering dies. If ink is depth localised morphology this
                   must destroy it. If the prediction survives, the model is
                   reading a depth independent summary.
    phase_scramble randomises the in plane FFT phase per layer, preserving the
                   amplitude spectrum, so texture power is preserved and
                   structure is not.
    voxel_shuffle  permutes all voxels in the crop. Preserves the global
                   histogram only. The weakest null and the easiest to beat.
"""
import argparse, json, os, sys, time, pathlib
import numpy as np, torch, zarr

def _find_root(start):
    """Walk up for the tree holding villa/ and models/, so this file works both
    at ops/ and at bench/ in the published repo. A fixed parent.parent resolved
    to the wrong tree the moment the file was copied. VESUV_ROOT overrides."""
    env = os.environ.get("VESUV_ROOT")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    for d in [start, *start.parents]:
        if (d / "villa").is_dir() and (d / "models").is_dir():
            return d
    return start.parent

R = _find_root(pathlib.Path(__file__).resolve().parent)
D = R/"villa"/"ink-detection"/"optimized_inference"
sys.path.insert(0, str(D)); sys.path.insert(0, str(R/"ops"))
import inference as vinf
from inference import run_inference, CFG
from model_resnet3d_3d_decoder import load_model
from device_utils import select_device

BUCKET = "vesuvius-challenge-open-data"


def read_stack(zpath, y0, x0, size, z0, z1, level="0"):
    store = zarr.storage.FsspecStore.from_url(f"s3://{BUCKET}/{zpath}",
                                              storage_options={"anon": True})
    arr = zarr.open_group(store=store, mode="r")[level]
    return np.asarray(arr[z0:z1, y0:y0+size, x0:x0+size]), arr.shape


def blend(res):
    p = np.asarray(zarr.open(res["mask_pred"], mode="r"))
    c = np.asarray(zarr.open(res["mask_count"], mode="r"))
    return np.divide(p, c, out=np.zeros_like(p), where=c > 0)


def stats(pred):
    """Summaries that need no ground truth."""
    f = pred.ravel()
    hi = float((f > 0.5).mean())
    # bimodality: real ink maps separate into ink and not-ink. A mushy
    # unimodal map is what an off-surface or hallucinating model gives.
    lo_m = float(f[f <= 0.5].mean()) if (f <= 0.5).any() else 0.0
    hi_m = float(f[f > 0.5].mean()) if (f > 0.5).any() else 0.0
    return {"mean": float(f.mean()), "std": float(f.std()), "max": float(f.max()),
            "frac_above_0.5": hi, "separation": round(hi_m - lo_m, 6)}


def nulls(vol, seed=0):
    """vol is (C,H,W) uint8. Each null states exactly what it preserves."""
    rng = np.random.default_rng(seed)
    out = {}
    o = vol.copy(); rng.shuffle(o, axis=0)              # per-(y,x) column permute
    out["depth_shuffle"] = o
    amp = np.abs(np.fft.rfft2(vol.astype(np.float32), axes=(1, 2)))
    ph = rng.uniform(-np.pi, np.pi, amp.shape)
    sc = np.fft.irfft2(amp*np.exp(1j*ph), s=vol.shape[1:], axes=(1, 2))
    sc = np.clip(sc, 0, 255)
    out["phase_scramble"] = sc.astype(np.uint8)
    fl = vol.ravel().copy(); rng.shuffle(fl)
    out["voxel_shuffle"] = fl.reshape(vol.shape)
    # depth_reverse: preserves EVERYTHING except direction. If ink morphology
    # is directional (ink on one face of the sheet) this must cost something.
    out["depth_reverse"] = vol[::-1].copy()
    # depth_roll: preserves adjacency almost everywhere, moves absolute depth.
    out["depth_roll_half"] = np.roll(vol, vol.shape[0]//2, axis=0)
    # per_column_shuffle: independent permutation per (y,x). Kills depth order
    # AND in-plane coherence, so it sits between depth_shuffle and voxel_shuffle.
    idx = np.argsort(rng.random(vol.shape), axis=0)
    out["per_column_shuffle"] = np.take_along_axis(vol, idx, axis=0)
    # single_layer_repeat: THE sharpest one. Every layer replaced by a copy of
    # the middle layer. In-plane texture is preserved perfectly and ALL depth
    # information is destroyed. Confident ink here means the model is reading
    # 2D texture, not 3D morphology.
    mid = vol[vol.shape[0]//2]
    out["single_layer_repeat"] = np.repeat(mid[None], vol.shape[0], axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default=("PHerc1667/segments/20240304141531-w013_20240304141531_flatboi"
                                       "/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr"))
    ap.add_argument("--y0", type=int, default=20480)
    ap.add_argument("--x0", type=int, default=16384)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--nominal-start", type=int, default=1)
    ap.add_argument("--chans", type=int, default=62)
    ap.add_argument("--offsets", default="-1,0,4,8,12,16,20,24,28,32,40,46")
    ap.add_argument("--pace", type=float, default=1.0, help="MPS duty cycle, 1.0 = flat out")
    ap.add_argument("--tag", default="diag")
    ap.add_argument("--skip-nulls", action="store_true")
    a = ap.parse_args()

    offs = [int(v) for v in a.offsets.split(",")]
    dev = select_device()
    zmax = a.nominal_start + max(offs) + a.chans
    print(f"device {dev}; reading z[0,{zmax}) for offsets {offs}", flush=True)
    t0 = time.time()
    stack, shp = read_stack(a.zarr, a.y0, a.x0, a.size, 0, zmax)
    print(f"read {stack.shape} in {time.time()-t0:.0f}s (volume {shp})", flush=True)

    CFG.model_type = "resnet3d-152-3d-decoder"; CFG.in_chans = a.chans
    CFG.tile_size = CFG.size = 256; CFG.stride = 128; CFG.batch_size = 4; CFG.workers = 0
    CFG.zarr_output_dir = str(R/"results"/f"part_{a.tag}")
    pathlib.Path(CFG.zarr_output_dir).mkdir(parents=True, exist_ok=True)
    model = load_model(str(R/"models"/"r152_3ddec_v2_l5_epoch13.ckpt"), dev, num_frames=a.chans)

    def run(vol3d):
        layers = np.ascontiguousarray(np.transpose(vol3d, (1, 2, 0)))
        t = time.time(); out = blend(run_inference(layers, model, dev)); dt = time.time()-t
        if a.pace < 1.0:
            time.sleep(dt*(1.0/a.pace - 1.0))
        return out, dt

    rec = {"crop": [a.y0, a.x0, a.size], "offsets": offs, "chans": a.chans,
           "nominal_start": a.nominal_start, "device": str(dev), "pace": a.pace,
           "zarr": a.zarr, "depth_profile": [], "nulls": []}

    base = None
    for d in offs:
        z0 = a.nominal_start + d
        if z0 < 0 or z0 + a.chans > stack.shape[0]:
            print(f"  offset {d}: OUT OF RANGE, skipped", flush=True); continue
        pred, dt = run(stack[z0:z0+a.chans])
        row = {"offset": d, "z_window": [z0, z0+a.chans], "seconds": round(dt, 1), **stats(pred)}
        if d == 0:
            base = pred
        if base is not None:
            row["pearson_vs_offset0"] = float(np.corrcoef(pred.ravel(), base.ravel())[0, 1])
        np.save(R/"results"/f"{a.tag}_off{d}.npy", pred.astype(np.float32))
        rec["depth_profile"].append(row)
        print(f"  offset {d:+3d} z{row['z_window']} mean {row['mean']:.4f} "
              f"sep {row['separation']:.4f} frac>0.5 {row['frac_above_0.5']:.4f} "
              f"r_vs_0 {row.get('pearson_vs_offset0', float('nan')):.4f}  {dt:.0f}s", flush=True)
        (R/"results"/f"inkdiag_{a.tag}.json").write_text(json.dumps(rec, indent=2))

    if not a.skip_nulls:
        real = stack[a.nominal_start:a.nominal_start+a.chans]
        print("nulls:", flush=True)
        for name, vol in nulls(real).items():
            pred, dt = run(vol)
            row = {"null": name, "seconds": round(dt, 1), **stats(pred)}
            if base is not None:
                row["pearson_vs_real"] = float(np.corrcoef(pred.ravel(), base.ravel())[0, 1])
            np.save(R/"results"/f"{a.tag}_null_{name}.npy", pred.astype(np.float32))
            rec["nulls"].append(row)
            print(f"  {name:15s} mean {row['mean']:.4f} sep {row['separation']:.4f} "
                  f"frac>0.5 {row['frac_above_0.5']:.4f} "
                  f"r_vs_real {row.get('pearson_vs_real', float('nan')):.4f}  {dt:.0f}s", flush=True)
            (R/"results"/f"inkdiag_{a.tag}.json").write_text(json.dumps(rec, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
