"""ARM W: can the correct depth window be found WITHOUT a reference map?

Pre-registered in PREREGISTRATION.md, amendment 2026-09-22, before this ran.

For each crop: sweep the fixed grid, and at every window record the ground
truth (Pearson against villa's published prediction) plus five candidate
selectors that never see the reference. The extra cost is one inference per
window on the depth shuffled stack, which is what S4 and S5 need.

Resumable: one JSON per crop, finished crops are skipped.
"""
import json, pathlib, sys, time
import numpy as np, tifffile, zarr, s3fs, torch
R = pathlib.Path(__file__).resolve().parent.parent
D = R / "villa" / "ink-detection" / "optimized_inference"
sys.path.insert(0, str(D)); sys.path.insert(0, str(R / "ops"))
from inference import run_inference, CFG
from model_resnet3d_3d_decoder import load_model
from device_utils import select_device
from inkdiag import nulls, stats, blend          # production functions, not copies

B = "vesuvius-challenge-open-data"
GRID = [1, 13, 19, 23, 24, 28, 33, 40]
SIZE = 1024

def selectors(p):
    """Every statistic here is computed from the prediction ALONE.

    C4 of the pre-registration: this function takes no reference argument, so
    a selector structurally cannot see the answer it is supposed to find.
    """
    f = np.asarray(p, dtype=np.float64).ravel()
    s = stats(np.asarray(p))                     # production stats()
    return {"separation": s["separation"],
            "frac_above_0.5": s["frac_above_0.5"],
            "frac_mushy": float(((f > 0.2) & (f < 0.8)).mean()),
            "pred_std": float(f.std())}

# C4, structural: one argument, the prediction. A selector that can be handed
# the reference is not a selector, and the assert is cheaper than the promise.
assert selectors.__code__.co_argcount == 1, "C4: selector takes more than the prediction"
assert "ref" not in selectors.__code__.co_varnames, "C4: selector sees the reference"

def main():
    fs = s3fs.S3FileSystem(anon=True)
    tal = json.load(open(R / "results" / "window_tally.json"))["rows"]
    CFG.model_type = "resnet3d-152-3d-decoder"; CFG.in_chans = 62
    CFG.tile_size = CFG.size = 256; CFG.stride = 128; CFG.batch_size = 4; CFG.workers = 0
    dev = select_device()
    model = load_model(str(R / "models" / "r152_3ddec_v2_l5_epoch13.ckpt"), dev, num_frames=62)
    t0 = time.time(); ndone = 0
    for i, row in enumerate(tal, 1):
        tag = f"sel_{row['scroll']}_{row['crop'][0]}_{row['crop'][1]}"
        out = R / "results" / f"window_selector_{tag}.json"
        if out.exists():
            print(f"[{i}/{len(tal)}] {tag}: done, skipping", flush=True); ndone += 1; continue
        segp = f"{B}/{row['scroll']}/segments/{row['seg']}"
        vol = [p for p in fs.ls(f"{segp}/surface-volumes")
               if p.endswith(".zarr") and ("2.399um" in p or "2.4um" in p)][0]
        pred = [p for p in fs.ls(f"{segp}/ink-detection")
                if p.endswith(".tif") and "new_canon_autoresearch_recipe" in p][0]
        st = zarr.storage.FsspecStore.from_url(f"s3://{vol}", storage_options={"anon": True})
        arr = zarr.open_group(store=st, mode="r")["0"]
        nlay, H, W = arr.shape
        local = R / "reference" / f"ref_sel_{row['scroll']}_{row['crop'][0]}.tif"
        if not local.exists():
            tmp = local.with_suffix(".partial"); fs.get(pred, str(tmp))
            tifffile.TiffFile(tmp).close(); tmp.replace(local)
        full = zarr.open(tifffile.imread(local, aszarr=True), mode="r")
        Y0, X0, sz = row["crop"]
        ref = np.asarray(full[Y0:Y0+sz, X0:X0+sz]).astype(np.float64) / 255.0
        rng = np.random.default_rng(0); rs = ref.ravel().copy(); rng.shuffle(rs); rs = rs.reshape(ref.shape)
        zmax = max(GRID) + 62
        assert zmax <= nlay, f"{tag}: grid needs {zmax} layers, volume has {nlay}"
        stack = np.asarray(arr[0:zmax, Y0:Y0+sz, X0:X0+sz])
        CFG.zarr_output_dir = str(R / "results" / f"part_{tag}")
        pathlib.Path(CFG.zarr_output_dir).mkdir(parents=True, exist_ok=True)
        rows = []
        for z0 in GRID:
            win = stack[z0:z0+62]
            p = blend(run_inference(np.ascontiguousarray(np.transpose(win, (1, 2, 0))), model, dev))
            nl = nulls(win, seed=0)["depth_shuffle"]
            pn = blend(run_inference(np.ascontiguousarray(np.transpose(nl, (1, 2, 0))), model, dev))
            sr, sn = selectors(p), selectors(pn)
            r = float(np.corrcoef(np.asarray(p).ravel(), ref.ravel())[0, 1])
            rsh = float(np.corrcoef(np.asarray(p).ravel(), rs.ravel())[0, 1])
            rows.append({"start_layer": z0, "pearson_vs_reference": round(r, 4),
                         "pearson_vs_shuffled": round(rsh, 4),
                         "S1_separation": round(sr["separation"], 6),
                         "S2_frac_mushy": round(sr["frac_mushy"], 6),
                         "S3_pred_std": round(sr["pred_std"], 6),
                         "S4_null_collapse": round(sn["frac_above_0.5"], 6),
                         "S5_null_margin": round(sr["frac_above_0.5"] - sn["frac_above_0.5"], 6),
                         "real_frac_above_0.5": round(sr["frac_above_0.5"], 6)})
            print(f"  {tag} sl {z0:3d}  r {r:+.4f}  S1 {sr['separation']:.4f} "
                  f"S2 {sr['frac_mushy']:.4f} S3 {sr['pred_std']:.4f} "
                  f"S4 {sn['frac_above_0.5']:.4f}", flush=True)
        out.write_text(json.dumps({"scroll": row["scroll"], "seg": row["seg"],
                                   "crop": row["crop"], "layers": nlay,
                                   "grid": GRID, "rows": rows}, indent=2))
        ndone += 1
        per = (time.time() - t0) / ndone
        print(f"[{i}/{len(tal)}] {tag} DONE  ETA {(len(tal)-i)*per/60:.0f} min", flush=True)
    print(f"ALL DONE {ndone}/{len(tal)} in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
