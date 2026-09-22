"""ARM X, stage 2: villa's own flat ink inference on every prepared segment.

The model, normalisation, tiling, blending and uint8 encoding are villa's
`vesuvius.ink_detection.inference.infer.main`, called in-process. The ONLY change
is the device: the non-CUDA branch of `prepare_model_for_inference` returns MPS,
the same change as nerln's villa PR #1865. Autocast stays off because infer.py
guards it on CUDA, so every number here is fp32.

Resumable per unit: one TIFF per (segment, checkpoint, direction), written to a
.partial path and renamed, so a killed run never leaves a TIFF that looks done.

    env/bin/python3 ops/xs_infer.py [segment ...] [--ckpts all|final|primary]
"""
import json, os, pathlib, sys, time
import numpy as np, tifffile, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "villa" / "vesuvius" / "src"))
from vesuvius.ink_detection.inference import infer as INF  # noqa: E402
from vesuvius.ink_detection.inference import inference_runtime as RT  # noqa: E402


def prepare_mps(model, *, gpu_ids, compile_model, compile_mode):
    if gpu_ids:
        return RT.prepare_model_for_inference(model, gpu_ids=gpu_ids, compile_model=compile_model, compile_mode=compile_mode)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model, _ = RT.maybe_compile_model(model, enabled=False, mode=compile_mode)
    return model, device


INF.prepare_model_for_inference = prepare_mps
PRIMARY = ROOT / "models" / "ink_9um" / "hybrid_3d2d-seed42" / "step-075000.pth"


def checkpoints(which):
    allc = sorted((ROOT / "models" / "ink_9um").glob("hybrid_3d2d-seed*/step-*.pth"))
    if which == "primary": return [PRIMARY]
    if which == "final": return [c for c in allc if c.stem == "step-075000"]
    return [PRIMARY] + [c for c in allc if c != PRIMARY]  # primary first, so it lands earliest


def valid_tiff(p, shape):
    try:
        return p.exists() and tuple(tifffile.TiffFile(p).pages[0].shape) == tuple(shape)
    except Exception:
        return False


def main(argv):
    which = "all"
    if "--ckpts" in argv:
        i = argv.index("--ckpts"); which = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    segs = argv or sorted(p.name for p in (ROOT / "data" / "xs").iterdir() if (p / "meta.json").exists())
    log = ROOT / "results" / "xs" / "infer_runs.jsonl"
    villa_sha = os.popen(f"git -C {ROOT / 'villa'} rev-parse HEAD").read().strip()
    for seg in segs:
        sd = ROOT / "data" / "xs" / seg; meta = json.load(open(sd / "meta.json"))
        depth, H, W = meta["ct_shape"]
        # 32 pooled planes, labelled plane 16: [6, 27) puts it at index 10 of 21 (pre-registered)
        window = ["--layer-start", "6", "--layer-end", "27"] if depth == 32 else []
        for ck in checkpoints(which):
            for d in ("forward", "reverse"):
                out = ROOT / "results" / "xs" / "pred" / seg / f"{ck.parent.name}_{ck.stem}_{d}.tif"
                if valid_tiff(out, (H, W)):
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(f"{out.stem}.partial{os.getpid()}.tif")  # per process: a side job and the chain may overlap
                if tmp.exists(): tmp.unlink()
                if valid_tiff(out, (H, W)): continue  # another process finished it meanwhile
                argv_i = [str(sd / "ct.zarr"), str(ck), str(tmp), *window, "--direction", d,
                          "--no-compile", "--num-workers", "0", "--batch-size", "32"]
                t = time.time(); INF.main(argv_i); dt = time.time() - t
                assert valid_tiff(tmp, (H, W)), f"no valid output at {tmp}"
                os.replace(tmp, out)
                rec = dict(seg=seg, ckpt=f"{ck.parent.name}/{ck.name}", direction=d, secs=round(dt, 1), shape=[H, W],
                           window=window, device="mps" if torch.backends.mps.is_available() else "cpu", villa=villa_sha,
                           torch=torch.__version__, when=time.strftime("%Y-%m-%dT%H:%M:%S"))
                with open(log, "a") as f: f.write(json.dumps(rec) + "\n")
                print(f"DONE {seg} {rec['ckpt']} {d} {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
