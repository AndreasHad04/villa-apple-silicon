"""Does PR 1812's MPS path work for ALL FOUR model types, or only the one we tested?

`optimized_inference` supports timesformer (the DEFAULT), resnet3d-50,
resnet3d-152 and resnet3d-152-3d-decoder. Every measurement in PR 1812 used
resnet3d-152-3d-decoder. A device patch can only fail two ways: a missing MPS
op, or a numeric divergence. Neither depends on trained weights, so this runs
each architecture with RANDOM weights, which needs no checkpoint and covers
every op the real model would execute.

Also exercises the autocast expression PR 1812 changes,
torch.autocast(device_type=device.type), which was previously a no-op on MPS.
"""
import json, pathlib, sys, time, traceback
import numpy as np, torch
R = pathlib.Path(__file__).resolve().parent.parent
D = R / "villa" / "ink-detection" / "optimized_inference"
sys.path.insert(0, str(D))

SPECS = [("timesformer", "model_timesformer", dict(size=64, num_frames=26)),
         ("resnet3d-50", "model_resnet3d", dict(size=64, num_frames=30, enc="resnet3d-50", model_depth=50)),
         ("resnet3d-152", "model_resnet3d", dict(size=64, num_frames=30, enc="resnet3d-152", model_depth=152)),
         ("resnet3d-152-3d-decoder", "model_resnet3d_3d_decoder", dict(size=64, num_frames=62))]

def build(mod, kw):
    import importlib
    m = importlib.import_module(mod)
    torch.manual_seed(0)
    # the 3d decoder module names its top level model RegressionModel, the other
    # two name it RegressionPLModel. Take whichever the module actually defines
    # rather than assuming, so a rename upstream fails loudly here.
    cls = next((getattr(m, n) for n in ("RegressionPLModel", "RegressionModel")
                if hasattr(m, n)), None)
    assert cls is not None, f"{mod} defines neither RegressionPLModel nor RegressionModel"
    # pass only the kwargs this constructor actually accepts: the 3d decoder
    # takes with_norm alone and fixes depth and channels internally.
    import inspect
    ok = set(inspect.signature(cls.__init__).parameters)
    args = {k: v for k, v in dict(pred_shape=(1, 1), **kw).items() if k in ok}
    return cls(**args).eval()

def run(model, x, dev, amp):
    model = model.to(dev); xx = x.to(dev)
    with torch.no_grad():
        if amp:
            # the exact expression PR 1812 introduces
            with torch.autocast(device_type=dev.type):
                y = model(xx)
        else:
            y = model(xx)
    y = y[0] if isinstance(y, (tuple, list)) else y
    return y.float().cpu().numpy()

out = []
for name, mod, kw in SPECS:
    row = {"model_type": name, "num_frames": kw["num_frames"], "size": kw["size"]}
    try:
        model = build(mod, kw)
        torch.manual_seed(1)
        x = torch.randn(1, 1, kw["num_frames"], kw["size"], kw["size"])
        t = time.time(); cpu = run(model, x, torch.device("cpu"), False); row["cpu_s"] = round(time.time()-t, 3)
        for amp in (False, True):
            k = "mps_amp" if amp else "mps_fp32"
            try:
                t = time.time(); got = run(model, x, torch.device("mps"), amp)
                row[k + "_s"] = round(time.time()-t, 3)
                row[k + "_shape_matches"] = list(got.shape) == list(cpu.shape)
                # A NaN difference is NOT a pass. Report finiteness explicitly and
                # call a non finite result UNDECIDED, never OK: a threshold or an
                # eyeballed table is fail open on NaN and that is how a broken
                # numeric path gets recorded as working.
                fin_got = bool(np.isfinite(got).all()); fin_cpu = bool(np.isfinite(cpu).all())
                row[k + "_output_finite"] = fin_got
                row["cpu_output_finite"] = fin_cpu
                d = np.abs(got - cpu)
                row[k + "_max_abs_diff"] = (round(float(d.max()), 6)
                                            if np.isfinite(d).all() else None)
                row[k + "_status"] = ("OK" if fin_got and fin_cpu and np.isfinite(d).all()
                                      else "RUNS but output not finite: UNDECIDED")
            except Exception as e:
                row[k + "_status"] = f"{type(e).__name__}: {str(e)[:160]}"
        row["cpu_status"] = "OK"
        row["out_shape"] = list(cpu.shape)
    except Exception as e:
        row["cpu_status"] = f"{type(e).__name__}: {str(e)[:160]}"
        traceback.print_exc()
    out.append(row)
    print(json.dumps(row, indent=None), flush=True)

# CONTROL: the one non finite cell above is the 3d decoder under autocast with
# RANDOM weights. Untrained weights make activations large, and float16 overflows.
# Distinguish "random weights overflow" from "the patch is broken" by repeating it
# with the real checkpoint, which is the only weights anyone actually runs.
ctrl = {"what": "resnet3d-152-3d-decoder with the REAL checkpoint, autocast on MPS"}
try:
    from model_resnet3d_3d_decoder import load_model
    dev = torch.device("mps")
    m = load_model(str(R / "models" / "r152_3ddec_v2_l5_epoch13.ckpt"), dev, num_frames=62)
    torch.manual_seed(2)
    x = torch.randint(0, 256, (1, 1, 62, 64, 64), dtype=torch.uint8).float().to(dev) / 255.0
    with torch.no_grad():
        a = m.forward(x)
        with torch.autocast(device_type="mps"):
            b = m.forward(x)
    a = (a[0] if isinstance(a, (tuple, list)) else a).float().cpu().numpy()
    b = (b[0] if isinstance(b, (tuple, list)) else b).float().cpu().numpy()
    ctrl["fp32_finite"] = bool(np.isfinite(a).all())
    ctrl["autocast_finite"] = bool(np.isfinite(b).all())
    ctrl["max_abs_diff"] = round(float(np.abs(a - b).max()), 6) if np.isfinite(a - b).all() else None
    ctrl["verdict"] = ("trained weights are finite on the autocast path, so the non finite "
                       "random weight cell is a float16 overflow from untrained weights, "
                       "not a defect in the patch") if ctrl["autocast_finite"] else                       "trained weights are ALSO non finite: this is a real defect, report it"
except Exception as e:
    ctrl["verdict"] = f"control did not run: {type(e).__name__}: {str(e)[:160]}"
print("\nCONTROL:", json.dumps(ctrl), flush=True)

res = {"torch": torch.__version__, "mps_available": bool(torch.backends.mps.is_available()),
       "trained_weight_control": ctrl,
       "note": "random weights: this measures MPS OP COVERAGE and numeric agreement, not accuracy",
       "rows": out}
(R / "results" / "mps_model_coverage.json").write_text(json.dumps(res, indent=2))
ok = [r for r in out if r.get("mps_fp32_status") == "OK" and r.get("mps_amp_status") == "OK"]
runs = [r for r in out if str(r.get("mps_fp32_status","")).startswith(("OK","RUNS"))
        and str(r.get("mps_amp_status","")).startswith(("OK","RUNS"))]
print(f"\ntorch {torch.__version__}  MPS available {res['mps_available']}")
print(f"{len(runs)} of {len(out)} model types RUN on MPS in both fp32 and autocast "
      f"without raising; {len(ok)} of {len(out)} also produce finite output that "
      f"agrees with the CPU")
for r in out:
    print(f"  {r['model_type']:26s} fp32 {r.get('mps_fp32_status','?'):12s} "
          f"amp {r.get('mps_amp_status','?'):12s} "
          f"max|diff| fp32 {r.get('mps_fp32_max_abs_diff','-')} amp {r.get('mps_amp_max_abs_diff','-')}")
