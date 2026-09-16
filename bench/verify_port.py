"""Controls for the Apple Silicon port. Each is watched FAILING on a mutant
before it is believed, per the project discipline.

The control that matters most is NOT "does MPS work". It is "is the CUDA path
unchanged", because that is the path every existing user is on and I have no
CUDA hardware to test it on.
"""
import json, sys, pathlib, importlib, types
import torch

R = pathlib.Path(__file__).resolve().parent.parent
D = R / "villa" / "ink-detection" / "optimized_inference"
sys.path.insert(0, str(D))
import device_utils as du

checks, failed = [], 0
def ck(name, got, want, note=""):
    global failed
    ok = got == want
    if not ok: failed += 1
    checks.append({"name": name, "ok": ok, "got": repr(got), "want": repr(want), "note": note})
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}: {got!r}" + (f"  ({note})" if note else ""))

print("REGRESSION: the CUDA path must be untouched")
ck("amp_device_type(cuda) is still cuda", du.amp_device_type(torch.device("cuda")), "cuda",
   "this is the line villa had; changing it would alter every existing GPU run")
ck("amp_device_type(cpu) is still cpu", du.amp_device_type(torch.device("cpu")), "cpu")
ck("an unknown backend falls back to cpu", du.amp_device_type(torch.device("meta")), "cpu",
   "fail CLOSED: never hand autocast a device_type it does not know")

# select_device must prefer cuda when cuda exists, on a machine that has none.
real = torch.cuda.is_available
try:
    torch.cuda.is_available = lambda: True
    ck("select_device prefers cuda when present", du.select_device().type, "cuda",
       "monkeypatched; no CUDA hardware here, so this is the only way to assert it")
finally:
    torch.cuda.is_available = real

print("\nNEW BEHAVIOUR: Apple Silicon")
ck("select_device picks mps on this machine", du.select_device().type,
   "mps" if torch.backends.mps.is_available() else "cpu")
ck("INK_DEVICE overrides", du.select_device(prefer="cpu").type, "cpu")
ck("amp_device_type(mps) is mps, not cpu", du.amp_device_type(torch.device("mps")), "mps",
   "the whole bug: villa passed 'cpu' here and autocast silently did nothing")

print("\nPOSITIVE CONTROL: the old line is watched FAILING")
def old_amp(device):                      # villa's original expression
    return "cuda" if device.type == "cuda" else "cpu"
ck("old code returns cpu for an mps device (the defect)", old_amp(torch.device("mps")), "cpu",
   "if this ever returns 'mps', the bug is gone and this control is vacuous")
ck("and the new code disagrees with it exactly there",
   du.amp_device_type(torch.device("mps")) != old_amp(torch.device("mps")), True)

print("\nMEASURED: autocast under the two device_types is not the same computation")
if torch.backends.mps.is_available():
    sys.path.insert(0, str(R / "ops"))
    from bench_device import load_model, tiles, CKPT, FRAMES
    m = load_model(str(CKPT), torch.device("mps"), num_frames=FRAMES)
    x = tiles(1, seed=0).to("mps")
    outs = {}
    for tag in ("cpu", "mps"):
        with torch.inference_mode():
            with torch.autocast(device_type=tag, enabled=True):
                outs[tag] = m.forward(x).float().cpu()
    with torch.inference_mode():
        fp32 = m.forward(x).float().cpu()
    d_cpu = float((outs["cpu"] - fp32).abs().max())
    d_mps = float((outs["mps"] - fp32).abs().max())
    ck("villa's device_type='cpu' is a NO-OP on mps (bit-identical to fp32)", d_cpu, 0.0,
       "so Apple Silicon users never got autocast at all")
    ck("device_type='mps' actually casts (differs from fp32)", d_mps > 1e-5, True,
       f"max|diff| = {d_mps:.2e}, fp16 magnitude")

print("\nIMPORTS: the patched modules still load")
for mod in ("device_utils",):
    try:
        importlib.reload(importlib.import_module(mod)); ck(f"import {mod}", True, True)
    except Exception as e:
        ck(f"import {mod}", f"{type(e).__name__}: {e}"[:120], True)

print(f"\n{len(checks)} checks, {failed} failures")
(R / "results" / "verify_port.json").write_text(
    json.dumps({"checks": checks, "failed": failed}, indent=2))
sys.exit(1 if failed else 0)
