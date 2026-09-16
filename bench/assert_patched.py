"""Refuse to measure against an UNPATCHED villa tree.

Switching the villa checkout to another branch silently reverts inference.py
to the no-op autocast, which would produce measurements that look patched and
are not. This asserts the tree under test is the patched one.
"""
import pathlib, sys
R = pathlib.Path(__file__).resolve().parent.parent
f = R/"villa"/"ink-detection"/"optimized_inference"/"inference.py"
src = f.read_text()
bad = 'amp_device = "cuda" if device.type == "cuda" else "cpu"'
good = "amp_device = amp_device_type(device)"
ok = (good in src) and (bad not in src)
print(("OK: villa tree is PATCHED" if ok else
       "REFUSING: villa tree is UNPATCHED, measurements would be wrong"))
print(f"  {f}")
print(f"  patched expression present: {good in src}")
print(f"  original expression present: {bad in src}")
sys.exit(0 if ok else 1)
