"""ARM FLINJ phase 1 smoke run: ONE host sub-crop of at most 768 x 768 px, primary checkpoint, forward.

At the registered geometry (cap 2.5 mm = 267 px on ag174) a row of 7 letters is about 2.2k px long and the far
background starts 534 px from it, so neither fits a 768 px sub-crop. The smoke row is therefore drawn at cap
height 72 px with the registered stroke/cap ratio (0.35 / 2.5). Its C and AUC test the plumbing only and are
NOT estimates of the registered endpoint. What it does measure for real: C0 determinism, seconds per
inference, injection and zarr-write cost, and peak memory.

Gate: armguard check --need 3, polled every 60 s for up to 30 minutes; refuses rather than runs.

    /usr/bin/time -l -o /Users/andreashad04/money/vesuv/results/flinj/smoke/time_l.txt \
        /Users/andreashad04/money/vesuv/env/bin/python3 /Users/andreashad04/money/vesuv/ops/flinj/smoke.py
"""
raise SystemExit(f"{__file__}: superseded by amendments 2 and 3; kept as the record of what it measured")
import resource, sys, time
import numpy as np, tifffile
from common import DATA, PRIMARY_CKPT, RES, REG, armguard, atomic_json, banner, host_seed, infer, load_host, sha_array, write_zarr
import figs, glyphs as GL, metric as MT, residuals as RS
from inject import inject

HOST, SUB, CAP = "ag174", 768, 72.0
STROKE = CAP * REG["stroke_mm"] / REG["cap_mm"]
OUT = RES / "smoke"


def main():
    banner(__file__); OUT.mkdir(parents=True, exist_ok=True)
    ok, msg = armguard(3, poll_s=60, max_wait_s=1800)
    if not ok:
        atomic_json(OUT / "smoke.json", dict(status="BLOCKED by armguard --need 3 after 30 minutes", armguard=msg)); return 3
    I, _, _, valid = load_host(HOST); _, H, W = I.shape
    y0, x0 = (H - SUB) // 2, (W - SUB) // 2
    Is, vs = np.ascontiguousarray(I[:, y0:y0 + SUB, x0:x0 + SUB]), valid[y0:y0 + SUB, x0:x0 + SUB]
    lay = GL.plan_layout(vs, CAP, STROKE, host_seed(HOST), border=REG["border_px"])
    assert lay["axis"] is not None, "smoke row does not fit"
    g = GL.build(vs, CAP, STROKE, lay, sigma=REG["soft_sigma_px"]); geom = {k: g[k] for k in ("strokes", "counters", "far")}
    libs = RS.host_libraries(HOST)
    rec = dict(host=HOST, subcrop=[y0, y0 + SUB, x0, x0 + SUB], cap_px=CAP, stroke_px=STROKE, layout=dict(lay, glyphs=[list(x) for x in lay["glyphs"]]),
               n=dict(strokes=int(geom["strokes"].sum()), counters=int(geom["counters"].sum()), far=int(geom["far"].sum())),
               library_patches=int(len(libs["all"])), ckpt=str(PRIMARY_CKPT.relative_to(PRIMARY_CKPT.parents[3])), armguard=msg, inputs={}, runs={})
    inputs = {"none": Is}
    for a in (1.0, 16.0):
        t = time.time(); v, box = inject(Is, g["G"], a, "real", libs, host_seed(HOST)); rec["inputs"][f"a{a:g}"] = dict(inject_s=time.time() - t, box=box)
        inputs[f"a{a:g}"] = v
    for k, v in inputs.items():
        t = time.time(); write_zarr(DATA / "smoke" / HOST / k / "ct.zarr", v)
        rec["inputs"].setdefault(k, {}).update(zarr_write_s=time.time() - t, sha256=sha_array(v), changed_voxels=int((v != Is).sum()))
    order = [("none_r1", "none"), ("none_r2", "none"), ("a1", "a1"), ("a16", "a16")]
    P = {}
    for name, k in order:
        out = OUT / "pred" / f"{name}.tif"
        secs = infer(DATA / "smoke" / HOST / k / "ct.zarr", PRIMARY_CKPT, out)
        P[name] = tifffile.imread(out); rec["runs"][name] = dict(secs=secs, input=k)
        print(f"DONE smoke {name} {secs:.2f}s", flush=True)
    secs = [rec["runs"][n]["secs"] for n, _ in order]
    rec["seconds_per_inference"] = dict(min=min(secs), median=float(np.median(secs)), max=max(secs), n=len(secs), mpx=SUB * SUB / 1e6,
                                        s_per_mpx_median=float(np.median(secs)) / (SUB * SUB / 1e6))
    rec["C0"] = dict(max_abs_D=int(np.abs(P["none_r1"].astype(np.int32) - P["none_r2"]).max()), pass_=bool((P["none_r1"] == P["none_r2"]).all()))
    for a in ("a1", "a16"):
        rec[a] = MT.score(P[a], P["none_r1"], geom)
        figs.panels(OUT / f"smoke_{a}.png", g["binary"], P["none_r1"], P[a], f"FLINJ SMOKE {HOST} sub-crop, cap {CAP:g} px (NOT registered size), a = {a[1:]}")
    rec["peak_rss_bytes_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    rec["status"] = "done"
    atomic_json(OUT / "smoke.json", rec)
    print("C0", rec["C0"], "| a1 C %.4f AUC_P %.4f | a16 C %.4f AUC_P %.4f | secs %s | rss %.2f GB" % (
        rec["a1"]["C"], rec["a1"]["auc_P"], rec["a16"]["C"], rec["a16"]["auc_P"], [round(s, 2) for s in secs], rec["peak_rss_bytes_self"] / 1e9), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
