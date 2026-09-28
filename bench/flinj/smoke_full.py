"""ARM FLINJ amendment 1 smoke on a FULL-SIZE letter row: the whole ag174 crop, the A2 primary (smoothed) residual,
one amplitude, primary checkpoint, forward. Also C0 at full size (no-injection input run twice).

The amplitude is fixed by a model-free rule: the largest grid amplitude whose clipped fraction of stroke voxels is
within A5's 5% on ag174 (results/flinj/a5_precheck.json), i.e. the strongest injection A5 lets into a*.
Gate: armguard check --need 7, polled every 60 s for up to 30 minutes; refuses rather than runs.

    /usr/bin/time -l -o /Users/andreashad04/money/vesuv/results/flinj/smoke_full/time_l.txt \
        /Users/andreashad04/money/vesuv/env/bin/python3 /Users/andreashad04/money/vesuv/ops/flinj/smoke_full.py
"""
raise SystemExit(f"{__file__}: superseded by amendments 2 and 3; kept as the record of what it measured")
import json, resource, sys, time
import numpy as np, tifffile, zarr
from common import DATA, HOSTS, PRIMARY_CKPT, REG, RES, armguard, atomic_json, banner, cap_px, host_seed, infer, load_host, mm_px, sha_array, target_auc, write_zarr
import figs, glyphs as GL, metric as MT, residuals as RS
from inject import inject

HOST = "ag174"; OUT = RES / "smoke_full"


def main():
    banner(__file__); OUT.mkdir(parents=True, exist_ok=True)
    A = json.load(open(RES / "a5_precheck.json"))["hosts"][HOST]
    a = max(float(x) for x in A["admissible"]["smooth"])
    ok, msg = armguard(7, poll_s=60, max_wait_s=1800)
    if not ok:
        atomic_json(OUT / "smoke_full.json", dict(status="BLOCKED by armguard --need 7 after 30 minutes", armguard=msg)); return 3
    I, _, _, valid = load_host(HOST); cap, w = cap_px(HOST), mm_px(HOST, REG["stroke_mm"])
    lay = GL.plan_layout(valid, cap, w, host_seed(HOST), border=REG["border_px"])
    assert lay["letters"] == json.load(open(RES / "layouts.json"))[HOST]["letters"]
    g = GL.build(valid, cap, w, lay, sigma=REG["soft_sigma_px"]); geom = {k: g[k] for k in ("strokes", "counters", "far")}
    t = time.time(); v, info = inject(I, g["G"], a, "real", RS.host_libraries(HOST, "smooth"), host_seed(HOST), g["binary"]); inj_s = time.time() - t
    ct = DATA / "smoke_full" / HOST / f"smooth_a{a:g}" / "ct.zarr"
    t = time.time(); write_zarr(ct, v); wr_s = time.time() - t
    rec = dict(host=HOST, mpx=I.shape[1] * I.shape[2] / 1e6, a=a, amplitude_rule="largest grid a with <= 5% clipped on this host (A5), model-free",
               layout=dict(n=lay["n"], letters=lay["letters"], origin=lay["origin"]), input=dict(info, sha256=sha_array(v), inject_s=inj_s, zarr_write_s=wr_s),
               target_auc=target_auc(HOST), armguard=msg, runs={})
    P = {}
    for name, src in (("base_r1", HOSTS[HOST]["ct"]), ("base_r2", HOSTS[HOST]["ct"]), (f"smooth_a{a:g}", ct)):
        secs = infer(src, PRIMARY_CKPT, OUT / "pred" / f"{name}.tif"); P[name] = tifffile.imread(OUT / "pred" / f"{name}.tif")
        rec["runs"][name] = dict(secs=secs); print(f"DONE {name} {secs:.1f}s", flush=True)
    secs = [r["secs"] for r in rec["runs"].values()]
    rec["seconds_per_inference"] = dict(min=min(secs), median=float(np.median(secs)), max=max(secs), n=len(secs))
    rec["C0"] = dict(max_abs_D=int(np.abs(P["base_r1"].astype(np.int32) - P["base_r2"]).max()), pass_=bool((P["base_r1"] == P["base_r2"]).all()))
    ref = HOSTS[HOST]["ct"].parents[1]  # the stored ARM Z prediction for the same input, as a cross-check
    rec["max_abs_vs_stored_armz"] = int(np.abs(P["base_r1"].astype(np.int32) - tifffile.imread(RES.parents[0] / "zs/pred" / HOST / "w04" / "hybrid_3d2d-seed42_step-075000_forward.tif")).max())
    rec["score"] = MT.score(P[f"smooth_a{a:g}"], P["base_r1"], geom)
    figs.panels(OUT / f"smoke_full_a{a:g}.png", g["binary"], P["base_r1"], P[f"smooth_a{a:g}"], f"FLINJ full-size smoke {HOST}, A2 smoothed residual, a = {a:g}")
    rec["peak_rss_bytes_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; rec["status"] = "done"
    atomic_json(OUT / "smoke_full.json", rec)
    s = rec["score"]
    print(f"a {a:g} C {s['C']:.4f} C_a3 {s['C_a3']} d_prob {s['d_prob']:.5f} AUC_P {s['auc_P']:.4f} AUC_D {s['auc_D']:.4f} clip {info['clip_frac']:.4f} "
          f"C0 {rec['C0']} vs stored {rec['max_abs_vs_stored_armz']} secs {[round(x, 1) for x in secs]} rss {rec['peak_rss_bytes_self'] / 1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
