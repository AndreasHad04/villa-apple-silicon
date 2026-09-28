"""ARM FLINJ amendment 2, full-size PLUMBING smoke pair on ag174: the B1 ink transplant and the B2 non-ink transplant
of the whole crop, primary checkpoint, forward. It measures seconds per inference, peak memory, that the two inputs
are identical outside the glyph mask, and that D = P(ink) - P(non-ink) is exactly 0 on the far background.

Host and donor background per amendment 3, C1 (valid, not-labelled-ink annulus pixels), layout per A1 plus C2. As the
coordinator asked, the outcome (C, the pair AUC, any stroke statistic) is NOT computed or written here. Gate: armguard check --need 7, polled every 60 s for up to 30 minutes.

    /usr/bin/time -l -o /Users/andreashad04/money/vesuv/results/flinj/smoke_pair/time_l.txt \
        /Users/andreashad04/money/vesuv/env/bin/python3 /Users/andreashad04/money/vesuv/ops/flinj/smoke_pair.py
"""
import json, resource, sys, time
import numpy as np, tifffile
from common import DATA, PRIMARY_CKPT, REG, RES, armguard, atomic_json, banner, cap_px, host_seed, infer, load_host, mm_px, sha_array, write_zarr
import glyphs as GL, residuals as RS
from inject import transplant

HOST = "ag174"; OUT = RES / "smoke_pair"


def main():
    banner(__file__); OUT.mkdir(parents=True, exist_ok=True)
    ok, msg = armguard(7, poll_s=60, max_wait_s=1800)
    if not ok:
        atomic_json(OUT / "smoke_pair.json", dict(status="BLOCKED by armguard --need 7 after 30 minutes", armguard=msg)); return 3
    I, ink, support, valid = load_host(HOST); cap, w = cap_px(HOST), mm_px(HOST, REG["stroke_mm"])
    lay = GL.plan_layout(valid, cap, w, host_seed(HOST), border=REG["border_px"], ink=ink); g = GL.build(valid, cap, w, lay, sigma=REG["soft_sigma_px"])
    bg = valid & ~ink
    rec = dict(host=HOST, mpx=I.shape[1] * I.shape[2] / 1e6, host_background=REG["host_bg"] + " (amendment 3, C1)", armguard=msg, inputs={}, runs={})
    V = {}
    for kind in ("ink", "non"):
        t = time.time(); v, info = transplant(I, bg, g["binary"], RS.host_transplant_libraries(HOST, kind), host_seed(HOST), "real", 10 if kind == "ink" else 20)
        b = time.time() - t; ct = DATA / "smoke_pair" / HOST / kind / "ct.zarr"; t = time.time(); write_zarr(ct, v)
        rec["inputs"][kind] = dict(info, sha256=sha_array(v), build_s=b, zarr_write_s=time.time() - t); V[kind] = v
    rec["inputs_identical_outside_glyph"] = bool((V["ink"][:, ~g["binary"]] == V["non"][:, ~g["binary"]]).all() and (V["ink"][:, ~g["binary"]] == I[:, ~g["binary"]]).all())
    P = {}
    for kind in ("ink", "non"):
        out = OUT / "pred" / f"{kind}.tif"; secs = infer(DATA / "smoke_pair" / HOST / kind / "ct.zarr", PRIMARY_CKPT, out)
        P[kind] = tifffile.imread(out); rec["runs"][kind] = dict(secs=secs); print(f"DONE {kind} {secs:.1f}s", flush=True)
    secs = [r["secs"] for r in rec["runs"].values()]
    rec["seconds_per_inference"] = dict(min=min(secs), median=float(np.median(secs)), max=max(secs), n=len(secs))
    D = P["ink"].astype(np.int32) - P["non"]; far = g["far"]
    rec["max_abs_D_far"] = int(np.abs(D[far]).max()); rec["n_far"] = int(far.sum())
    rec["peak_rss_bytes_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; rec["status"] = "done"
    atomic_json(OUT / "smoke_pair.json", rec)
    print(f"far max |D| {rec['max_abs_D_far']} over {rec['n_far']} px; inputs identical outside glyph {rec['inputs_identical_outside_glyph']}; "
          f"clip ink {rec['inputs']['ink']['clip_frac']:.4f} non {rec['inputs']['non']['clip_frac']:.4f}; secs {[round(s, 1) for s in secs]}; rss {rec['peak_rss_bytes_self'] / 1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
