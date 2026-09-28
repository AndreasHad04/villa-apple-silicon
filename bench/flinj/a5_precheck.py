"""ARM FLINJ amendment 1, model-free pre-checks of A1 and A5 (no model output involved).

Per host: the A1 layout; the fraction of stroke voxels clipped (A5) at every grid amplitude for the primary smoothed
residual (A2), the raw residual (S5) and the depth profile (S6); and C3's scale, mean change and clipped fraction.
A5 bounds a grid unit at 5% clipped (excluded from a* above it) and C3 at 1%. run_all refuses while C3 is infeasible.

    /Users/andreashad04/money/vesuv/env/bin/python3 /Users/andreashad04/money/vesuv/ops/flinj/a5_precheck.py
"""
raise SystemExit(f"{__file__}: superseded by amendments 2 and 3; kept as the record of what it measured")
import sys, time
from common import HOSTS, REG, RES, atomic_json, banner, cap_px, host_seed, load_host, mm_px
import glyphs as GL, residuals as RS
from inject import c3_scale, inject


def main():
    banner(__file__); out = dict(clip_max=REG["clip_max"], c3_clip_max=REG["c3_clip_max"], c3_mean_change=REG["c3_mean_change"], hosts={})
    for h in HOSTS:
        t = time.time(); I, _, _, valid = load_host(h); cap, w = cap_px(h), mm_px(h, REG["stroke_mm"])
        lay = GL.plan_layout(valid, cap, w, host_seed(h), border=REG["border_px"]); g = GL.build(valid, cap, w, lay, sigma=REG["soft_sigma_px"])
        libs = dict(smooth=RS.host_libraries(h, "smooth"), raw=RS.host_libraries(h, "raw"), profile=RS.host_profiles(h))
        clip = {k: {f"{a:g}": inject(I, g["G"], a, "profile" if k == "profile" else "real", libs[k], host_seed(h), g["binary"])[1]["clip_frac"]
                    for a in REG["grid"]} for k in libs}
        c3 = c3_scale(I, g["G"], libs["profile"], g["binary"])
        r = dict(n_letters=lay["n"], letters=lay["letters"], origin=lay["origin"], clip=clip,
                 admissible={k: [a for a, c in v.items() if c <= REG["clip_max"]] for k, v in clip.items()},
                 C3=dict(c3, feasible=bool(c3["clip_frac"] <= REG["c3_clip_max"])),
                 profile={k: [float(x) for x in v] for k, v in libs["profile"].items()})
        out["hosts"][h] = r
        print(h, "n", lay["n"], "admissible", r["admissible"], "C3 scale %.3f clip %.4f feasible %s %.0fs" % (c3["scale"], c3["clip_frac"], r["C3"]["feasible"], time.time() - t), flush=True)
    out["C3_feasible_all"] = all(v["C3"]["feasible"] for v in out["hosts"].values())
    atomic_json(RES / "a5_precheck.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
