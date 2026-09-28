"""ARM FLINJ amendments 2 and 3, model-free pre-checks (no model output involved).

Per host: the layout under A1 plus C2 (every glyph pixel more than 48 px from host-labelled ink), or the reason the
host is excluded; for the primary and every S4 stroke width, the fraction of glyph pixels where b_host is defined under
C1 (valid, not-labelled-ink pixels of the 16 to 48 px annulus), the smallest distance from a glyph pixel to host ink,
and the clearance to the border and no-data; the C1 transplant library sizes and the clipped fraction of both
transplants (B1, B2); C3's clipped fraction against B4's 10%; S5 / S6 clipping at a = 0.5.

    /Users/andreashad04/money/vesuv/env/bin/python3 /Users/andreashad04/money/vesuv/ops/flinj/b_precheck.py
"""
import sys, time
import numpy as np
from scipy import ndimage as ndi
from common import HOSTS, REG, RES, atomic_json, banner, cap_px, host_seed, load_host, mm_px
import glyphs as GL, residuals as RS
from inject import c3_scale, host_background, inject, transplant


def main():
    banner(__file__); out = dict(host_bg_registered=REG["host_bg"], hosts={}, excluded={})
    for h in HOSTS:
        t = time.time(); I, ink, support, valid = load_host(h); cap = cap_px(h)
        lays = {w: GL.plan_layout(valid, cap, mm_px(h, w), host_seed(h), border=REG["border_px"], ink=ink) for w in (REG["stroke_mm"],) + tuple(REG["s4_stroke_mm"])}
        bg = valid & ~ink; d_ink = ndi.distance_transform_edt(~ink)
        d_edge = ndi.distance_transform_edt(np.pad(valid, 1, constant_values=False))[1:-1, 1:-1]
        r = dict(layouts={f"{w:g}": (dict(n=l["n"], letters=l["letters"], origin=l["origin"]) if l["axis"] else None) for w, l in lays.items()},
                 coverage={}, min_dist_to_ink={}, clearance={}, libs={})
        for w, l in lays.items():
            if l["axis"] is None: continue
            b = GL.build(valid, cap, mm_px(h, w), l)["binary"]; ys, xs = np.nonzero(b)
            box = [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]; st = b[box[0]:box[1], box[2]:box[3]]
            _, d = host_background(I, bg, box)
            r["coverage"][f"{w:g}"] = {REG["host_bg"]: float(d[st].mean())}
            r["min_dist_to_ink"][f"{w:g}"] = float(d_ink[b].min()); r["clearance"][f"{w:g}"] = float(d_edge[b].min() - 1)
        for kind in ("ink", "non"):
            r["libs"][kind] = {k: int(len(v)) for k, v in RS.host_transplant_libraries(h, kind).items()}
        if lays[REG["stroke_mm"]]["axis"] is None:
            out["excluded"][h] = "C2: no row of 4 letters fits 64 px clear of borders and no-data and more than 48 px from host ink"
            out["hosts"][h] = r; print(h, "EXCLUDED from the primary; S4 layouts", r["layouts"], "coverage", r["coverage"], flush=True); continue
        g = GL.build(valid, cap, mm_px(h, REG["stroke_mm"]), lays[REG["stroke_mm"]])
        for kind in ("ink", "non"):
            r[f"clip_{kind}_transplant"] = transplant(I, bg, g["binary"], RS.host_transplant_libraries(h, kind), host_seed(h), "real", 10 if kind == "ink" else 20)[1]["clip_frac"]
        r["C3"] = c3_scale(I, g["G"], RS.host_profiles(h), g["binary"])  # amendment 4, D2: reported, no bound
        for name, res, mode in (("S5", RS.host_libraries(h, "raw"), "real"), ("S6", RS.host_profiles(h), "profile")):
            r[f"clip_{name}_a0.5"] = inject(I, g["G"], REG["s56_a"], mode, res, host_seed(h), g["binary"])[1]["clip_frac"]
        out["hosts"][h] = r
        print(h, "layouts", {w: (v["n"] if v else None) for w, v in r["layouts"].items()}, "coverage", {w: v[REG["host_bg"]] for w, v in r["coverage"].items()},
              "min dist to ink", {w: round(v, 1) for w, v in r["min_dist_to_ink"].items()}, "clearance", {w: round(v) for w, v in r["clearance"].items()},
              "clip ink %.4f non %.4f C3 %.4f S5 %.4f S6 %.4f %.0fs" % (r["clip_ink_transplant"], r["clip_non_transplant"], r["C3"]["clip_frac"], r["clip_S5_a0.5"], r["clip_S6_a0.5"], time.time() - t), flush=True)
    out["B1_defined_everywhere"] = {h: all(v[REG["host_bg"]] == 1.0 for v in r["coverage"].values()) for h, r in out["hosts"].items()}
    atomic_json(RES / "b_precheck.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
