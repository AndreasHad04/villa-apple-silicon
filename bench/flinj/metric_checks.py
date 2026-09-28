"""ARM FLINJ phase 1: model-free metric checks M1 and M2, run before any model output is scored.

M1, ceiling: C of the soft glyph mask, and of the glyph mask blurred by the full measured eligible-scan blur,
must each be >= 0.8. M2, blob floor: C of the glyph mask blurred to letter scale (sigma = cap height / 3) must
be <= 0.2. Computed on every host's own layout and geometry masks.

"The full measured eligible-scan blur" has no second definition in the pre-registration or the repo
(results/fs/blur_calibration.json measures in-plane sigma 0.675 to 0.7 px). Both readings are computed and M1
must pass under BOTH: (a) sigma 0.7 px on the binary mask, which is the soft mask itself; (b) the soft mask
blurred again by the scan's 0.7 px, sigma = sqrt(0.7^2 + 0.7^2) = 0.99 px. A sigma sweep shows where C
would fall below 0.8, so the reading can be seen not to matter.

    /Users/andreashad04/money/vesuv/env/bin/python3 /Users/andreashad04/money/vesuv/ops/flinj/metric_checks.py
"""
import json, sys, time
import numpy as np
from scipy import ndimage as ndi
from common import HOSTS, REG, RES, atomic_json, banner, cap_px, host_seed, load_host, mm_px
import glyphs as GL
from metric import mask_contrast

READINGS = {"a_sigma_0.7_px": 0.7, "b_sigma_0.99_px": float(np.hypot(0.7, 0.7))}
SWEEP = [0.0, 0.35, 0.7, 0.99, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 45.0, 60.0]


def blur(binary, sigma):
    """Gaussian of a binary mask, zero outside the crop. FFT on a 4 sigma zero pad for large sigma."""
    if sigma == 0: return binary.astype(np.float64)
    if sigma < 10: return ndi.gaussian_filter(binary.astype(np.float64), sigma, mode="constant", truncate=4.0)
    pad = int(np.ceil(4 * sigma)); a = np.pad(binary.astype(np.float64), pad)
    ky, kx = np.fft.fftfreq(a.shape[0])[:, None], np.fft.rfftfreq(a.shape[1])[None, :]
    out = np.fft.irfft2(np.fft.rfft2(a) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (ky ** 2 + kx ** 2)), s=a.shape)
    return out[pad:-pad, pad:-pad]


def clearance(lab, valid):
    padded = np.pad(valid, 1, constant_values=False)
    d = ndi.distance_transform_edt(padded)[1:-1, 1:-1]
    return float(d[lab > 0].min() - 1) if (lab > 0).any() else float("nan")  # clear px between glyph and edge/no-data


def main():
    banner(__file__)
    out = dict(readings_sigma_px=READINGS, bars=dict(m1=REG["m1_bar"], m2=REG["m2_bar"], m2_sigma_caps=REG["m2_sigma_caps"]), hosts={})
    layouts, layouts4 = {}, {}
    for h in HOSTS:
        t = time.time(); _, ink, _, valid = load_host(h)
        cap, stroke = cap_px(h), mm_px(h, REG["stroke_mm"])
        lay = GL.plan_layout(valid, cap, stroke, host_seed(h), border=REG["border_px"], ink=ink)
        s4, layouts4[h] = {}, {}
        for w in REG["s4_stroke_mm"]:  # amendment 4, D3: each S4 width gets its own A1 + C2 layout
            lw = GL.plan_layout(valid, cap, mm_px(h, w), host_seed(h), border=REG["border_px"], ink=ink)
            if lw["axis"] is None:
                s4[f"{w}"] = dict(skipped="no row of 4 fits under A1 and C2 at this width"); layouts4[h][f"{w:g}"] = None; continue
            gw = GL.build(valid, cap, mm_px(h, w), lw, sigma=REG["soft_sigma_px"]); gg = {k: gw[k] for k in ("strokes", "counters", "far")}
            s4[f"{w}"] = dict(n=lw["n"], letters=lw["letters"], origin=lw["origin"], M1_soft=mask_contrast(gw["G"], gg)["C"], M2=mask_contrast(blur(gw["binary"], cap * REG["m2_sigma_caps"]), gg)["C"],
                              n_counters=int(gg["counters"].sum()), clearance_px=clearance(gw["lab"], valid), overlap_px=gw["overlap_px"])
            layouts4[h][f"{w:g}"] = dict(lw, glyphs=[list(x) for x in lw["glyphs"]])
        if lay["axis"] is None:
            out["hosts"][h] = dict(excluded="C2: no row of 4 letters fits 64 px clear of borders and no-data and more than 48 px from host ink", S4_descriptive=s4); print(h, "EXCLUDED from the primary; S4", {w: v.get("n", v.get("skipped")) for w, v in s4.items()}, flush=True); continue
        assert lay["axis"] is not None, f"{h}: no position holds a row of 7 at the registered geometry"
        g = GL.build(valid, cap, stroke, lay, sigma=REG["soft_sigma_px"])
        geom = {k: g[k] for k in ("strokes", "counters", "far")}
        r = dict(cap_px=cap, stroke_px=stroke, axis=lay["axis"], origin=lay["origin"], letters=lay["letters"],
                 angles_deg=[round(x[3], 3) for x in lay["glyphs"]], clearance_px=clearance(g["lab"], valid), overlap_px=g["overlap_px"],
                 n_strokes=int(geom["strokes"].sum()), n_counters=int(geom["counters"].sum()), n_far=int(geom["far"].sum()),
                 far_fraction_of_valid=float(geom["far"].sum() / valid.sum()))
        r["M1_soft"] = mask_contrast(g["G"], geom)["C"]
        r["M1_full_blur"] = {k: mask_contrast(blur(g["binary"], s), geom)["C"] for k, s in READINGS.items()}
        r["M2_sigma_px"] = cap * REG["m2_sigma_caps"]; r["M2"] = mask_contrast(blur(g["binary"], r["M2_sigma_px"]), geom)["C"]
        r["sweep"] = {f"{s:g}": mask_contrast(blur(g["binary"], s), geom)["C"] for s in SWEEP}
        r["S4_descriptive"] = s4
        r["pass_M1"] = bool(r["M1_soft"] >= REG["m1_bar"] and all(v >= REG["m1_bar"] for v in r["M1_full_blur"].values()))
        r["pass_M2"] = bool(r["M2"] <= REG["m2_bar"])
        out["hosts"][h] = r; layouts[h] = lay
        print(h, lay["axis"], lay["origin"], lay["letters"], "M1 soft %.4f full %s M2 %.4f (sigma %.1f px) far %d px, %.1fs" % (
            r["M1_soft"], {k: round(v, 4) for k, v in r["M1_full_blur"].items()}, r["M2"], r["M2_sigma_px"], r["n_far"], time.time() - t), flush=True)
    inc = [v for v in out["hosts"].values() if "excluded" not in v]  # C2-excluded hosts carry no glyphs
    out["M1_pass"] = bool(inc) and all(v["pass_M1"] for v in inc)
    out["M2_pass"] = bool(inc) and all(v["pass_M2"] for v in inc)
    out["verdict"] = "PASS: model runs may proceed" if out["M1_pass"] and out["M2_pass"] else "FAIL: stop, the metric must be redefined by amendment"
    atomic_json(RES / "metric_checks.json", out)
    atomic_json(RES / "layouts_s4.json", layouts4)
    atomic_json(RES / "layouts.json", {h: dict(v, glyphs=[list(x) for x in v["glyphs"]]) for h, v in layouts.items()} | {h: dict(excluded=v["excluded"], n=0) for h, v in out["hosts"].items() if "excluded" in v})
    print(out["verdict"], flush=True)
    return 0 if out["M1_pass"] and out["M2_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
