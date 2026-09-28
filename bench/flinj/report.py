"""ARM FLINJ phase 1 as amended (amendments 1 and 2): generates results/flinj/FLINJ_REPORT.md and results/flinj/figs/
from the unit TIFFs and JSONs. Every number is computed here from source files; none is typed. Works on partial
results; no verdict is printed until its inputs exist.

    /Users/andreashad04/money/vesuv/env/bin/python3 /Users/andreashad04/money/vesuv/ops/flinj/report.py
"""
import hashlib, json, math, os, pathlib, time
import numpy as np, tifffile
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image
from common import AMEND2_SHA, AMEND3_SHA, AMEND4_SHA, AMEND_SHA, HOSTS, PREREG_SHA, REG, RES, banner, cap_px, host_config, host_seed, load_host, mm_px, reference_pred, target_auc
import figs, glyphs as GL, metric as MT
from run_all import PRED, S3_CKPTS, UNITS, ckpt_key, units

FIGS = RES / "figs"
DECISIONS = [
    "Amendment 1, A7: Arial skeletons; cap height = outer height of the redrawn Eta; rotation per letter; 32 px patch grid; "
    "PHerc0009B split at W // 2 with the donor half chosen per pixel; S1 hulls per connected label component; S4 keeps the layout.",
    "A1 as implemented fits 4 letters on w00, 6 on ag144, 7 on ag174, p0009b and p0500p2 (the first n of each host's 7 seeded draws).",
    "B1 / B2 patches: grid cells WHOLLY inside support with b defined at every pixel; B1 >= 90% labelled ink; B2 every pixel "
    "farther than 48 px from any label. The transplant carries r = I - b at every pixel of a patch (the donor column as it is).",
    "C1 / C2 kept from the registration and moved onto the transplant (the ink residual's planes permuted per patch / replaced by "
    "its patch mean), each scored against the B2 non-ink transplant.",
    "The B3 floor applied to C3 as a pair (C3 against the untouched host): d >= 0.02 and AUC of P(C3) against P(host) over strokes "
    ">= 0.55. S5 and S6 keep amendment 1's A3 floor (stroke-versus-far AUC).",
    "Verdict precedence: VOID (C3 below D1's ceil(0.75 x included hosts)) first, then NOT DETECTABLE (fewer than 3 defined), then the 0.5 / 0.2 thresholds.",
    "C2 read as MORE than 48 px from host ink, so no 16 to 48 px annulus (inclusive at 48) reaches it; that also satisfies 'at least 48 px'.",
    "C1 applies to the transplants (B1, B2) as it names them; S5 and S6 / C3 keep the registered residual library.",
    "D4: the C2-excluded host still runs base_r1 / base_r2 (C0, S1) and each S4 width whose own layout fits, with its own halves as donors.",
    "C3 scale by bisection to +20 grey levels mean change on stroke voxels; clipped fraction reported with no bound (D2).",
    "Clipped fraction: stroke pixels x all 21 planes whose rounded value fell outside [0, 255]. D5 counters over several glyphs: union "
    "of hulls minus the union of glyphs dilated by half a stroke. D6 masks restricted to pixels where the surface volume holds data.",
]


def fmt(v, d=4):
    if v is None: return "not run"
    if isinstance(v, (bool, np.bool_)): return "yes" if v else "no"
    if isinstance(v, str): return v
    return "nan" if not np.isfinite(v) else f"{v:.{d}f}"


def cb(s, key="b3"):
    if s is None: return "not run"
    ok = s[f"{key}_defined"]; auc = s["auc_pair"] if key == "b3" else s["auc_P"]
    return fmt(s[f"C_{key}"]) if ok else f"UNDEFINED (C {fmt(s['C'])}, d {fmt(s['d_prob'])}, AUC {fmt(auc)})"


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def tif(h, name):
    f = PRED / h / f"{name}.tif"
    return tifffile.imread(f) if (UNITS / h / f"{name}.json").exists() and f.exists() else None


def real_geometry(h, label, support, valid):
    cap, r = cap_px(h), mm_px(h, REG["stroke_mm"]) / 2.0
    cc, _ = ndi.label(label, structure=np.ones((3, 3), bool)); hull = np.zeros_like(label)
    for k, sl in enumerate(ndi.find_objects(cc)):
        if sl is not None: hull[sl] |= convex_hull_image(cc[sl] == k + 1)
    dist = ndi.distance_transform_edt(~label); ok = support & valid
    return dict(strokes=label & ok, counters=hull & (dist > r) & ok, far=(dist > REG["far_caps"] * cap) & ok)


def host_results(h):
    _, label, support, valid = load_host(h); cap = cap_px(h); ink = label
    lays = {w: GL.plan_layout(valid, cap, mm_px(h, w), host_seed(h), border=REG["border_px"], ink=ink) for w in (REG["stroke_mm"],) + tuple(REG["s4_stroke_mm"])}
    ref, ref4 = json.load(open(RES / "layouts.json"))[h], json.load(open(RES / "layouts_s4.json"))[h]
    for w, l in lays.items():
        r_ = ref if w == REG["stroke_mm"] else ref4[f"{w:g}"]
        if l["axis"] is None: assert r_ is None or r_.get("excluded"), (h, w)
        else: assert (l["letters"], l["origin"], l["n"]) == (r_["letters"], r_["origin"], r_["n"]), f"{h} {w}: layout differs from the recorded one"
    excl = lays[REG["stroke_mm"]]["axis"] is None
    geo = {w: GL.build(valid, cap, mm_px(h, w), l, sigma=REG["soft_sigma_px"]) for w, l in lays.items() if l["axis"] is not None}
    G = {w: {k: g[k] for k in ("strokes", "counters", "far")} for w, g in geo.items()}
    R = dict(excluded="C2: no row of 4 letters fits more than 48 px from host ink and 64 px clear of borders and no-data" if excl else None,
             layout=dict(n=0, letters="", origin=None) if excl else lays[REG["stroke_mm"]], layouts_s4={f"{w:g}": (l["letters"] if l["axis"] else None) for w, l in lays.items() if w != REG["stroke_mm"]},
             target=target_auc(h), units_done=sum((UNITS / h / f"{n}.json").exists() for n, _ in units(h)), units_total=len(units(h)), pairs={}, clip={})
    for n, _ in units(h):
        if (UNITS / h / f"{n}.json").exists():
            u = json.load(open(UNITS / h / f"{n}.json"))
            if "input" in u: R["clip"][n] = u["input"]["clip_frac"]
    P = lambda n: tif(h, n); P0 = P("base_r1")
    if P0 is not None and P("base_r2") is not None: R["C0_max_abs_D"] = int(np.abs(P0.astype(np.int32) - P("base_r2")).max())
    if P0 is not None and reference_pred(h).exists(): R["vs_stored_max_abs"] = int(np.abs(P0.astype(np.int32) - tifffile.imread(reference_pred(h))).max())
    pairs = {}
    if not excl:
        g0 = G[REG["stroke_mm"]]; pairs.update(primary=("b1_ink", "b2_non", g0), c1=("c1_shuffle", "b2_non", g0), c2=("c2_flat", "b2_non", g0), c3=("c3_clean", "base_r1", g0))
    pairs.update({f"s4_{w:g}": (f"s4_w{w:g}_ink", f"s4_w{w:g}_non", G[w]) for w in REG["s4_stroke_mm"] if w in G})
    for k, (a, b, gg) in pairs.items():
        if P(a) is not None and P(b) is not None: R["pairs"][k] = MT.score_pair(P(a), P(b), gg)
    if not excl:
        for k in ("s5_raw", "s6_profile"):
            if P(k) is not None and P0 is not None: R["pairs"][k] = MT.score(P(k), P0, G[REG["stroke_mm"]])
        ink_ = [P("b1_ink")] + [P(f"s3_ink_{ckpt_key(c)}") for c in S3_CKPTS]; non = [P("b2_non")] + [P(f"s3_non_{ckpt_key(c)}") for c in S3_CKPTS]
        if all(p is not None for p in ink_ + non):
            R["pairs"]["s3"] = MT.score_pair(np.sum([p.astype(np.int64) for p in ink_], 0), np.sum([p.astype(np.int64) for p in non], 0), G[REG["stroke_mm"]], unit=255.0 * len(ink_))
    if P0 is not None: R["S1"] = MT.contrast(P0.astype(np.int32), real_geometry(h, label, support, valid))
    FIGS.mkdir(parents=True, exist_ok=True)
    if not excl:
        for k, (a, b, labels) in dict(primary=("b1_ink", "b2_non", ("P(non-ink transplant)", "P(ink transplant)")), c3=("c3_clean", "base_r1", ("P(host)", "P(C3)"))).items():
            if P(a) is not None and P(b) is not None:
                figs.panels(FIGS / f"{h}_{k}.png", geo[REG["stroke_mm"]]["binary"], P(b), P(a), f"FLINJ {h} {k}", labels=labels)
    return R


def verdict(H):
    """Amendment 4 D1 (VOID unless C3 >= 0.5 with the B3 floor on ceil(0.75 x included hosts)), then B3 (NOT DETECTABLE
    below 3 hosts with a defined C), then the registered thresholds. C2-excluded hosts are listed, not counted."""
    inc = [h for h in H if not H[h].get("excluded")]
    L = [f"- {h}: excluded from the primary, {H[h]['excluded']}." for h in H if H[h].get("excluded")]
    c3 = {h: H[h]["pairs"].get("c3") for h in inc}
    if any(v is None for v in c3.values()): return None, L + ["Not complete: C3 has not run on every included host."]
    need = math.ceil(REG["c3_min_frac"] * len(inc)); n3 = sum(bool(v["b3_defined"] and v["C_b3"] >= REG["c3_bar"]) for v in c3.values())
    L.append(f"C3 positive control: C (with the B3 floor) >= {REG['c3_bar']} on {n3} of {len(inc)} included hosts (D1 needs {need}).")
    if n3 < need: return "VOID", L
    pr = {h: H[h]["pairs"].get("primary") for h in inc}
    if any(v is None for v in pr.values()): return None, L + ["Not complete: the B1 / B2 pair has not run on every included host."]
    d = {h: v["C_b3"] for h, v in pr.items() if v["b3_defined"]}
    L += [f"- {h}: C UNDEFINED under B3, left out." for h in pr if h not in d]
    if len(d) < REG["b3_min_hosts"]:
        return "NOT DETECTABLE", L + [f"{len(d)} hosts have a defined C (B3 needs {REG['b3_min_hosts']}): the model does not respond to real ink in letter shape above the floor."]
    med = float(np.median(list(d.values())))
    v = "STROKES" if med >= REG["strokes_bar"] else "BLOBS" if med <= REG["blobs_bar"] else "MIXED"
    return v, L + [f"Median C over {len(d)} hosts with a defined C: {med:.4f}. STROKES if >= {REG['strokes_bar']}, BLOBS if <= {REG['blobs_bar']}, MIXED otherwise."]


def main():
    banner(__file__)
    M, B = json.load(open(RES / "metric_checks.json")), json.load(open(RES / "b_precheck.json"))
    SP = json.load(open(RES / "smoke_pair" / "smoke_pair.json")) if (RES / "smoke_pair" / "smoke_pair.json").exists() else {}
    SF = json.load(open(RES / "smoke_full" / "smoke_full.json")) if (RES / "smoke_full" / "smoke_full.json").exists() else {}
    H = {h: host_results(h) for h in HOSTS}; me = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:16]
    L = ["# ARM FLINJ phase 1 report (as amended)", "",
         f"Generated {time.strftime('%Y-%m-%dT%H:%M:%S')} by ops/flinj/report.py (sha256 {me}). Pre-registration {PREREG_SHA}; amendment 1 {AMEND_SHA}; "
         f"amendment 2 {AMEND2_SHA}; amendment 3 {AMEND3_SHA}; amendment 4 {AMEND4_SHA}. Amendment 2 replaces the additive injection with a background-matched transplant of real ink (B1) against a "
         "matched non-ink transplant (B2), scores C on their difference with the B3 floor, retires the amplitude grid, keeps C3 with a 10% clip "
         "bound (B4) and adds S7. Amendment 3 uses one background (valid, not-labelled-ink annulus pixels) for donor and host (C1) and keeps "
         "rows more than 48 px from host ink (C2). Amendment 4 judges C3 on ceil(0.75 x included hosts) (D1) with no clip bound (D2), gives each "
         "S4 width its own layout (D3) and states the limit below (D4). Every number below is computed from the result files.", ""]
    exc = [h for h, r in H.items() if r.get("excluded")]; inc_cfg = sorted({host_config(h) for h in HOSTS if h not in exc})
    L += [f"**Limit (D4):** excluded from the primary: " + (", ".join(f"{h} ({host_config(h)[0]:g} um, {host_config(h)[1]} keV)" for h in exc) or "none") +
          ". The primary speaks only for " + ", ".join(f"{u:g} um / {k} keV" for u, k in inc_cfg) + " scans.", ""]
    L += ["## Status", ""] + table(["host", "letters", "units done", "of"], [[h, r["layout"]["letters"], r["units_done"], r["units_total"]] for h, r in H.items()]) + [""]
    L += ["## Model-free checks", "", f"M1 / M2: {M['verdict']}.", ""] + table(["host", "M1 soft", "M1 sigma 0.99", "M2", "M2 0.25 mm", "M2 0.5 mm", "M2 0.7 mm"],
        [[h, fmt(r["M1_soft"]), fmt(r["M1_full_blur"]["b_sigma_0.99_px"]), fmt(r["M2"])] + [fmt(r["S4_descriptive"][w]["M2"]) for w in ("0.25", "0.5", "0.7")] if "excluded" not in r else [h, r["excluded"]] + [""] * 5 for h, r in M["hosts"].items()]) + [""]
    L += [f"b over '{B['host_bg_registered']}' annulus pixels for donor and host (C1); rows more than 48 px from host ink (C2); C3 clipped fraction reported without a bound (D2).", ""] + table(
        ["host", "letters 0.35 / 0.25 / 0.5 / 0.7 mm", "b_host coverage", "min distance to host ink px", "clipped ink / non-ink transplant", "C3 clipped", "S5 clipped", "S6 clipped"],
        [[h, " / ".join(str(v["n"]) if v else "none" for v in r["layouts"].values()), " / ".join(fmt(c[B["host_bg_registered"]]) for c in r["coverage"].values()),
          " / ".join(fmt(v, 1) for v in r["min_dist_to_ink"].values()), f"{fmt(r.get('clip_ink_transplant'))} / {fmt(r.get('clip_non_transplant'))}",
          fmt(r.get("C3", {}).get("clip_frac")), fmt(r.get("clip_S5_a0.5")), fmt(r.get("clip_S6_a0.5"))] for h, r in B["hosts"].items()]) + [""]
    if SP.get("status") == "done":
        sp = SP["seconds_per_inference"]
        L += ["## Plumbing smoke pair (unregistered host background, outcome not computed)", "",
              f"{SP['host']} whole crop: far-background max |D| = {SP['max_abs_D_far']} over {SP['n_far']} px; inputs identical outside the glyph mask: "
              f"{fmt(SP['inputs_identical_outside_glyph'])}; seconds per inference {sp['min']:.1f} to {sp['max']:.1f}. Full-size C0 (smoke_full): max |D| = {SF.get('C0', {}).get('max_abs_D', 'not run')}.", ""]
    L += ["## C0, determinism", ""] + table(["host", "max |D| run 1 vs run 2", "max |diff| vs stored ARM Z / F6"],
                                            [[h, r.get("C0_max_abs_D", "not run"), r.get("vs_stored_max_abs", "not run")] for h, r in H.items()]) + [""]
    v, why = verdict(H)
    L += ["## Primary (B3): D = P(ink transplant) - P(non-ink transplant)", ""] + table(
        ["host", "C", "C with the B3 floor", "d (prob)", "AUC ink vs non-ink on strokes", "far max |D|", "clipped ink", "clipped non-ink"],
        [[h, fmt(p["C"]), cb(p), fmt(p["d_prob"]), fmt(p["auc_pair"]), p["max_abs_D_far"], fmt(r["clip"].get("b1_ink")), fmt(r["clip"].get("b2_non"))]
         if (p := r["pairs"].get("primary")) else ([h, "excluded (C2)"] + [""] * 6 if r.get("excluded") else [h] + ["not run"] * 7) for h, r in H.items()]) + [""] + why + [""]
    L += [f"**Primary verdict: {v}.**" if v else "**Primary verdict: not yet computable.**", ""]
    L += ["## S7, realism", ""] + table(["host", "ink vs non-ink stroke AUC", "real held-out AUC", "difference"],
        [[h, fmt(r["pairs"]["primary"]["auc_pair"]), fmt(r["target"]), fmt(r["pairs"]["primary"]["auc_pair"] - r["target"])] if "primary" in r["pairs"] else [h, "not run", fmt(r["target"]), ""] for h, r in H.items()]) + [""]
    L += ["## Controls", ""] + table(["host", "C1 (A3)", "C1 AUC", "C2 (A3)", "C2 AUC", "C3 C (floor)", "C3 AUC", "C3 clipped", "C3 far max |D|"],
        [[h, cb(r["pairs"].get("c1")), fmt(r["pairs"].get("c1", {}).get("auc_pair")), cb(r["pairs"].get("c2")), fmt(r["pairs"].get("c2", {}).get("auc_pair")),
          cb(r["pairs"].get("c3")), fmt(r["pairs"].get("c3", {}).get("auc_pair")), fmt(r["clip"].get("c3_clean")), r["pairs"].get("c3", {}).get("max_abs_D_far", "not run")] for h, r in H.items()]) + [""]
    L += ["## Secondary", "", "S1, C on the REAL labelled letters, on P(host):", ""] + table(["host", "C", "mean P strokes", "mean P counters", "mean P far"],
        [[h, fmt(r["S1"]["C"]), fmt(r["S1"]["mean_strokes"], 2), fmt(r["S1"]["mean_counters"], 2), fmt(r["S1"]["mean_far"], 2)] if "S1" in r else [h] + ["not run"] * 4 for h, r in H.items()]) + [""]
    L += ["S3, 14-checkpoint mean on the transplant pair: " + "; ".join(f"{h} {cb(r['pairs'].get('s3'))}" for h, r in H.items()) + ".", ""]
    def s4cell(h, r, w):
        m = M["hosts"][h]; m2 = m.get("M2") if w == "0.35" else m.get("S4_descriptive", {}).get(w, {}).get("M2")
        if m2 is None: return "excluded (C2)" if w == "0.35" else m.get("S4_descriptive", {}).get(w, {}).get("skipped", "not run")
        return f"{cb(r['pairs'].get('primary' if w == '0.35' else f's4_{w}'))} / {fmt(m2)}"
    L += ["S4 on the transplant pair, each width on its own A1 + C2 layout (D3), each C beside its own M2 (A6: no verdict from S4):", ""] + table(
        ["host"] + [f"{w} mm C / M2" for w in ("0.25", "0.35", "0.5", "0.7")], [[h] + [s4cell(h, r, w) for w in ("0.25", "0.35", "0.5", "0.7")] for h, r in H.items()]) + [""]
    L += ["S5 (raw additive residual) and S6 (mean depth profile) at a = 0.5, against the host, A3 floor:", ""] + table(["host", "S5 C", "S5 AUC", "S5 clipped", "S6 C", "S6 AUC", "S6 clipped"],
        [[h, cb(r["pairs"].get("s5_raw"), "a3"), fmt(r["pairs"].get("s5_raw", {}).get("auc_P")), fmt(r["clip"].get("s5_raw")),
          cb(r["pairs"].get("s6_profile"), "a3"), fmt(r["pairs"].get("s6_profile", {}).get("auc_P")), fmt(r["clip"].get("s6_profile"))] for h, r in H.items()]) + [""]
    L += ["## Decisions (fixed before any registered model output was scored)", ""] + [f"- {d}" for d in DECISIONS]
    L += ["", "Primary layouts: " + "; ".join((f"{h} excluded (C2)" if r.get("excluded") else f"{h} {r['layout']['n']} letters {r['layout']['letters']} at {r['layout']['origin']}") for h, r in H.items()) + ".",
          "S4 layouts (letters per width): " + "; ".join(f"{h} {r['layouts_s4']}" for h, r in H.items()) + ".", ""]
    L += ["## Figures", ""] + [f"- figs/{f.name}" for f in sorted(FIGS.glob("*.png"))] + [""]
    text = "\n".join(L); assert chr(0x2014) not in text and chr(0x2013) not in text
    tmp = RES / f"FLINJ_REPORT.md.partial{os.getpid()}"; tmp.write_text(text); os.replace(tmp, RES / "FLINJ_REPORT.md")
    tmp = RES / f"report_data.json.partial{os.getpid()}"
    json.dump({h: {k: v for k, v in r.items() if k != "layout"} for h, r in H.items()}, open(tmp, "w"), indent=1, default=float); os.replace(tmp, RES / "report_data.json")
    print(text)


if __name__ == "__main__":
    main()
