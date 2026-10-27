"""Re-derive every number quoted in the public artifacts from the raw results.

The README, both PR bodies and SUBMISSION.md are generated or hand-edited from
the JSON in results/. This asserts that what they SAY matches what was
MEASURED, so a stale number cannot survive an edit.
"""
import json, pathlib, re, sys
R = pathlib.Path(__file__).resolve().parent.parent
J = lambda n: json.loads((R/"results"/n).read_text())
checks, failed = [], 0

def ck(name, ok, detail=""):
    global failed
    if not ok: failed += 1
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def has(text, *frags):
    return all(f in text for f in frags)

# ONE file, TWO homes: ~/money/vesuv/ops/ in the working tree, and bench/ in the
# published repo. Resolve the layout instead of keeping two copies, because the
# published copy silently diverged and then crashed on its own paths.
_PUB = R/"public"/"README.md"
README = (_PUB if _PUB.exists() else R/"README.md").read_text()
def _opt(p): return p.read_text() if p.exists() else None
PR1 = _opt(R/"results"/"PR_BODY.md")
PR2 = _opt(R/"results"/"PR2_BODY.md")
SUB = _opt(R/"SUBMISSION.md")
ALL = {k: v for k, v in (("README", README), ("PR1812", PR1),
                         ("PR1813", PR2), ("SUBMISSION", SUB)) if v is not None}
_absent = [k for k, v in (("PR1812", PR1), ("PR1813", PR2), ("SUBMISSION", SUB)) if v is None]
if _absent:
    print(f"  NOTE: not published in this repo, so not checked here: {', '.join(_absent)}")
def present(*texts): return [t for t in texts if t is not None]

print("SPEED AND MEMORY")
bs = J("bench_stats.json")
row = {r["amp_device"]+"|"+r["device"]: r for r in bs["rows"]}
cpu, fix = row["off|cpu"], row["mps|mps"]
ck("CPU median matches bench_stats", f"{cpu['median_s']:.3f}" == "3.261", f"{cpu['median_s']:.3f}")
ck("MPS corrected median matches", f"{fix['median_s']:.3f}" == "0.427", f"{fix['median_s']:.3f}")
sp = fix["speedup_vs_cpu_median"]
# PR1813 is the window PR and deliberately does not quote the port speedup,
# so requiring it there was a defect in this check, not in the artifact.
SPEED_ARTIFACTS = {k: ALL[k] for k in ("README", "PR1812", "SUBMISSION") if k in ALL}
ck("speedup is 7.63 and is quoted in every artifact that claims it", sp == 7.63 and
   all(str(sp) in t for t in SPEED_ARTIFACTS.values()), f"{sp}")
ck("the superseded mean-based 7.85x speedup appears nowhere as a SPEEDUP",
   all(not re.search(r"7\.85\s*x|speedup[^.]{0,40}7\.85", t) for t in ALL.values()),
   "7.85 survives only as the 7.85 GB memory figure")
# This used to load mem_probe_summary.json and never use it, and the file was
# never written, so every memory figure in the artifacts was typed text that
# nothing checked. bench/mem_summary.py now generates it.
mp = J("mem_probe_summary.json")
_m = {f"{r['device']}{r['batch']}": r for r in mp["rows"]}
for _k, _label in (("cpu1", "CPU batch 1"), ("cpu2", "CPU batch 2"),
                   ("mps1", "MPS batch 1"), ("mps4", "MPS batch 4")):
    _g = f"{_m[_k]['forward_memory_gb']} GB"
    ck(f"{_label} forward memory {_g} is quoted in the README", _g in README, _g)
ck("each row measured in its own process (ru_maxrss is a process high-water mark)",
   mp["fresh_process_per_row"] is True)
ck("CPU and MPS memory are labelled as the different quantities they are",
   {r["forward_memory_is"] for r in mp["rows"]} ==
   {"forward_rss_delta_gb", "mps_driver_allocated_gb"})
_lean = mp["leaner_x_at_batch1"]
ck(f"leaner factor at batch 1 is {_lean}x and no artifact claims the superseded ~6x",
   abs(_lean - round(_m["cpu1"]["forward_memory_gb"]/_m["mps1"]["forward_memory_gb"], 2)) < 1e-9
   and all("6x less memory" not in t and "~6x leaner" not in t for t in ALL.values()),
   f"{_lean}x")

print("\nWINDOW RECOVERY")
o = J("offset_vs_reference_prof.json")
one = next(r["pearson_vs_reference"] for r in o["rows"] if r["z_window"][0] == 1)
ck("PHerc1667 r at START_LAYER 1", f"{one:.4f}" == "0.8521", f"{one:.4f}")
ck("PHerc1667 best", o["verdict"]["best_z_window"][0] == 25 and
   f"{o['verdict']['best_pearson']:.4f}" == "0.9851", str(o["verdict"]["best_pearson"]))
for tag, exp1, expb, expr in (("g0139", 0.9054, 23, 0.99),
                              ("gparis", 0.6240, 24, 0.9698),
                              ("g0814", 0.8263, 21, 0.9005)):
    v = J(f"window_generalise_{tag}.json")["verdict"]
    ck(f"{tag}: r@1 {exp1}, best {expb}, r {expr}",
       abs(v["readme_start_layer_1_pearson"]-exp1) < 5e-5 and v["best_start_layer"] == expb
       and abs(v["best_pearson"]-expr) < 5e-5,
       f"{v['readme_start_layer_1_pearson']} / {v['best_start_layer']} / {v['best_pearson']}")
regions = [J(f"window_generalise_r1667_{i}.json")["verdict"]["best_start_layer"] for i in (1,2,3)]
ck("three further PHerc1667 regions all peak at 24", regions == [24,24,24], str(regions))
allbest = [25,23,24,21] + regions
ck("all seven optima lie in 21..25", all(21 <= b <= 25 for b in allbest), str(allbest))
ck("the 21 to 25 range is stated", all("21 to 25" in t for t in present(README, PR2, SUB)))

print("\nSHUFFLE FLOORS (the null that makes the peak meaningful)")
ty = J("window_tally.json")["summary"]
ck(f"shuffle floor is {ty['max_shuffle_floor']} and is quoted as such",
   all(str(ty["max_shuffle_floor"]) in t for t in present(README, PR2, SUB)),
   f"{ty['max_shuffle_floor']}")
ck(f"{ty['n']} measurements, best beats START_LAYER 1 in all of them",
   ty["best_beats_one"] == ty["n"], f"{ty['best_beats_one']}/{ty['n']}")
ck("every optimum lies in 21..25", 21 <= ty["best_min"] and ty["best_max"] <= 25,
   f"{ty['best_min']}..{ty['best_max']}")
ck("the N is quoted consistently", all(f"{ty['n']} independent measurements" in t
   or f"{ty['n']} measurements" in t for t in (README,)), f"n={ty['n']}")

print("\nSTRESS CHECKS")
st = J("stress_window.json")["checks"]
for k, v in st.items():
    ck(f"stress {k}", v["pass"], "")
rows = J("stress_window.json")["rows"]
ink1 = next(r["pearson_ink_only"] for r in rows if r["start_layer"] == 1)
ink25 = next(r["pearson_ink_only"] for r in rows if r["start_layer"] == 25)
ck("ink-only 0.4993 -> 0.8592", f"{ink1:.4f}" == "0.4993" and f"{ink25:.4f}" == "0.8592",
   f"{ink1:.4f} -> {ink25:.4f}")

print("\nAUTOCAST, END TO END")
ae = J("amp_endtoend.json")
ck("Pearson between the two outputs is 0.9999995",
   f"{ae['pearson_between_the_two']:.7f}" == "0.9999995", f"{ae['pearson_between_the_two']:.7f}")
ck("decision flip rate is 0.014 percent",
   f"{ae['frac_pixels_crossing_0.5_decision']*100:.3f}" == "0.014",
   f"{ae['frac_pixels_crossing_0.5_decision']*100:.3f}%")

print("\nNULLS")
nl = {n["null"]: n for n in J("inkdiag_nulls25.json")["nulls"]}
real25 = J("inkdiag_nulls25.json")["depth_profile"][0]["frac_above_0.5"]
ck("real input at the centred window is 0.4138", f"{real25:.4f}" == "0.4138", f"{real25:.4f}")
for n in ("single_layer_repeat","depth_shuffle","voxel_shuffle","per_column_shuffle"):
    ck(f"{n} gives exactly zero confident ink", nl[n]["frac_above_0.5"] == 0.0,
       f"{nl[n]['frac_above_0.5']}")
old = {n["null"]: n for n in J("inkdiag_prof.json")["nulls"]}
# The battery must be reported WHOLE. A public comment once listed the five
# nulls that return zero at the centred window and omitted depth_roll_half,
# which returns 0.1429, i.e. the only one that does not support the reading.
# Selective omission is not catchable by checking the quoted numbers, so the
# test is structural: any artifact that discusses the battery must name EVERY
# null in it.
n1f = J("inkdiag_nulls1.json")
n1 = {n["null"]: n for n in n1f["nulls"]}
ck("the battery was run at BOTH windows on the same crop",
   set(n1) == set(nl) and n1f["crop"] == J("inkdiag_nulls25.json")["crop"],
   f"{len(n1)} nulls, crop {n1f['crop']}")
ck("START_LAYER 1 real input is 0.3323",
   f"{n1f['depth_profile'][0]['frac_above_0.5']:.4f}" == "0.3323",
   f"{n1f['depth_profile'][0]['frac_above_0.5']:.4f}")
for _n in ("single_layer_repeat", "per_column_shuffle"):
    ck(f"{_n} is exactly zero at START_LAYER 1, so the 7.32% is not a texture reader",
       n1[_n]["frac_above_0.5"] == 0.0, f"{n1[_n]['frac_above_0.5']}")
ck("depth_roll_half is the null that SURVIVES at the centred window",
   nl["depth_roll_half"]["frac_above_0.5"] > 0.1,
   f"{nl['depth_roll_half']['frac_above_0.5']:.4f}")
for _k, _t in ALL.items():
    if "depth_shuffle" not in _t:
        continue
    _miss = sorted(x for x in nl if x not in _t)
    ck(f"{_k}: reports the WHOLE battery, no null omitted", not _miss,
       f"omits {_miss}" if _miss else f"all {len(nl)} named")

ck("depth_shuffle at START_LAYER 1 is 0.0732 and anti-correlated",
   f"{old['depth_shuffle']['frac_above_0.5']:.4f}" == "0.0732"
   and old["depth_shuffle"]["pearson_vs_real"] < 0,
   f"{old['depth_shuffle']['frac_above_0.5']:.4f} / r={old['depth_shuffle']['pearson_vs_real']:.4f}")

print("\nBLANK REGION")
bl = J("inkdiag_blank.json")
b0 = next(r for r in bl["depth_profile"] if r["offset"] == 0)
ck("blank at START_LAYER 1 gives 0.0167", f"{b0['frac_above_0.5']:.4f}" == "0.0167",
   f"{b0['frac_above_0.5']:.4f}")
ck("blank gives exactly zero at every other window",
   all(r["frac_above_0.5"] == 0.0 for r in bl["depth_profile"] if r["offset"] != 0),
   str([r["frac_above_0.5"] for r in bl["depth_profile"]]))

print("\n4096 CROP")
c = J("compare_reference_big25.json")
ck("corrected 4096 aligned r is 0.9685", f"{c['aligned']['pearson']:.4f}" == "0.9685",
   f"{c['aligned']['pearson']:.4f}")
cold = J("compare_reference_big.json")
ck("the same pixels at START_LAYER 1 were 0.608", f"{cold['aligned']['pearson']:.3f}" == "0.608",
   f"{cold['aligned']['pearson']:.4f}")

print("\nFIXTURE COVERAGE")
fx = J("canon_fixtures.json")
ck("156 canonical 2.4um segments, none off 109 layers",
   len(fx) == 156 and all(h["layers"] == 109 for h in fx), f"{len(fx)} segments")

print("\nINTERNAL CONSISTENCY (a document must not contradict itself)")
_ty = J("window_tally.json")["summary"]
_stale_floors = {"0.0017", "0.0013"}
for _k, _t in ALL.items():
    _bad = sorted(f for f in _stale_floors if f"exceeds {f}" in _t or f"floor never exceeds {f}" in _t)
    ck(f"{_k}: quotes only one shuffle floor", not _bad,
       f"also quotes {_bad}" if _bad else f"{_ty['max_shuffle_floor']}")

# ---- the two 2026-09-22 comments: every figure they quote, re-derived ----
_C13 = _opt(R/"results"/"PR1813_COMMENT_halflayer.md")
_C12 = _opt(R/"results"/"PR1812_COMMENT_coverage.md")
_HL = (R/"results"/"halflayer.json")
if _HL.exists():
    print("\nHALF LAYER, 23 AGAINST 24")
    h = J("halflayer.json")
    ck("halflayer: wins sum to the crop count",
       h["wins_23"] + h["wins_24"] + h["ties"] == h["n"], f"n={h['n']}")
    ck("halflayer: every crop really carries both windows",
       all("r23" in x and "r24" in x for x in h["rows"]), f"{len(h['rows'])} rows")
    ck("halflayer: the sign test agrees with the recounted wins",
       h["wins_23"] == sum(1 for x in h["rows"] if x["diff"] > 0) and
       h["wins_24"] == sum(1 for x in h["rows"] if x["diff"] < 0),
       f"{h['wins_23']} to {h['wins_24']}")
    ck("halflayer: the shuffled reference floor stays near zero",
       h["worst_shuffle_floor"] < 0.01, f"{h['worst_shuffle_floor']}")
    ck("halflayer: the window fix dwarfs the half layer",
       abs(h["window_fix_worth"]) > 20 * abs(h["median_diff"]),
       f"{h['window_fix_worth']:+.4f} against {h['median_diff']:+.4f}")
    if _C13:
        for f in (f"23 wins {h['wins_23']}, 24 wins {h['wins_24']}",
                  f"p = {h['sign_test_p']:.4f}",
                  f"{h['median_r_at_1']:.4f}", f"{h['median_r_at_best_half']:.4f}",
                  f"{h['window_fix_worth']:+.4f}", f"{h['worst_shuffle_floor']:.4f}"):
            ck(f"PR1813 comment quotes {f!r} from source", f in _C13)
        ck("PR1813 comment states the grid defect rather than hiding it",
           has(_C13, "14 contained", "not 23"))
        ck("PR1813 comment does not claim the window finding as ours",
           "flummoxjr" in _C13 and "priority" in _C13.lower())
if (R/"results"/"mps_model_coverage.json").exists():
    print("\nMPS MODEL COVERAGE, ALL FOUR TYPES")
    mc = J("mps_model_coverage.json")
    _r = {x["model_type"]: x for x in mc["rows"]}
    ck("coverage: all four model types are present",
       set(_r) == {"timesformer", "resnet3d-50", "resnet3d-152",
                   "resnet3d-152-3d-decoder"}, f"{len(_r)} types")
    ck("coverage: every type RUNS on MPS in fp32 and autocast",
       all(str(x.get("mps_fp32_status","")).startswith(("OK","RUNS")) and
           str(x.get("mps_amp_status","")).startswith(("OK","RUNS")) for x in _r.values()))
    ck("coverage: a non finite output is UNDECIDED, never OK",
       all((x.get("mps_amp_status") == "OK") == bool(x.get("mps_amp_output_finite"))
           for x in _r.values()),
       "fail open on nan would break this")
    ck("coverage: fp32 agrees with the CPU to 1e-06 or better",
       all(x["mps_fp32_max_abs_diff"] <= 1e-6 for x in _r.values()))
    _ct = mc["trained_weight_control"]
    ck("coverage: the trained weight control ran and is finite",
       _ct.get("autocast_finite") is True and _ct.get("max_abs_diff") is not None,
       f"max abs diff {_ct.get('max_abs_diff')}")
    if _C12:
        for f in (f"torch {mc['torch']}", str(_ct["max_abs_diff"]),
                  "#1764", "#1770", "timesformer"):
            ck(f"PR1812 comment quotes {f!r} from source", f in _C12)
        ck("PR1812 comment reports the non finite cell rather than smoothing it",
           "not finite" in _C12 and "undecided" in _C12.lower())

if (R/"results"/"window_selector_tally.json").exists():
    print("\nARM W, REFERENCE FREE WINDOW SELECTION")
    ws = J("window_selector_tally.json")
    _m = ws["median_regret"]
    ck("ARM W: 15 crops on 4 scrolls", ws["n_crops"] == 15, f"{ws['n_crops']} crops")
    ck("ARM W: the shuffled reference floor stays near zero",
       ws["shuffle_floor"] < 0.01, f"{ws['shuffle_floor']}")
    ck("ARM W: the PRE-REGISTERED primary S4 is reported as FAILING",
       ws["primary_S4_works"] is False,
       f"S4 median regret {_m['S4_null_collapse']}, bar {ws['bar']}")
    ck("ARM W: S4 genuinely does not beat random, which is why it is worthless",
       _m["S4_null_collapse"] >= _m["C1_random"],
       f"{_m['S4_null_collapse']} vs random {_m['C1_random']}")
    ck("ARM W: the winning selector is named from the data, not asserted",
       ws["best_selector"] == min(
           (k for k in _m if k.startswith("S")), key=lambda k: _m[k]),
       ws["best_selector"])
    ck("ARM W: the winner beats random",
       _m[ws["best_selector"]] < _m["C1_random"],
       f"{_m[ws['best_selector']]} vs {_m['C1_random']}")
    if SUB is not None:
        for f in (f"{_m['S1_separation']:.4f}", f"{_m['S4_null_collapse']:.4f}",
                  f"{_m['C1_random']:.4f}", f"{_m['C2_readme_sl1']:.4f}"):
            ck(f"SUBMISSION quotes ARM W's {f!r} from source", f in SUB)
        ck("SUBMISSION reports the primary as a FAILURE rather than burying it",
           "PRIMARY FAILED" in SUB.upper() and "worthless" in SUB)
        ck("SUBMISSION states the scroll where the winner fails",
           "PHerc. 0814" in SUB and "0.1732" in SUB)

print("\nARM X, CROSS-SCROLL TRANSFER OF ink_9um")
_ax = R / "results" / "xs" / "armx_scores.json"
_f5 = R / "results" / "FORM_FIELD5_ARMX.txt"
if not _ax.exists() or "<!-- armx:start -->" not in README:
    print("  NOTE: ARM X not published here, not checked")
else:
    _S = json.loads(_ax.read_text()); _V, _G = _S["verdict"], _S["segments"]
    _sec = README[README.index("<!-- armx:start -->"):README.index("<!-- armx:end -->")]
    _docs = {"README": _sec}
    if _f5.exists(): _docs["FORM"] = _f5.read_text()
    _gate = "PASS" if _V["C1_pass"] else "FAIL"
    ck("ARM X: README states the C1 gate as the data decided it", f"bar 0.85: **{_gate}**" in _sec, _gate)
    ck("ARM X: README verdict is the pre-registered verdict, verbatim", f"Verdict: **{_V['verdict']}**" in _sec, _V["verdict"][:40])
    ck("ARM X: nothing claims TRANSFERS unless the verdict is TRANSFERS",
       _V["verdict"] == "TRANSFERS" or all("TRANSFERS" not in t for t in _docs.values()))
    for _n, _t in _docs.items():
        ck(f"ARM X {_n}: quotes the C1 primary AUC {_V['C1_auc_chosen_direction']:.4f}", f"{_V['C1_auc_chosen_direction']:.4f}" in _t)
        _dg = _V.get("C1_diagnostic_seed43_step060000_forward")
        if _dg is not None:
            ck(f"ARM X {_n}: quotes the pipeline diagnostic {_dg:.4f}", f"{_dg:.4f}" in _t)
    for _s, _r in _G.items():
        _pp = _r["rows"].get("hybrid_3d2d-seed42_step-075000", {})
        if "auc_chosen" in _pp:
            ck(f"ARM X README: {_s} primary AUC {_pp['auc_chosen']:.4f} quoted", f"**{_pp['auc_chosen']:.4f}**" in _sec)
    _PRIM = ["p0841_w00", "p0841_ag144", "p0841_ag174", "p0500p2a"]
    for _k in sorted({k for r in _G.values() for k in r["rows"]}):
        _held = [_G[s]["rows"][_k]["auc_chosen"] for s in _PRIM if s in _G and _k in _G[s]["rows"] and "auc_chosen" in _G[s]["rows"][_k]]
        if len(_held) == len(_PRIM):
            _m = f"{float(sorted(_held)[1] + sorted(_held)[2]) / 2:.4f}"
            ck(f"ARM X README: {_k.replace('hybrid_3d2d-', '')} held-out median {_m}", f"| {_m} (4 of 4) |" in _sec)
    for _f in sorted((R / "results" / "xs" / "depthcheck").glob("*.json")):
        _j = json.loads(_f.read_text())
        if len(_j["auc_by_shift"]) >= 2:
            for _k, _v in _j["auc_by_shift"].items():
                ck(f"ARM X README: depth check {_j['segment']} k={_k} {_v:.4f}", f"{_v:.4f}" in _sec)
    _iss = R / "results" / "ARMX_ISSUE.md"
    if _iss.exists(): _docs["ISSUE"] = _iss.read_text()
    for _n in ("FORM", "ISSUE"):
        if _n in _docs:
            _miss = sorted({x for x in re.findall(r"(?<![\d.])-?\d\.\d{3,4}(?![\d])", _docs[_n]) if x not in _sec})
            for _h, _t in re.findall(r"in (\d+) of (\d+) (?:\(segment, checkpoint\) )?cases", _docs[_n]):
                ck(f"ARM X {_n}: direction-rule count {_h} of {_t} matches the README", f"in {_h} of {_t} (segment, checkpoint) cases" in _sec)
            ck(f"ARM X {_n}: every 4-decimal number also appears in the README section generated from the JSON", not _miss, ", ".join(_miss[:5]))
            ck(f"ARM X {_n}: no em or en dashes, no AI attribution", not any(chr(c) in _docs[_n] for c in (0x2014, 0x2013)) and "claude" not in _docs[_n].lower())
    if "primary_median" in _V and "FORM" in _docs:
        ck("ARM X FORM: quotes the primary median", f"{_V['primary_median']:.4f}" in _docs["FORM"])

print("\nARM Y AND ARM Z, THE SAME PHerc0841 SEGMENTS ON THREE ORGANISER ARRAYS")
_ay, _az = R / "results" / "ys" / "army_scores.json", R / "results" / "zs" / "armz_scores.json"
if not (_ay.exists() and _az.exists()) or "<!-- armyz:start -->" not in README:
    print("  NOTE: ARM Y/Z not published here, not checked")
else:
    _Y, _Z = json.loads(_ay.read_text()), json.loads(_az.read_text()); _P = "hybrid_3d2d-seed42_step-075000"
    _ysec = README[README.index("<!-- armyz:start -->"):README.index("<!-- armyz:end -->")]
    _xsec = README[README.index("<!-- armx:start -->"):README.index("<!-- armx:end -->")]
    for _k in ("w00", "ag144", "ag174"):
        _w = f"w{_Y['segments'][_k]['prep']['production_window'][0]:02d}"; _ry = _Y["segments"][_k]["windows"][_w][_P]; _rz = _Z["segments"][_k]["rows"][_P]
        for _lab, _v in (("2.403 um forward", _ry["forward"]["auc"]), ("2.403 um reverse", _ry["reverse"]["auc"]), ("9.366 um forward", _rz["forward"]["auc"]),
                         ("9.366 um reverse", _rz["reverse"]["auc"]), ("C3", _Y["segments"][_k]["C3"]["auc"])):
            ck(f"ARM Y/Z README: {_k} {_lab} {_v:.4f}", f"{_v:.4f}" in _ysec)
        ck(f"ARM Y/Z README: C4 PASS on {_k}", _Y["segments"][_k]["prep"]["C4"]["pass_"] and f"{_k} PASS" in _ysec)
    for _name, _v in (("Y1", _Y["Y1"]["verdict"]), ("Y2", _Y["Y2"]["verdict"]), ("Z1", _Z["Z1"]["verdict"])) + ((("Y3", _Y["Y3"]["verdict"]),) if "Y3" in _Y else ()):
        ck(f"ARM Y/Z README: {_name} verdict verbatim", f"**{_v}**" in _ysec, _v[:40])
    for _v in (_Y["Y2"]["published_median"], _Y["Y2"]["render_median"], _Z["Z1"]["eligible_median"], _Y["Y2"]["c1_in_distribution"]):
        ck(f"ARM Y/Z README: median {_v:.4f}", f"{_v:.4f}" in _ysec)
    if "ALLCKPT" in _Y:
        _A = _Y["ALLCKPT"]
        for _lab, _v in (("C1 vs 2.403 Spearman", _A["spearman_c1_vs_published"]), ("C1 correct-direction vs 2.403 Spearman", _A["spearman_c1oracle_vs_published"]),
                         ("render vs 2.403 Spearman", _A["spearman_render_vs_published"]), ("best single 2.403", _A["best_single_median"])):
            ck(f"ARM Y/Z README: {_lab} {_v:.4f}", f"{_v:.4f}" in _ysec)
        for _c, _v in _A["published_median"].items():
            ck(f"ARM Y/Z README: {_c.replace('hybrid_3d2d-', '')} 2.403 median {_v:.4f}", f"| {_v:.4f} |" in _ysec)
        ck("ARM Y/Z README: the two wrong-direction C1 rows are named", all(c.replace("hybrid_3d2d-", "") in _ysec for c in _A["c1_rule_wrong"]))
        _dk = _A["direction_rule_kept_better"]; ck(f"ARM Y/Z README: direction rule {_dk[0]} of {_dk[1]}", f"**{_dk[0]} of {_dk[1]}**" in _ysec)
    if "ALLCKPT" in _Z:
        _B = _Z["ALLCKPT"]
        for _lab, _v in (("C1 vs eligible", _B["Z2_spearman_c1_vs_eligible"]), ("C1 correct vs eligible", _B["Z2_spearman_c1oracle_vs_eligible"]),
                         ("render vs eligible", _B["Z3_spearman_render_vs_eligible"]), ("best single eligible", _B["Z4_best_single_median"]), ("finals averaged eligible", _B["Z4_ensemble_finals"])):
            ck(f"ARM Y/Z README: {_lab} {_v:.4f}", f"{_v:.4f}" in _ysec)
    ck("ARM X README section carries the render note generated from ARM Y/Z",
       "as a property of the organisers' 4.681 um renders, not of the scroll" in _xsec and f"{_Z['Z1']['eligible_median']:.4f}" in _xsec)
    _docs2 = {n: (R / "results" / f).read_text() for n, f in (("REPLY", "ys/REPLY_1867_armyz.md"), ("FORM", "FORM_FIELD5_ARMYZ.txt"), ("C1582", "ys/COMMENT_1582_family.md")) if (R / "results" / f).exists()}
    for _n, _t in _docs2.items():
        _miss = sorted({x for x in re.findall(r"(?<![\d.])-?\d\.\d{3,4}(?![\d])", _t) if x not in _ysec and x not in _xsec})
        ck(f"ARM Y/Z {_n}: every 4-decimal number appears in the generated README sections", not _miss, ", ".join(_miss[:5]))
        ck(f"ARM Y/Z {_n}: no em or en dashes, no AI attribution", not any(chr(c) in _t for c in (0x2014, 0x2013)) and "claude" not in _t.lower())
        ck(f"ARM Y/Z {_n}: does not repeat the retracted 'weaker on PHerc0841' reading", "weaker signal on" not in _t and "weaker evidence than" not in _t)

print("\nARM F, THE ELIGIBLE-SCAN GAP AND THE DEPTH FILTER")
_af = R / "results" / "fs" / "armf_scores.json"
if not _af.exists() or "<!-- armf:start -->" not in README:
    print("  NOTE: ARM F not published here, not checked")
else:
    _F = json.loads(_af.read_text()); _P = "hybrid_3d2d-seed42_step-075000"
    _fsec = README[README.index("<!-- armf:start -->"):README.index("<!-- armf:end -->")]
    for _k, _rows in _F["segments"].items():
        for _c, _r in _rows.items():
            if _P in _r and "auc_chosen" in _r[_P]:
                ck(f"ARM F README: {_k} {_c} {_r[_P]['auc_chosen']:.4f}", f"{_r[_P]['auc_chosen']:.4f}" in _fsec)
    ck("ARM F README: F1 verdict verbatim, the primary FAILED", f"Verdict: {_F['F1']['verdict']}." in _fsec and _F["F1"]["verdict"] == "not repairable this way",
       _F["F1"]["verdict"])
    ck(f"ARM F README: F1 median {_F['F1']['median_gain']:+.4f}", f"{_F['F1']['median_gain']:+.4f}" in _fsec)
    ck("ARM F README: F2 verdict verbatim", _F["F2"]["verdict"] in _fsec, _F["F2"]["verdict"])
    ck("ARM F: C0 and C-DOSE pass and are stated", _F["C0"]["pass_"] and _F["C_DOSE"]["pass_"] and "C0: N0 and P0 reproduce" in _fsec and "C-DOSE (blurring must not help): PASS" in _fsec)
    _n14 = R / "results" / "fs" / "armf_ndh14.json"
    if _n14.exists():
        _S = json.loads(_n14.read_text())["summary"]
        ck(f"ARM F README: all-checkpoint count {_S['pairs_improved']} of {_S['pairs']}", f"**improved in {_S['pairs_improved']} of {_S['pairs']} checkpoint" in _fsec)
    _f5p = R / "results" / "fs" / "f5_scores.json"
    if _f5p.exists():
        _G = json.loads(_f5p.read_text())
        for _n, _v in _G["auc"].items():
            for _c, _x in _v.items():
                if _x is not None: ck(f"ARM F README: PHerc0139 {_n} {_c} {_x:.4f}", f"{_x:.4f}" in _fsec)
        for _e in ("F5", "F5b", "F5c"):
            if _e in _G: ck(f"ARM F README: {_e} {_G[_e]['median_gain']:+.4f} {_G[_e]['verdict']}", f"{_G[_e]['median_gain']:+.4f}, {_G[_e]['verdict']}." in _fsec)
    _tool, _par = R / "public" / "bench" / "depth_sharpen_9um.py", R / "public" / "bench" / "depth_sharpen_9um.json"
    _d5 = R / "data" / "f5" / "data"
    if _tool.exists() and _par.exists() and _d5.exists():
        import importlib.util, zarr as _zarr, numpy as _np
        _sp = importlib.util.spec_from_file_location("_ds9", _tool); _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
        # in row strips: sharpen() works per pixel column along depth, so strips equal the whole-volume result bit for bit
        for _n in sorted(p.name for p in (_d5 / "fs").iterdir() if (p / "NDH" / "ct.zarr").exists()):
            _src = _np.asarray(_zarr.open_array(str(_d5 / "zs" / _n / "w04" / "ct.zarr"), mode="r")[:])
            for _c, _um in (("NDH", True), ("ND", False)):
                _ref = _np.asarray(_zarr.open_array(str(_d5 / "fs" / _n / _c / "ct.zarr"), mode="r")[:])
                ck(f"ARM F tool: bench/depth_sharpen_9um.py reproduces the tested {_c} input on PHerc0139 {_n}", bool(_src.shape == _ref.shape and all(_np.array_equal(_m.sharpen(_src[:, _y:_y + 256], 9.362, _um), _ref[:, _y:_y + 256]) for _y in range(0, _src.shape[1], 256))))
    else:
        print("  NOTE: tool equivalence needs the local data tree, not checked here")
    _docs3 = {n: (R / "results" / f).read_text() for n, f in (("ISSUE", "fs/ISSUE_armf.md"), ("C1582", "fs/COMMENT_1582_armf.md"), ("FORM", "FORM_FIELD5_ARMF.txt"),
                                                            ("DISCORD", "fs/DISCORD_armf.md"), ("C1867", "fs/COMMENT_1867_armf.md"), ("R1898", "fs/REPLY_1898_khj1222.md")) if (R / "results" / f).exists()}
    for _n, _t in _docs3.items():
        _miss = sorted({x for x in re.findall(r"(?<![\d.])-?\d\.\d{3,4}(?![\d])", _t) if x not in README})
        ck(f"ARM F {_n}: every 3 or 4 decimal number appears in the generated README", not _miss, ", ".join(_miss[:5]))
        ck(f"ARM F {_n}: no em or en dashes, no AI attribution", not any(chr(c) in _t for c in (0x2014, 0x2013)) and "claude" not in _t.lower())
        ck(f"ARM F {_n}: states that the pre-registered primary failed", "primary" in _t.lower() and ("failed" in _t.lower() or "did not" in _t.lower()))

print("\nHECATE PRECISION ON MPS")
# Added 2026-09-23: the README's bf16 table had NO check until now.
for _n, _mode in (("hecate_bf16.json", "bf16"), ("hecate_fp16.json", "fp16")):
    if not (R / "results" / _n).exists():
        print(f"  NOTE: {_n} not published here, not checked"); continue
    _h = J(_n); _a, _b, _c = _h["fp32_mps"], _h[f"{_mode}_mps"], _h[f"{_mode}_vs_fp32"]
    for _v in (_a["min"], _a["median"], _a["max"], _b["min"], _b["median"], _b["max"]):
        ck(f"README quotes {_mode} table value {_v:.3f} from {_n}", f"{_v:.3f}" in README)
    _x = f"{_c['pixels_crossing_0.5']:,} of {_c['total_pixels']:,}"
    ck(f"README quotes the {_mode} decision flips {_x!r}", _x in README, _x)
    if _mode == "bf16":
        _r = round(float(_a["median"]) / float(_b["median"]), 2)
        ck("README calls bf16 1.54x SLOWER and the data agree", _r == 0.65 and "**bf16 is 1.54x slower**" in README, f"ratio {_r}")
    else:
        _r = f"{_h['speedup_median']:.2f}x"
        ck(f"README quotes the fp16 speedup {_r}", f"**fp16 is {_r} faster on the median.**" in README, _r)

print("\nSEED VERSUS STEP (reply to PedroR4321 on #1867)")
_sd, _rp = R / "results" / "seed_decomp.json", R / "results" / "ys" / "REPLY_1867_pedro.md"
if not _sd.exists() or not _rp.exists():
    print("  NOTE: seed decomposition not published here, not checked")
else:
    _S = json.loads(_sd.read_text()); _t = _rp.read_text(); _ok = {"2.403", "9.366", "4.681"}
    for _a in _S["arrays"].values():
        _ok |= {f"{100 * _a[_k]:.1f}" for _k in ("share_seed", "share_step", "share_resid")}
        _ok |= {f"{_a['between_seed_gap_mean']:+.4f}", f"{_a['between_seed_gap_mean']:+.3f}", f"{_a['step_effect_corr_between_seeds']:+.2f}"}
    for _r in _S["c1_ranking"].values():
        _ok |= {f"{_v:+.2f}" for _v in _r.values()}
    _bad = [x for x in re.findall(r"[+-]?\d+\.\d+", _t) if x not in _ok]
    ck("REPLY 1867 seed vs step: every number is generated from results/seed_decomp.json", not _bad, ", ".join(_bad[:5]))
    ck("REPLY 1867 seed vs step: no em or en dashes, no AI attribution",
       not any(chr(c) in _t for c in (0x2014, 0x2013)) and "claude" not in _t.lower())

print("\nFIFTH SEPTEMBER RESPONSE (plain summary of #1898 and evidence of use)")
_f5, _ev = R / "results" / "FORM_FIELD5_USE.txt", R / "results" / "evidence"
if not _f5.exists() or not all((R / "results" / "fs" / f).exists() for f in ("armf_scores.json", "f6_scores.json")):
    print("  NOTE: fifth response not published here, not checked")
else:
    import statistics as _st
    _t = _f5.read_text(); _FF = json.loads((R / "results" / "fs" / "armf_scores.json").read_text()); _P = "hybrid_3d2d-seed42_step-075000"
    _ndh = _st.median(r["NDH"][_P]["auc_chosen"] - r["N0"][_P]["auc_chosen"] for r in _FF["segments"].values() if "NDH" in r and "N0" in r and _P in r["NDH"])
    _ours = {f"{json.loads((R / 'results' / 'zs' / 'armz_scores.json').read_text())['Z1']['eligible_median']:.4f}",f"{json.loads((R / 'results' / 'ys' / 'army_scores.json').read_text())['Y2']['published_median']:.4f}",
             f"{_ndh:+.4f}", f"{json.loads((R / 'results' / 'fs' / 'f6_scores.json').read_text())['F6']['mean_gain']:+.4f}"}
    _ext = {"+0.029": "bullo27_1898_5845328465.md", "+0.062": "bullo27_1898_5845328465.md", "+0.018": "bullo27_1898_5845328465.md", "+0.010": "lil_1907_5859563049.md"}
    _nums = set(re.findall(r"[+-]?\d\.\d{3,4}", _t)) - {"2.403", "9.366"}
    ck("FORM 5 (use): every 3 or 4 decimal number is ours from JSON or quoted from a saved source", _nums <= (_ours | set(_ext)), ", ".join(sorted(_nums - _ours - set(_ext))))
    ck("FORM 5 (use): all four of our figures are present", _ours <= _nums, ", ".join(sorted(_ours - _nums)))
    ck("FORM 5 (use): each quoted external figure is in its saved source",
       all((_ev / f).exists() and x in (_ev / f).read_text() for x, f in _ext.items() if x in _nums))
    ck("FORM 5 (use): the pre-registration credit is in liliandevarieux's saved file",
       "Practice borrowed from" in (_ev / "lil_prereg_head.md").read_text() and "pre-registering" in _t)
    ck("FORM 5 (use): the negative (no readable letters) is stated", "does not make letters readable" in _t and "does not by itself produce legible letters" in _t)
    ck("FORM 5 (use): no em or en dashes, no AI attribution", not any(chr(c) in _t for c in (0x2014, 0x2013)) and "claude" not in _t.lower())

print("\nFLINJ (can ink_9um draw a letter on the eligible scan type)")
_fd = R / "results" / "flinj" / "report_data.json"
_fl = next((q for q in (R / "ops" / "flinj", R / "bench" / "flinj") if (q / "make_flinj_texts.py").exists()), None)
if not _fd.exists() or _fl is None or "<!-- FLINJ:BEGIN -->" not in README:
    print("  NOTE: FLINJ not published here, not checked")
else:
    import hashlib as _hl, os as _os, subprocess as _sp, statistics as _st, tempfile as _tf
    _sec = README.split("<!-- FLINJ:BEGIN -->")[1].split("<!-- FLINJ:END -->")[0].strip("\n")
    with _tf.TemporaryDirectory() as _td:
        _r = _sp.run([sys.executable, str(_fl / "make_flinj_texts.py")], capture_output=True, text=True,
                     env={**_os.environ, "FLINJ_TEXT_OUT": _td, "FLINJ_JSON": str(_fd)})
        _nomodels = _r.returncode != 0 and "all_ckpts" in _r.stderr   # the pipeline modules need models/ (14 checkpoints)
        if _nomodels:
            print("  NOTE: generator re-run skipped: it needs the 14 ink_9um checkpoints under models/; the checks below still run")
        else:
            ck("FLINJ: the generator runs on this repo's report_data.json", _r.returncode == 0, _r.stderr.strip()[-200:])
        _g = {n: (pathlib.Path(_td) / n).read_text() if (pathlib.Path(_td) / n).exists() else None
              for n in ("PUBLIC_SECTION.md", "FORM_FIELD5_FLINJ.txt", "DISCORD_FLINJ.md")}
    if not _nomodels:
        ck("FLINJ: the README section is exactly what report_data.json generates now",
           _g["PUBLIC_SECTION.md"] is not None and _sec == _g["PUBLIC_SECTION.md"].strip("\n"))
        for _n in ("FORM_FIELD5_FLINJ.txt", "DISCORD_FLINJ.md"):
            if (R / "results" / "flinj" / _n).exists():
                ck(f"FLINJ: {_n} is exactly what the JSON generates now", (R / "results" / "flinj" / _n).read_text() == _g[_n])
    ck("FLINJ: the README section is the committed PUBLIC_SECTION.md",
       (R / "results" / "flinj" / "PUBLIC_SECTION.md").exists() and _sec == (R / "results" / "flinj" / "PUBLIC_SECTION.md").read_text().strip("\n"))
    _H = json.loads(_fd.read_text()); _inc = [h for h in _H if not _H[h].get("excluded")]
    _med = _st.median(_H[h]["pairs"]["primary"]["C_b3"] for h in _inc)   # recomputed here, independent of the generator
    ck("FLINJ: the headline median C, recomputed here, is the README's", f"median C {_med:.4f}" in _sec, f"{_med:.4f}")
    ck("FLINJ: every host's C in the table is its primary C_b3", all(f"| {_H[h]['pairs']['primary']['C_b3']:.4f} |" in _sec for h in _inc))
    ck("FLINJ: STROKES follows from the registered bar (median >= 0.5) and all primaries pass the floor",
       _med >= 0.5 and all(_H[h]["pairs"]["primary"]["b3_defined"] for h in _inc) and "verdict is STROKES" in _sec)
    ck("FLINJ: the registration and all four amendments hash to what prereg.sha256 records",
       all(_hl.sha256((R / "results" / "flinj" / f).read_bytes()).hexdigest() in (R / "results" / "flinj" / "prereg.sha256").read_text()
           for f in ("PREREGISTRATION_FLINJ.md", "AMENDMENT_1.md", "AMENDMENT_2.md", "AMENDMENT_3.md", "AMENDMENT_4.md")))
    ck("FLINJ: the limits are stated", has(_sec, "Limits:", "113 keV scans only", "loses part of what the model uses"))

print("\nSPEC (does teaching ink_9um to ignore depth-shuffled input help)")
_sd, _sq = R / "results" / "spec" / "spec_scores.json", R / "results" / "spec" / "flinj" / "rescore.json"
_sg = next((q for q in (R / "ops" / "spec", R / "bench" / "spec") if (q / "make_spec_texts.py").exists()), None)
if not _sd.exists() or not _sq.exists() or _sg is None or "<!-- SPEC:BEGIN -->" not in README:
    print("  NOTE: SPEC not published here, not checked")
else:
    import hashlib as _hl, os as _os, subprocess as _sp, statistics as _st, tempfile as _tf
    _sec = README.split("<!-- SPEC:BEGIN -->")[1].split("<!-- SPEC:END -->")[0].strip("\n")
    with _tf.TemporaryDirectory() as _td:
        _r = _sp.run([sys.executable, str(_sg / "make_spec_texts.py")], capture_output=True, text=True,
                     env={**_os.environ, "SPEC_TEXT_OUT": _td, "SPEC_JSON": str(_sd), "SPEC_RESCORE": str(_sq)})
        ck("SPEC: the generator runs on this repo's JSON", _r.returncode == 0, _r.stderr.strip()[-200:])
        _g = {n: (pathlib.Path(_td) / n).read_text() if (pathlib.Path(_td) / n).exists() else None
              for n in ("PUBLIC_SECTION.md", "DISCORD_SPEC.md")}
    ck("SPEC: the README section is exactly what the JSON generates now",
       _g["PUBLIC_SECTION.md"] is not None and _sec == _g["PUBLIC_SECTION.md"].strip("\n"))
    ck("SPEC: the README section is the committed PUBLIC_SECTION.md",
       (R / "results" / "spec" / "PUBLIC_SECTION.md").exists() and _sec == (R / "results" / "spec" / "PUBLIC_SECTION.md").read_text().strip("\n"))
    if (R / "results" / "spec" / "DISCORD_SPEC.md").exists():
        ck("SPEC: DISCORD_SPEC.md is exactly what the JSON generates now", (R / "results" / "spec" / "DISCORD_SPEC.md").read_text() == _g["DISCORD_SPEC.md"])
    _S, _Q = json.loads(_sd.read_text()), json.loads(_sq.read_text()); _cr = ["w00", "ag144", "ag174", "p0500p2"]
    _d = [_S["rows"]["T"][c]["real"]["auc"] - _S["rows"]["K"][c]["real"]["auc"] for c in _cr]   # recomputed from the rows
    _dm, _nw = _st.median(_d), sum(x > 0 for x in _d)
    ck("SPEC: the headline median change, recomputed from the rows, is the README's", f"is {_dm:+.4f}" in _sec, f"{_dm:+.4f}")
    ck("SPEC: the count of crops where the fine-tune is higher is the README's", f"higher on {_nw} of 4 crops" in _sec, str(_nw))
    ck("SPEC: NULL follows from the registered bars (WIN >= +0.010 with 3 of 4; HARM <= -0.010)",
       not (_dm >= 0.010 and _nw >= 3) and not (_dm <= -0.010) and _S["verdict"] == "NULL" and "verdict NULL" in _sec)
    ck("SPEC: positive control and manipulation check passed on all 4 crops",
       all(_S["positive_control"].values()) and all(_S["manipulation"].values()))
    ck("SPEC: the planted-letter re-score reproduced FLINJ's stored predictions exactly (control)",
       all(v == 0 for v in _Q["control"].values()))
    ck("SPEC: the registration hashes to what prereg.sha256 records",
       _hl.sha256((R / "results" / "spec" / "PREREGISTRATION_SPEC.md").read_bytes()).hexdigest()
       in (R / "results" / "spec" / "prereg.sha256").read_text())
    ck("SPEC: the limits are stated", has(_sec, "**Limits.**", "Transductive", "113 keV scans only"))

print("\nNL343 (the PHerc. 343 First Letters render without VC3D, maps on Apple Silicon)")
_nlj = R / "results" / "nieuwlaar" / "nl343_summary.json"
if not _nlj.exists() or "<!-- NL343:BEGIN -->" not in README:
    print("  NOTE: NL343 not published here, not checked")
else:
    import importlib.util as _ilu, hashlib as _hl3
    _gen = next(p for p in (R / "ops" / "make_nl343_section.py", R / "bench" / "make_nl343_section.py") if p.exists())
    _sp = _ilu.spec_from_file_location("make_nl343_section", _gen); _nm = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_nm)
    _ns = json.loads(_nlj.read_text())
    _nsec = README[README.index(_nm.BEGIN):README.index(_nm.END) + len(_nm.END)]
    ck("NL343: the README section is exactly what the summary JSON generates now", _nsec == _nm.section_text(_ns))
    if (R / "results" / "nieuwlaar" / "g2.json").exists() and (R / "results" / "nieuwlaar" / "arm_S2" / "result.json").exists():
        ck("NL343: the summary is exactly what the raw G2 and S2 results give now", _nm.build_summary(R) == _ns)
    else:
        print("  NOTE: raw G2 and S2 results are not in this repo; the summary is checked against its own bars only")
    _gr = _ns.get("grids")
    ck("NL343: --num-slices 31 --flip-normals reproduces the published VC3D render (>= 99% identical), unflipped does not, default grid unchanged",
       _gr is not None and _gr["passed"] and _gr["default_identical"] and _gr["identical_median"] >= 0.99
       and _gr["noflip_corr_median"] < 0.9)
    _g, _k = _ns["g2"], _ns["s2"]
    ck("NL343: G2 passes its registered bars (f34 >= 0.99, coverage >= 0.5) and discriminates (f33, f35 < 0.99)",
       _g["f34"] >= 0.99 and _g["grey_share"] >= 0.5 and _g["f33"] < 0.99 and _g["f35"] < 0.99
       and _g["pass_f34"] and _g["discriminates"])
    ck("NL343: the count of pixels off by more than 1 follows from f34",
       _g["off_by_more_than_1"] == round((1 - _g["f34"]) * _g["grey_px"]))
    ck("NL343: REPRODUCED follows from the registered bars (three r >= 0.98, >= 16 of 18 boxes) and the depth control",
       min(_k["r_alpha"], _k["r_alpha_letters"], _k["r_lum"]) >= 0.98 and _k["boxes_agree"] >= 16
       and _k["r_alpha_d0"] < _k["r_alpha"] and _k["r_alpha_d2"] < _k["r_alpha"] and _k["verdict"] == "REPRODUCED")
    _rend = next(p for p in (R / "public" / "firstletters" / "render_tifxyz.py", R / "firstletters" / "render_tifxyz.py")
                 if p.exists())
    ck("NL343: the published renderer is byte-identical to the one that made the measured render",
       _hl3.sha256(_rend.read_bytes()).hexdigest() == _ns["renderer_sha256"])
    ck("NL343: the registration hashes to what the summary records",
       _hl3.sha256((R / "results" / "nieuwlaar" / "PREREGISTRATION_MAC.md").read_bytes()).hexdigest() == _ns["prereg_sha256"])
    ck("NL343: the limits are stated", has(_nsec, "**Limits.**", "his published statistics", "has not finished yet"))

print("\nPATCHES (axis-free sheet patches on First Letters scrolls)")
_pj = R / "results" / "patches" / "patches_summary.json"
if not _pj.exists() or "<!-- PATCHES:BEGIN -->" not in README:
    print("  NOTE: PATCHES not published here, not checked")
else:
    import importlib.util as _ilu, hashlib as _hlp, os as _osp
    _pg = next(p for p in (R / "ops" / "make_patches_section.py", R / "bench" / "make_patches_section.py") if p.exists())
    _spp = _ilu.spec_from_file_location("make_patches_section", _pg); _pm = _ilu.module_from_spec(_spp); _spp.loader.exec_module(_pm)
    _ps = json.loads(_pj.read_text())
    _psec = README[README.index(_pm.BEGIN):README.index(_pm.END) + len(_pm.END)]
    _NUM = re.compile(r"(?<![A-Za-z_\d.])\d+(?:\.\d+)?(?![A-Za-z_\d])")
    _STRUCT = {0: "zarr level 0 in the usage block", 1: "d = +1, the registered depth", 33: "centre layer of the 66-layer render",
               66: "layers of the render", float(_pm.OFF_CENTRE): "the generator's off-centre threshold"}
    _SCJ = _opt(R / "results" / "patches" / "survey_sheetcheck.json")
    _SC = json.loads(_SCJ) if _SCJ else None
    _PA = R / "research" / "prior_art"
    _PRIOR = sorted(_PA.glob("*survey_README*.md")) if _PA.exists() else []
    _PBODY = sorted(_PA.glob("villa_*_body_*.md")) if _PA.exists() else []

    _UD, _disk = R / "research" / "uncontested", {}
    for _d in (sorted(_UD.glob("patch*")) if _UD.exists() else []):
        if _d.is_dir() and (_d / "validation.json").exists() and not (_d / "WRITEUP_EXCLUDE").exists():
            _dv = json.loads((_d / "validation.json").read_text())
            _db = json.loads((_d / "build.json").read_text()) if (_d / "build.json").exists() else {}
            _dr = json.loads((_d / "run.json").read_text()) if (_d / "run.json").exists() else None
            _disk.setdefault(_pm._scroll_name(R, _dv, _db, _dr), []).append((_d, _dv, _db, _dr))

    def _raw(x):
        """validation.json and build.json of one scroll's patch folder, (None, None) in the published repo"""
        d = R / x["folder"]
        if not (d / "validation.json").exists():
            return None, None
        return json.loads((d / "validation.json").read_text()), (json.loads((d / "build.json").read_text())
                                                                  if (d / "build.json").exists() else {})

    def _patches_checks(sec):
        """every check of this block on one section text, as (name, ok, detail); the control runs it on a planted copy."""
        flat = " ".join(sec.split())
        out = [("PATCHES: the README section is exactly what the summary JSON generates now", sec == _pm.section_text(_ps), "")]
        allowed = {float(x) for x in _NUM.findall(json.dumps(_ps))} | set(_STRUCT)
        extra = sorted({x for x in _NUM.findall(sec) if float(x) not in allowed})
        out.append(("PATCHES: every number in the section is a value of the summary JSON or a named constant", not extra,
                    ", ".join(extra[:8])))
        parts = {c.split("\n", 1)[0].strip(): c for c in sec.split("\n### ")[1:]}
        names = [x["scroll"] for x in _ps["scrolls"]]
        out.append(("PATCHES: one table per scroll of the summary, in its order", list(parts) == names, f"{list(parts)} vs {names}"))
        tot_n = tot_pass = 0
        tot_area, raw_all, recs = 0.0, True, []
        pub = next(q for q in (R / "public" / "uncontested" / "sheet_patches.py", R / "uncontested" / "sheet_patches.py") if q.exists())
        psh = _hlp.sha256(pub.read_bytes()).hexdigest()
        for x in _ps["scrolls"]:
            sn = x["scroll"]
            pflat = " ".join(parts.get(sn, "").split())
            V, B = _raw(x)
            if V is None:
                raw_all = False
                continue
            P, suf = V["patches"], sn[len("PHerc"):]
            EXP = [z for z in _disk.get(sn, []) if str(z[0].relative_to(R)) != x["folder"]]
            out.append((f"PATCHES {sn}: its patch folders on disk (a validation.json, no WRITEUP_EXCLUDE) are the summary's, all "
                        "but the first an expansion", sorted(str(z[0].relative_to(R)) for z in _disk.get(sn, [])) ==
                        sorted([x["folder"]] + [e["folder"] for e in x.get("expansions") or []])
                        and all(z[2].get("existing_keys") for z in EXP), ", ".join(z[0].name for z in _disk.get(sn, []))))
            PA, clash = dict(P), []
            for z in EXP:
                clash += sorted(set(PA) & set(z[1]["patches"]))
                PA.update(z[1]["patches"])
            recs += ([B["params"]] if "params" in B else [r["params"] for r in (B.get("per_key_records") or {}).values()]) + \
                [r["params"] for z in EXP for r in (z[2].get("per_key_listing_records") or {}).values()]
            seed0 = (B.get("params") or {}).get("Z0")
            Q = ((_SC or {}).get("scrolls") or {}).get(sn)
            shp = {q["patch"]: q for q in (Q or {}).get("patches", [])}
            rows = {l.split("|")[1].strip(): [c.strip() for c in l.split("|")[2:-1]]
                    for l in parts.get(sn, "").splitlines() if re.match(r"\| p[a-z]\d+_", l)}
            bad = []
            for n, p in PA.items():
                vt, q = p["vertical"], shp.get(n)
                want = [p.get("seed_key", seed0), f"{p['keys'][0]} to {p['keys'][1]}", p["angle_deg"], p["area_cm2"],
                        p["m7"]["on_sheet_vertices"], p["keys_check"]["second_diff_max"], vt["on_track_of_covered_all"],
                        max(v or 0 for v in vt["control_half_spacing_shift_on_track"].values()),
                        p["orientation"]["outward_fraction_chosen"], f"{vt['switch_events']} / {vt['lines']}",
                        (p.get("render") or {}).get("peak_layer", "not rendered"),
                        "not scored" if q is None else
                        f"{q['good_fraction']:g} ({q['tiles']} tiles), {'pass' if q['passes'] else 'fail'}"]
                got = rows.get(n, [])
                if len(got) != len(want):
                    bad.append(f"{n}: {len(got)} cells")
                    continue
                for g, w in zip(got, want):
                    try:
                        same = g == w if isinstance(w, str) else float(g) == float(w)
                    except ValueError:
                        same = False
                    if not same:
                        bad.append(f"{n}: {g} != {w}")
            out.append((f"PATCHES {sn}: one table row per patch of every folder's validation.json, every cell equal to it and to "
                        "survey_sheetcheck.json (re-read here)", sorted(rows) == sorted(PA) and not bad and not clash,
                        "; ".join((bad + clash)[:4])))
            area = round(sum(p["area_cm2"] for p in PA.values()), 3)
            npass = sum(bool(p["pass"]) for p in PA.values())
            tot_n, tot_pass, tot_area = tot_n + len(PA), tot_pass + npass, tot_area + sum(p["area_cm2"] for p in PA.values())
            out.append((f"PATCHES {sn}: the patch count, the pass count and the total area, summed here over its {1 + len(EXP)} "
                        f"patch folder{'s' if EXP else ''} on disk, are the README's",
                        f"{len(PA)} patches, {npass} passing every gate, {area:.3f} cm2 in all" in pflat
                        and npass == V["n_pass"] + sum(z[1]["n_pass"] for z in EXP), f"{len(PA)}, {npass}, {area:.3f}"))
            e = V["extension"]
            if "passed_not_written_cap" in e:
                fa = round(sum(P[n]["area_cm2"] for n in B["selected"]), 3)
                sel = has(pflat, f"The first {len(B['selected'])} ({fa:.3f} cm2) come from seeds at z {B['params']['Z0']}",
                          f"The other {len(e['accepted'])} ({e['new_area_cm2']:.3f} cm2)",
                          f"{e['validated']} candidates validated, {e['passed']} passed, {len(e['refused_duplicate_of_new'])} refused",
                          f"{len(e['passed_not_written_cap'])} passing candidates were not written")
            else:
                pk, nd = list(e["per_key"].values()), len(e["refused_duplicate"])
                sel = has(pflat, f"wrote {sum(k['n_selected'] for k in pk)} patches and {sum(k['n_pass'] for k in pk)} passed",
                          f"{nd} {'was' if nd == 1 else 'were'} refused as duplicates", f"so {len(e['accepted'])} were kept") \
                    and sorted(e["accepted"]) == sorted(P)
            out.append((f"PATCHES {sn}: how the patches were selected is quoted from build.json and validation.json", sel, ""))
            for d, Ve, Be, Re in EXP:
                e2, Pe = Ve["extension"], Ve["patches"]
                na, nc, dc = len(e2["accepted"]), len(e2["refused_cap"]), e2["duplicate_control"]
                frag = [f"The last {len(Pe)} ({round(sum(q['area_cm2'] for q in Pe.values()), 3):.3f} cm2",
                        f"{len(Be['candidates_built'])} in all; {len(e2['failed_bars'])} failed the tracer's own bars, "
                        f"{len(e2['failed_gates'])} failed validation",
                        f"{len(e2['refused_duplicate'])} were refused as duplicates "
                        f"({sum(q['duplicate_of'] in P for q in e2['refused_duplicate'].values())} of them of the {len(P)} earlier patches",
                        f"the {na} largest of the other {na + nc} were kept" if nc else f"and the other {na} were kept"] + \
                    [f"{e2['per_region'][m]} nearer {m}" for m in sorted(e2["per_region"])] + \
                    ([f"the {len(dc)} candidates that rebuild an earlier patch came out identical to it"]
                     if dc and all(q["identical"] and q["rebuilt"] in e2["refused_duplicate"] for q in dc) else [])
                parts_n = sum(len(e2[k]) for k in ("accepted", "refused_cap", "refused_duplicate", "failed_gates", "failed_bars",
                                                     "failed_build", "failed_validate"))
                out.append((f"PATCHES {sn}: how the expansion {d.name} was selected is quoted from its build.json and validation.json",
                            has(pflat, *frag) and sorted(e2["accepted"]) == sorted(Pe) and parts_n == len(Be["candidates_built"]),
                            " | ".join(frag)[:160]))
            cr = V["criteria"]
            out.append((f"PATCHES {sn}: the gates quoted are the criteria in its validation.json",
                        has(flat, f"at least {cr['on_sheet_vertices_min']:g} of its vertices",
                            f"never exceeds {cr['key_second_difference_max_max']:g} voxels",
                            f"at least {cr['vertical_on_track_vs_control']:g} times as often",
                            f"at least {cr['outward_fraction_min']:g} of its normals")
                        and all(z[1]["criteria"] == cr and z[1]["on_sheet_rule"] == V["on_sheet_rule"] for z in EXP), ""))
            pre = R / "results" / "fl" / f"PREREGISTRATION_FL_{suf}.md"
            if pre.exists():
                psha = _hlp.sha256(pre.read_bytes()).hexdigest()
                out.append((f"PATCHES {sn}: the registration quoted hashes to the file and to prereg.sha256",
                            f"{sn} (registration sha256 `{psha[:16]}`)" in sec
                            and psha in (R / "results" / "fl" / "prereg.sha256").read_text(), psha[:16]))
                pl = dict(re.findall(r"(pt\d+) \((\d+)\)", pre.read_text().split("off centre for", 1)[1].split(";")[0]))
                oc = {n.split("_")[0]: str(p["render"]["peak_layer"]) for n, p in P.items()
                      if "render" in p and abs(p["render"]["peak_layer"] - 33) >= _pm.OFF_CENTRE}
                out.append((f"PATCHES {sn}: the layers {_pm.OFF_CENTRE} or more from layer 33, recomputed here, are the "
                            "registration's off-centre list and are named", oc == pl and all(
                                f"{m} ({P[m]['render']['peak_layer']})" in flat for m in P if m.split("_")[0] in oc), f"{oc} vs {pl}"))
            ink, fl = [], R / "results" / "fl"
            if (fl / f"review_{suf}" / "UNSEALED.json").exists():
                U = json.loads((fl / f"review_{suf}" / "UNSEALED.json").read_text())
                ink.append(f"the review is {'VALID' if U['valid'] else 'VOID'} ({U['n_flagged_control_on_letters']} of the "
                           f"{U['n_control_windows_with_letters']} control windows")
                ink.append(f"flagged no {sn} window" if U["n_flagged_screen"] == 0 else f"flagged {U['n_flagged_screen']} {sn} window")
            if (fl / f"neighbour_{suf}" / "RESULT.json").exists():
                N = json.loads((fl / f"neighbour_{suf}" / "RESULT.json").read_text())
                ink.append(f"is {'VALID' if N['valid'] else 'VOID'}, and {N['n_pass_screen']} of "
                           f"{sum(not r['control'] for r in N['results'])} windows pass it")
            if (fl / f"confirm_{suf}" / "RESULT.json").exists():
                C = json.loads((fl / f"confirm_{suf}" / "RESULT.json").read_text())
                ink.append(f"The blind-side confirmation is {'VALID' if C['valid'] else 'VOID'}: the control passed")
            elif (fl / f"confirm_{suf}").exists():
                ink.append("The blind-side confirmation is running; its result is not in yet.")
            out.append((f"PATCHES {sn}: the ink screen quoted is UNSEALED.json, the neighbour RESULT.json and the confirmation "
                        "state (re-read here)", has(flat, *ink), " | ".join(ink)[:160]))
            for d, Ve, Be, Re in EXP:
                fo, Pe = str(d.relative_to(R)), Ve["patches"]
                prs = [q for q in sorted(fl.glob("PREREGISTRATION_FL_*.md")) if f"{fo}/validation.json" in q.read_text()]
                rvs = [q.parent for q in sorted(fl.glob("review_*/build.json"))
                       if {m for m in json.loads(q.read_text()).get("meshes") or [] if isinstance(m, str)} == set(Pe)]
                ok = len(prs) == 1 and len(rvs) == 1 and (rvs[0] / "UNSEALED.json").exists()
                det = f"{len(prs)} registrations, {len(rvs)} reviews"
                if ok:
                    esha = _hlp.sha256(prs[0].read_bytes()).hexdigest()
                    U, wk = json.loads((rvs[0] / "UNSEALED.json").read_text()), {}
                    for sh_ in json.loads((rvs[0] / "key.json").read_text())["sheets"].values():
                        if not sh_["control"]:
                            wk[sh_["mesh"]] = wk.get(sh_["mesh"], 0) + len(sh_["windows"])
                    a_, b_ = U["n_flagged_control_on_letters"], U["n_control_windows_with_letters"]
                    frag = [f"{sn} expansion, the {len(Pe)}", f"(registration sha256 `{esha[:16]}`)",
                            f"as a follow-up of the {sn} candidates above and not as an independent test",
                            f"The review is {'VALID' if U['valid'] else 'VOID'} ({a_} of the {b_} control windows on known "
                            "letters flagged)"]
                    if U["valid"] and U["n_flagged_screen"] == 0:
                        frag += [f"flagged none of the {sum(wk.values())} windows from {sum(1 for k_ in wk.values() if k_)} of "
                                 f"the {len(Pe)} patches", ": a null."]
                        frag += ["its sensitivity is low, so the null is weak evidence against text continuing there, not proof"
                                 ] if 2 * a_ < b_ else []
                    elif U["valid"]:
                        frag += [f"flagged {U['n_flagged_screen']} window"]
                    ok = has(flat, *frag) and esha in (fl / "prereg.sha256").read_text() and set(wk) <= set(Pe)
                    det = " | ".join(frag)[:160]
                out.append((f"PATCHES {sn}: the screen of the expansion {d.name} quoted is its registration (hash in "
                            "prereg.sha256), UNSEALED.json and key.json, found and re-counted here", ok, det))
                rs = (Re or {}).get("sheet_patches_sha256") or "unrecorded"
                out.append((f"PATCHES {sn}: the run record of the expansion {d.name} gives the sheet_patches.py shipped now, as "
                            "the section says", rs == psh and f"The {sn} expansion is a further run of the same file, unchanged"
                            in flat, rs[:16]))
            rp, run = R / x["folder"] / "public_repro.json", R / x["folder"] / "run.json"
            if rp.exists():
                rr = json.loads(rp.read_text())
                out.append((f"PATCHES {sn}: the reproduction quoted is public_repro.json, made with the sheet_patches.py shipped now",
                            rr["grids_all_bytes_identical"] == (f"byte-identical to the ones in the {sn} table" in flat)
                            and rr["public_sha256"] == psh, rr["public_sha256"][:16]))
            elif run.exists():
                rs = json.loads(run.read_text()).get("sheet_patches_sha256") or "unrecorded"
                out.append((f"PATCHES {sn}: its run record's sheet_patches.py is the one shipped now, as the section says",
                            rs == psh and f"{sn} is a further run of the same file, unchanged" in flat, rs[:16]))
            if Q:
                fails = [z["patch"] for z in Q["patches"] if not z["passes"]]
                out.append((f"PATCHES {sn}: the survey sheet-check counts quoted are survey_sheetcheck.json's (re-counted here)",
                            f"{Q['n_pass']} of the {Q['n']} {sn} patches" in flat and Q["n_pass"] == len(Q["patches"]) - len(fails)
                            and all(f"{z['patch']} ({z['tiles']} tile" in flat for z in Q["patches"] if z["tiles"] <= 1)
                            and all((f"It has not been run on the {m} {sn} expansion patches" if k == m else
                                     f"{k} of the {m} {sn} expansion patches") in flat
                                    for k, m in ((sum(n not in shp for n in z[1]["patches"]), len(z[1]["patches"])) for z in EXP)
                                    if k), f"{Q['n_pass']} of {Q['n']}"))
                if len(fails) >= 2:
                    top = sorted(PA, key=lambda n: (-PA[n]["vertical"]["switch_events"], int(n.split("_")[0][2:])))[:4]
                    res = {z["patch"]: z["passes"] for z in Q["patches"]}
                    ok = [n for n in top if res.get(n)]
                    ko = [n for n in top if res.get(n) is False]
                    rest = [n for n in fails if n not in top]
                    frag = ([f"{_pm._fmt_list(ok)} pass"] if ok else []) + ([f"{_pm._fmt_list(ko)} fail"] if ko else []) + \
                        ([f"the other {len(rest)} failing patches have at most "
                          f"{max(PA[n]['vertical']['switch_events'] for n in rest)} switch events each"] if rest else [])
                    out.append((f"PATCHES {sn}: the overlap of the survey's failures with the most switch events, recomputed "
                                "here from validation.json and survey_sheetcheck.json, is the one quoted", has(flat, *frag),
                                " | ".join(frag)[:160]))
            if _PRIOR:
                pt = _PRIOR[-1].read_text()
                m = re.fullmatch(r"(\d+)(\w*)", suf)
                lab = f"PHerc. {int(m.group(1))}{m.group(2)}" if m else sn
                srv = pt[pt.index("Survey of the"):].splitlines()
                row = next((l for l in srv if l.startswith(f"| {lab} |")), None)
                if row:
                    c = [z.strip() for z in row.split("|")[1:-1]]
                    out.append((f"PATCHES {sn}: the survey's own counts quoted are its README's survey table (re-read here)",
                                f"{lab} {c[1]} traced and {c[2]} passing its sheet check" in flat, f"{c[1]}, {c[2]}"))
        if raw_all:
            lead = f"{tot_n} validated sheet patches" if tot_pass == tot_n else f"{tot_n} sheet patches, {tot_pass} validated"
            out.append(("PATCHES: the totals across scrolls, summed here, are the README's lead and its table",
                        f"{lead}, {round(tot_area, 3):.3f} cm2 in all, on {len(names)} First Letters scrolls" in flat
                        and f"| all | {tot_n} | {tot_pass} | {round(tot_area, 3):.3f} |" in sec, f"{tot_n}, {tot_pass}, {tot_area:.3f}"))
            out.append(("PATCHES: every scroll with a patch folder on disk is in the summary", sorted(_disk) == sorted(names),
                        ", ".join(sorted(_disk))))
        if recs:
            pr, mk = recs[0], ("STEP_Z", "RAY", "DMAX", "FRAC_SP", "WMAX", "SP_MIN", "MIN_MATCH", "K2K_MAX", "COL_EVERY", "DS")
            out.append(("PATCHES: every build record of every scroll has the parameters the Method paragraph quotes",
                        all(all(r[k] == pr[k] for k in mk) for r in recs)
                        and has(flat, f"{pr['STEP_Z']} slices at a time", f"a ray of +-{pr['RAY']:g} voxels",
                                f"within {pr['DMAX']:g} voxels", f"at most {pr['WMAX']:g} voxels wide",
                                f"fewer than {pr['MIN_MATCH']:g} of the samples", f"one vertex every {pr['COL_EVERY'] * pr['DS']:g} voxels",
                                "Every scroll was traced with the same parameters."), f"{len(recs)} build records"))
        um = []
        for k in ("v1", "v2"):
            c = json.loads((R / "research" / "uncontested" / "phase2" / f"score_{k}.json").read_text())["checks_curve"] \
                if (R / "research" / "uncontested" / "phase2").exists() else None
            if c is not None:
                um.append(f"a median of {c['median_l0']} and a maximum of {c['max_l0']} over {c['planes']} planes" in flat and not c["PASS"])
        if um:
            out.append(("PATCHES: both umbilicus attempts are quoted from score_v1.json and score_v2.json and both failed",
                        all(um) and "Both failed" in flat, str(um)))
        if _SC:
            pc = _SC.get("positive_control") or {}
            out.append(("PATCHES: the survey sheet check's positive control is quoted from survey_sheetcheck.json, and the missing "
                        "negative control is stated", f"a good fraction of {pc.get('good_fraction', -1):g} over {pc.get('tiles')} tiles and "
                        f"{'passes' if pc.get('passes') else 'FAILS'}" in flat
                        and ("No negative control was run." in flat) == (not _SC.get("negative_control")), ""))
        if _PBODY:
            prn = re.search(r"villa_(\d+)_body", _PBODY[-1].name).group(1)
            url = _ps["prior_art"]["url"]
            out.append(("PATCHES: the related work names the villa PR and the survey repository of the saved PR body",
                        f"villa PR #{prn}" in flat and url in sec and url.split("github.com/", 1)[1] in _PBODY[-1].read_text(), prn))
        out.append(("PATCHES: the limits are stated", has(flat, "**Limits.**", "counted, not gated",
                                                        "has never been tested against a reference", "No letters are claimed",
                                                        "Flagged windows are candidates"), ""))
        return out

    _pl = next(l for l in _psec.splitlines() if l.startswith("| pt"))
    _pc = _pl.split("|"); _pc[5] = f" {float(_pc[5]) + 0.001:.3f} "
    _plant = _psec.replace(_pl, "|".join(_pc))
    _mx = next((x for x in _ps["scrolls"] if x.get("expansions")), None)
    _tn = f"**Totals.** {_mx['totals']['n']} patches," if _mx else None
    _plant2 = _psec.replace(_tn, f"**Totals.** {_mx['totals']['n'] - 1} patches,", 1) if _mx else None
    if _osp.environ.get("VERIFY_PLANT") == "PATCHES":
        print("  NOTE: VERIFY_PLANT=PATCHES, the checks below run on a copy with the first patch row's area raised by 0.001 cm2 "
              "and must FAIL")
        _psec = _plant
    if _osp.environ.get("VERIFY_PLANT") == "PATCHES_COUNT" and _mx:
        print(f"  NOTE: VERIFY_PLANT=PATCHES_COUNT, the checks below run on a copy whose {_mx['scroll']} totals say "
              f"{_mx['totals']['n'] - 1} patches instead of {_mx['totals']['n']}, and its merged-count check must FAIL")
        _psec = _plant2
    for _n, _ok, _d in _patches_checks(_psec):
        ck(_n, _ok, _d)
    if all(_raw(x)[0] is not None for x in _ps["scrolls"]):
        ck("PATCHES: the summary is exactly what the raw JSON gives now", _pm.build_summary(R) == _ps)
    else:
        print("  NOTE: raw validation.json and build.json are not in this repo; the section is checked against the summary")
    _caught = [n for n, ok, _ in _patches_checks(_plant) if not ok]
    ck("PATCHES control: a planted defect (the first patch row's area raised by 0.001 cm2) is caught by this block", bool(_caught),
       f"{len(_caught)} checks failed on it")
    if _mx and _raw(_mx)[0] is not None:
        _c2 = [n for n, ok, _ in _patches_checks(_plant2) if not ok and n.startswith(f"PATCHES {_mx['scroll']}: the patch count")]
        ck(f"PATCHES control: a planted wrong merged count ({_mx['scroll']} written as {_mx['totals']['n'] - 1} patches, not "
           f"{_mx['totals']['n']}) is caught by the merged-count check", _tn in README and len(_c2) == 1, f"{len(_c2)} caught")
    elif _mx:
        print("  NOTE: the raw patch folders are not in this repo, so the merged-count control is not run here")

print("\nHYGIENE")
for k, t in ALL.items():
    ck(f"{k}: no em or en dashes",
       not any(chr(0x2014) in l or chr(0x2013) in l for l in t.splitlines()))
    ck(f"{k}: no AI attribution", t.lower().count("claude") == 0)
    ck(f"{k}: no withdrawn mechanism claim",
       not any(p in t for p in ("mechanism is unknown", "refuted by the data",
                                "probably a different checkpoint", "different recipe from")))

print(f"\n{len(checks)} claims checked, {failed} disagreements")
(R/"results"/"verify_claims.json").write_text(json.dumps({"checks": checks, "failed": failed}, indent=2))
sys.exit(1 if failed else 0)
