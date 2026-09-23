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
