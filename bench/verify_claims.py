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
