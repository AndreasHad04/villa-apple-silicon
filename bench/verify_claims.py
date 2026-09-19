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
