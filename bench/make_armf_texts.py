"""GENERATE the ARM F outward texts from the JSON: villa issue, #1582 comment, Discord post, form field 5
(ARM Y/Z plus ARM F, for the fourth September response). Every number comes from results/*.json.

    env/bin/python3 ops/make_armf_texts.py
"""
import json, pathlib
import numpy as np
R = pathlib.Path(__file__).resolve().parents[1]; F = R / "results" / "fs"
A = json.load(open(F / "armf_scores.json")); N14 = json.load(open(F / "armf_ndh14.json"))["summary"]
F5 = json.load(open(F / "f5_scores.json")); F6 = json.load(open(F / "f6_scores.json"))
Y = json.load(open(R / "results" / "ys" / "army_scores.json")); Z = json.load(open(R / "results" / "zs" / "armz_scores.json"))
P = {s: json.load(open(F / f"profile_{s}.json")) for s in ("w00", "ag144", "ag174")}
p800 = json.load(open(F / "p0800_depth_probe.json"))
SEGS = ["w00", "ag144", "ag174"]; PR = "hybrid_3d2d-seed42_step-075000"
REPO = "https://github.com/AndreasHad04/villa-apple-silicon"
SEC = REPO + "#why-ink_9um-reads-the-eligible-12-m-scans-worse"


def a(s, c): return A["segments"][s][c][PR]["auc_chosen"]
def med(c): return float(np.median([a(s, c) for s in SEGS]))
def sg(x): return f"{x:+.4f}"


gap = med("P0") - med("N0"); f1 = A["F1"]; f2 = A["F2"]
pdz = float(np.median([a(s, "P0") - a(s, "PDEGZ") for s in SEGS]))
ndh = float(np.median([a(s, "NDH") - a(s, "N0") for s in SEGS]))
corr_n = float(np.mean([np.mean(P[s]["layer_corr_native"]) for s in SEGS])); corr_p = float(np.mean([np.mean(P[s]["layer_corr_pooled"]) for s in SEGS]))
corr_800 = float(np.mean(p800["adjacent_corr"]))
f5c = F5.get("F5c"); f6 = F6.get("F6")
fresh = []
w043 = F5["auc"]["w043"]["NDH"] - F5["auc"]["w043"]["N0"]
if f5c: fresh.append(("PHerc0139 (a training scroll, whose native scan supplied 5 training representations)", f5c["per_segment"], f5c["median_gain"], f5c["verdict"]))
if f6: fresh.append(("PHerc0009B (8.64 um, 116 keV) and PHerc0500P2 (9.362 um, 113 keV), never in training", f6["per_segment"], f6["mean_gain"], f6["verdict"]))
f6ok = bool(f6 and f6["verdict"] == "transfers"); f5ok = bool(f5c and f5c["verdict"] == "transfers")
if f6ok: head = "depth sharpening transfers to the two held-out scrolls tested"
elif f6 and f6["verdict"] == "small, as measured": head = "depth sharpening helps on held-out scrolls, by less than on PHerc0841"
else: head = "depth sharpening does not transfer reliably, so the tool is a diagnostic, not a recommendation"
recover = ("a fixed depth filter recovers part of it on the held-out scrolls tested" if f6ok else
           "a fixed depth filter recovers part of it on PHerc0841 and less on held-out scrolls" if f6 and f6["verdict"] == "small, as measured" else
           "a fixed depth filter recovers part of it on PHerc0841 but did not transfer to the held-out scrolls tested")
which = ("shows which cheap fix does not work and which does" if f6ok else "shows which cheap fix does not work, and how far a depth filter carries")


nd6 = {n: v["ND"] - v["N0"] for n, v in F6["auc"].items() if v.get("ND") is not None and v.get("N0") is not None}


def fresh_lines():
    out = []
    for lab, per, m, v in fresh:
        tail = (f" Descriptive, not in the endpoint: w043, whose 2.399 um render is a training representation, {sg(w043)}. The in-plane variants did not"
                f" transfer here either (NSH {sg(F5['F5']['median_gain'])}, NSDH {sg(F5['F5b']['median_gain'])}).") if "PHerc0139" in lab else ""
        out.append(f"- {lab}: " + ", ".join(f"{k} {sg(x)}" for k, x in per.items()) + f"; {'median' if 'PHerc0139' in lab else 'mean'} {sg(m)}, {v}.{tail}")
    if nd6: out.append("- Chosen after seeing those numbers, so exploratory: on the two scrolls never in training the depth filter alone, without the"
                       " PHerc0841 intensity map (`--no-map`), did better: " + ", ".join(f"{k} {sg(x)}" for k, x in nd6.items()) + f", mean {sg(float(np.mean(list(nd6.values()))))}.")
    return "\n".join(out)


body_core = f"""**What was measured.** ink_9um (the First Letters model, primary checkpoint seed42 step-075000) on three PHerc0841 segments, each in two organiser arrays: the 2.403 um volume through villa's own 9 um recipe, and the eligible 9.366 um 1.2 m volume. Same labels (the organisers' 20260918 set), same model, both depth directions with a label-free choice. The eligible array reads worse: median AUC {med('N0'):.4f} against {med('P0'):.4f}. Every First Letters volume is a 1.2 m scan, so this gap is the one a hunter pays.

**It is not geometry.** Without labels or a model, the eligible render's sheet sits within half a layer of the 2.403 um one in depth. It differs in blur, in depth and in plane: its adjacent layers correlate {corr_n:.3f} on average against {corr_p:.3f} for the 2.403 um input the model mostly trained on (and {corr_800:.3f} on a PHerc0800 8.64 um render, the other eligible configuration).

**The pre-registered fix failed.** Sharpening in plane plus an intensity map, fitted without labels on the other two segments' paired arrays, changed the median AUC by {sg(f1['median_gain'])}. Blurring the 2.403 um input in plane to look like the eligible one costs only {f2['median_drop']:.4f} ({100 * f2['median_drop'] / gap:.0f}% of the {gap:.4f} gap); blurring it in depth as well costs {pdz:.4f} ({100 * pdz / gap:.0f}%).

**Depth is the larger lever (found after the primary failed, so tested again on fresh data).** Sharpening each pixel's 21-sample depth profile with a fixed gain, plus the same intensity map, raised the PHerc0841 median by {sg(ndh)} and improved {N14['pairs_improved']} of {N14['pairs']} checkpoint and segment pairs across all 14 released checkpoints. Fresh tests, transforms fitted on PHerc0841 only:
{fresh_lines()}"""

tool = f"""**Tool.** `bench/depth_sharpen_9um.py` in {REPO} applies that fixed depth filter to a 21-layer input window (`python bench/depth_sharpen_9um.py IN.zarr OUT.zarr --layer-um 9.362`); it reproduces the tested inputs byte for byte. Pre-registration with four timestamped amendments, scripts and JSON are in the same repository: {SEC}"""

limits = """**Limits.** One checkpoint for the endpoints. Two labelled PHerc0841 segments per fit. PHerc0139's native scan is part of ink_9um's training, which weakens it as a fresh test. Only one labelled segment each on PHerc0009B and PHerc0500P2."""

issue = f"""ink_9um on the eligible 1.2 m scans: depth blur is the largest identified cause of the AUC loss; {head}

{body_core}

{tool}

{limits}

**On the training side:** the model saw mostly 2.4 um-derived inputs, which are sharper in depth than every 1.2 m render measured here (PHerc0841, PHerc0800). villa's default ink recipe already blurs all three axes on a minority of patches (per-axis sigma 0.3 to 1.5); the eligible scans correspond to a fixed blur inside that range (see the README's Gaussian calibration) applied to every input. A training arm that applies that calibrated depth blur to most patches, or more native 1.2 m renders in training, is the untested direct fix.
"""
c1582 = f"""A follow-up to my 2026-09-23 comment here, on WHY the native family reads worse on an unseen scroll. Label-free, the 9.366 um render differs from the pooled 2.403 um one in blur, in depth as well as in plane, not in placement; the pre-registered primary endpoint, an in-plane fix, failed ({sg(f1["median_gain"])}); blurring the pooled input in depth as well as in plane reproduces {100 * pdz / gap:.0f}% of the family gap, against {100 * f2['median_drop'] / gap:.0f}% for in plane alone; and {recover}. Full write-up and fresh-scroll tests: {SEC}
"""
discord = f"""ink_9um on the eligible 1.2 m scans: depth blur is the largest identified part of the AUC it loses against 2.4 um inputs (in-plane resolution is a smaller part, geometry none). On PHerc0841 a fixed, label-free depth filter raised the median by {sg(ndh)} ({N14['pairs_improved']} of {N14['pairs']} checkpoint pairs); on held-out scrolls it gives a small gain: {'; '.join(("PHerc0139 median " if "PHerc0139" in lab else "PHerc0009B and PHerc0500P2 mean ") + sg(m) + (" (a training scan)" if "PHerc0139" in lab else " (never in training)") for lab, _, m, _ in fresh)}. The pre-registered primary endpoint (an in-plane fix) failed. Tool and numbers: {SEC} Discussion: https://github.com/ScrollPrize/villa/issues/1898
"""
yz = Y["Y2"]; zz = Z["Z1"]
form = f"""This adds to our three September responses, correcting ARM X and extending it. Scroll data: PHerc0841 segments w00, auto_grown_20260220144552896 and auto_grown_20260220174252405 in three organiser arrays (the 4.681 um label-bucket renders, the 2.403 um surface volumes, and the eligible 9.366 um surface volumes), plus PHerc0139, PHerc0009B and PHerc0500P2 segments for fresh tests, all scored on the organisers' 20260918 labels with scrollprize/ink_9um through villa's own inference.

First, ink_9um reads an unseen scroll about as well as its own on the arrays hunters use: median AUC {yz['render_median']:.4f} on the renders, {yz['published_median']:.4f} on the 2.403 um volumes, {zz['eligible_median']:.4f} on the eligible 9.366 um volumes, against {yz['c1_in_distribution']:.4f} in distribution. ARM X's weaker PHerc0841 figures were a property of the renders.

Second, why the eligible array reads worse, which matters because every First Letters volume is a 1.2 m scan. {body_core.split('**It is not geometry.**')[1].strip()}

How it increases the probability of reading the scrolls: it names the largest identified loss on the eligible scans (depth blur), {which}, and gives hunters a tested input filter plus a training-side target for the organisers. Evidence: pre-registrations with timestamped amendments, controls that reproduce earlier results exactly, and a machine checker over every quoted number. Code and data: {SEC}
"""
form = form.replace("**", "").replace("`", "")  # a Google Form field is plain text
for path, text in ((F / "ISSUE_armf.md", issue), (F / "COMMENT_1582_armf.md", c1582), (F / "DISCORD_armf.md", discord), (R / "results" / "FORM_FIELD5_ARMF.txt", form)):
    path.write_text(text); print(path.name, len(text), "chars")
(R / "results" / "FORM_FIELD4_ARMF.txt").write_text(SEC + "\nhttps://github.com/ScrollPrize/villa/issues/1898\n" + REPO + "#the-same-pherc0841-segments-on-the-published-surface-volumes-including-the-eligible-scan\n")
