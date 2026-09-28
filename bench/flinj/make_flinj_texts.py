"""GENERATE the FLINJ outward texts from results/flinj/report_data.json: the public README section, form field 5 and a
Discord post. Every number is computed here from the JSON; the verdict comes from ops/flinj/report.py's own verdict().

    env/bin/python3 ops/flinj/make_flinj_texts.py
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import report as RP  # noqa: E402

F = CM.RES
H = json.load(open(os.environ.get("FLINJ_JSON") or F / "report_data.json"))  # verify_claims points this at the published copy
V, _ = RP.verdict(H)
assert V == "STROKES", V
INC = [h for h in H if not H[h].get("excluded")]
NAME = {"w00": "PHerc0841 w00", "ag144": "PHerc0841 ag144", "ag174": "PHerc0841 ag174", "p0500p2": "PHerc0500P2"}
REPO = "https://github.com/AndreasHad04/villa-apple-silicon"
SEC = REPO + "#can-ink_9um-draw-a-letter-on-the-eligible-scan-type"
pr = {h: H[h]["pairs"]["primary"] for h in INC}
assert all(pr[h]["b3_defined"] for h in INC)
C = {h: pr[h]["C_b3"] for h in INC}; med = float(np.median(list(C.values())))
d = {h: pr[h]["d_prob"] for h in INC}; auc = {h: pr[h]["auc_pair"] for h in INC}
real = {h: H[h]["target"] for h in INC}; gap = {h: real[h] - auc[h] for h in INC}
c3 = {h: H[h]["pairs"]["c3"] for h in INC}; n3 = sum(bool(v["b3_defined"] and v["C_b3"] >= 0.5) for v in c3.values())
c1 = {h: H[h]["pairs"]["c1"]["auc_pair"] for h in INC}; c2 = {h: H[h]["pairs"]["c2"]["auc_pair"] for h in INC}
s3 = {h: H[h]["pairs"]["s3"]["C"] for h in INC}


def rng(x, k=2):
    v = list(x.values())
    return f"{min(v):.{k}f} to {max(v):.{k}f}"


rows = "\n".join(f"| {NAME[h]} | {C[h]:.4f} | {d[h]:.4f} | {auc[h]:.4f} | {real[h]:.4f} |" for h in INC)
section = f"""## Can ink_9um draw a letter on the eligible scan type?

On the eligible 1.2 m scans, ink_9um finds ink but draws it as blobs, never as letters. Two causes are possible and
they call for different work: the model cannot draw strokes when ink is as faint as it is at 9 um, or the ink in these
scans is not stroke-shaped where the model looks.

To separate them I planted real ink in the shape of Greek capitals into eligible-type scans and checked whether the
model draws the letters back. The ink is real: voxel columns cut from labelled ink on PHerc0841's 9.366 um, 113 keV
scan, moved into other crops under a letter-shaped mask (cap height 2.5 mm, strokes 0.35 mm). The control is the same
transplant made of non-ink papyrus, so both runs share every seam and differ only in the ink; the difference of the two
predictions is what gets scored. Four host crops, each receiving ink from other crops than itself: three PHerc0841
segments and PHerc0500P2 (9.362 um, 113 keV). The test, its metric and its thresholds were registered before any input
existed; four amendments fixed design problems found by model-free checks and by one smoke run of the first design,
all before the final design's outcome was computed. Everything is in the same folder.

The score is counter contrast C: how much emptier the inside of a letter (the hole in O, A, Theta) is than its
strokes, measured against plain background. 1 is a drawn letter, 0 a filled blob, and above 1 the inside of the
letter reads even lower than plain background. Registered bars: 0.5 or more reads as strokes, 0.2 or less as blobs.

| host | C | response on strokes (probability) | AUC, ink vs non-ink transplant, stroke pixels | same model on the crop's real labelled ink |
|---|---|---|---|---|
{rows}

**The model keeps the letter shape: median C {med:.4f}, the registered verdict is STROKES.** The positive control (a
clean letter-shaped bump) is drawn as a letter on {n3} of {len(INC)} hosts, as the registration requires. The mean of all
14 released checkpoints gives C {rng(s3)}.

**What hides the letters is how weakly it responds.** On stroke pixels the planted ink is told apart from the non-ink
transplant with an AUC of {rng(auc)}, which is {rng(gap)} below the same model on the real labelled ink of the same crops.
In the prediction itself the planted letters barely rise above the model's response to the host papyrus (figure:
mask, prediction with the non-ink transplant, prediction with the ink transplant, difference).

![FLINJ w00](figures/flinj_w00.png)

**And it responds more to disorder than to ink.** The same planted ink with its depth planes shuffled is detected more
than the real ink (AUC {rng(c1)}), while a plain brightness bump is not detected at all (AUC {rng(c2)}). A model that
fires harder on depth-disordered texture than on ink paints blobs wherever a surface is disordered.

So on this scan type what stands between ink_9um and letters is not its ability to draw strokes but its sensitivity,
and its response to depth disorder. A fine-tune aimed at that second part is running.

Limits: 113 keV scans only (the one 116 keV crop had no room for a row of letters clear of its own ink); the planted
ink is real but tiled from 32-pixel patches of tracings about 1 mm wide, and the model detects it less well than real
labelled ink, so the transplant loses part of what the model uses; the primary is one checkpoint (seed 42, step 75,000).

Code: `bench/flinj/`. Registration, amendments and the full generated report: `results/flinj/`.
"""

field5 = f"""Which data: PHerc0841's 9.366 um scan (three labelled segments) and PHerc0500P2's 9.362 um scan, both taken with the 1.2 m / 113 keV setup of the First Letters scans.

How it raises the chance of reading: it tells First Letters work where not to spend effort. I planted real ink, cut from labelled PHerc0841 ink, in the shape of Greek letters into eligible-type scans, with an identical non-ink transplant as control, and ran ink_9um on both. The model keeps the letter shape (counter contrast median {med:.4f}, registered verdict STROKES, positive control passed on {n3} of {len(INC)} hosts). What hides letters is weak response (AUC {rng(auc)} against the non-ink control, {rng(gap)} below the same model on real labelled ink) and a stronger response to depth-disordered input (the same ink with shuffled depth planes: AUC {rng(c1)}).

What it enables: a test with known letters on the eligible scan type, so any 9 um model or pipeline can be checked for whether it can show a letter before it is run on a scroll nobody can read yet.

Evidence: pre-registered test (four amendments, all made before the final design's outcome was computed), full report, figures and code: {SEC}
"""

discord = f"""Can ink_9um draw a letter on the eligible scan type? I planted real PHerc0841 ink in the shape of Greek letters into 9.36 um / 113 keV scans (non-ink transplant as control). It keeps the letter shape: median counter contrast {med:.4f}, pre-registered verdict STROKES. What hides letters is weak response (AUC {rng(auc)} vs the control) and a stronger response to depth-shuffled ink (AUC {rng(c1)}). Details, figure and code: {SEC}"""

OUTDIR = pathlib.Path(os.environ.get("FLINJ_TEXT_OUT") or F)  # verify_claims regenerates into a temp dir
for name, text in (("PUBLIC_SECTION.md", section), ("FORM_FIELD5_FLINJ.txt", field5), ("DISCORD_FLINJ.md", discord)):
    assert not any(chr(c) in text for c in (0x2014, 0x2013)) and "claude" not in text.lower(), name
    (OUTDIR / name).write_text(text)
    print(f"wrote {OUTDIR / name} ({len(text)} chars)")
