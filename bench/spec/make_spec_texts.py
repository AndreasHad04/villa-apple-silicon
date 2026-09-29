#!/usr/bin/env python3
"""GENERATE the ARM SPEC outward texts (public README section and a short Discord follow-up) from
results/spec/spec_scores.json and results/spec/flinj/rescore.json. Every number is computed here; none is typed.

    env/bin/python3 ops/spec/make_spec_texts.py            writes results/spec/PUBLIC_SECTION.md, DISCORD_SPEC.md
    SPEC_TEXT_OUT=<dir> SPEC_JSON=<scores> SPEC_RESCORE=<rescore>   (how verify_claims regenerates from a published copy)
"""
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "spec"
S = json.load(open(os.environ.get("SPEC_JSON") or RES / "spec_scores.json"))
Q = json.load(open(os.environ.get("SPEC_RESCORE") or RES / "flinj" / "rescore.json"))
OUT = Path(os.environ.get("SPEC_TEXT_OUT") or RES)
CROPS = ["w00", "ag144", "ag174", "p0500p2"]
NAME = {"w00": "PHerc0841 w00", "ag144": "PHerc0841 ag144", "ag174": "PHerc0841 ag174", "p0500p2": "PHerc0500P2"}
REPO = "https://github.com/AndreasHad04/villa-apple-silicon"
SEC = REPO + "#does-teaching-ink_9um-to-ignore-depth-shuffled-input-help-it-read-the-eligible-scans"

assert S["verdict"] == "NULL", S["verdict"]
assert all(S["positive_control"].values()) and all(S["manipulation"].values())
assert all(v == 0 for v in Q["control"].values()), Q["control"]
f4 = lambda v: f"{v:.4f}"
rng = lambda xs: f"{min(xs):.4f} to {max(xs):.4f}" if min(xs) != max(xs) else f"{min(xs):.4f}"
btw = lambda xs: f"{min(xs):.4f} and {max(xs):.4f}"
row = lambda arm, c: S["rows"][arm][c]
delta, nwin = S["delta_median"], S["t_beats_k"]
perm_T, perm_K = [row("T", c)["permuted"] for c in CROPS], [row("K", c)["permuted"] for c in CROPS]
roll_T, roll_K = [row("T", c)["rolled"] for c in CROPS], [row("K", c)["rolled"] for c in CROPS]
kb = max(abs(v) for v in S["k_minus_b"].values())
P = lambda arm, h, k, q: Q["rows"][arm][h][k][q]
st_K, st_T = [P("K", h, "primary", "auc_pair") for h in CROPS], [P("T", h, "primary", "auc_pair") for h in CROPS]
c1_K, c1_T = [P("K", h, "c1", "auc_pair") for h in CROPS], [P("T", h, "c1", "auc_pair") for h in CROPS]
Cfmt = lambda v: "undefined" if v is None else f4(v)
c1_below = sum(t < 0.5 for t in c1_T)
st_fell = sum(t < k for t, k in zip(st_T, st_K))
below = [h for h, t in zip(CROPS, st_T) if t < 0.55]
below_names = " and ".join(NAME[h] for h in below) if below else "no crop"
assert all(P("T", h, "primary", "C_b3") is None for h in below)   # below the floor, C must be the undefined one
st_med_d = statistics.median(t - k for t, k in zip(st_T, st_K))
c1_med_d = statistics.median(t - k for t, k in zip(c1_T, c1_K))

t1 = ["| crop | released | control | treatment | treatment minus control | confident ink on depth-shuffled input, control / treatment | planes rolled by half (never trained), control / treatment |",
      "|---|---|---|---|---|---|---|"]
for c in CROPS:
    t1.append(f"| {NAME[c]} | {f4(row('B', c)['real']['auc'])} | {f4(row('K', c)['real']['auc'])} | {f4(row('T', c)['real']['auc'])} | "
              f"{S['t_minus_k'][c]:+.4f} | {f4(row('K', c)['permuted'])} / {f4(row('T', c)['permuted'])} | "
              f"{f4(row('K', c)['rolled'])} / {f4(row('T', c)['rolled'])} |")
t2 = ["| crop | stroke AUC, control / treatment | counter contrast C, control / treatment | shuffled-ink AUC, control / treatment |",
      "|---|---|---|---|"]
for h in CROPS:
    t2.append(f"| {NAME[h]} | {f4(P('K', h, 'primary', 'auc_pair'))} / {f4(P('T', h, 'primary', 'auc_pair'))} | "
              f"{Cfmt(P('K', h, 'primary', 'C_b3'))} / {Cfmt(P('T', h, 'primary', 'C_b3'))} | "
              f"{f4(P('K', h, 'c1', 'auc_pair'))} / {f4(P('T', h, 'c1', 'auc_pair'))} |")

section = f"""## Does teaching ink_9um to ignore depth-shuffled input help it read the eligible scans?

**No.** Registered verdict NULL: the median change in real-label AUC (fine-tuned minus control) is {delta:+.4f}, and the
fine-tuned model is higher on {nwin} of 4 crops. A win needed at least +0.010 and 3 of 4.

The planted-letter test above found that ink_9um responds more to ink whose depth planes are shuffled than to the same ink
in the right order. If that response is part of what hides letters, removing it should raise the AUC on real labels. This
tests that directly.

**What was run.** Registered before any fine-tuned weights existed ([registration](results/spec/PREREGISTRATION_SPEC.md),
sha256 `{S['registration'][:16]}`). The released ink_9um seed42 step-075000 was fine-tuned without labels on the inputs of
the four eligible-type crops scored above: the frozen released model's own output is the target (self-distillation),
BatchNorm frozen, 1500 steps, on an M1 Max GPU. Treatment: 4 of every 20 patches have their depth planes shuffled, with
target 0 (no ink). Control: the same 4 patches in the right order, with the released model's output as the target. The
labels are read only by the scorer.

{chr(10).join(t1)}

**The fine-tune did what it was told.** On depth-shuffled input the treatment calls {rng(perm_T)} of the pixels confident
ink on every crop (control {rng(perm_K)}), and it also stops responding to a disorder it never saw in training (planes
rolled by half: {rng(roll_T)} against {rng(roll_K)}). **Real-label AUC did not move.**

**Planted letters, the same two models** (the test above, same inputs, same scorer):

{chr(10).join(t2)}

On depth-shuffled planted ink the treatment's AUC falls from between {btw(c1_K)} to between {btw(c1_T)}, so it now rates shuffled ink below
the non-ink transplant on {c1_below} of 4 crops. It also responds less to the planted ink in the right order: stroke AUC falls
on {st_fell} of 4 crops (median change {st_med_d:+.4f}) and drops below the 0.55 floor on {below_names}, where C is undefined.
Its AUC on real labels, above, stays where it was.

**What it means.** The response to depth-shuffled input can be removed with no measurable change in real-label AUC, so it
is not what limits ink_9um on these scans, and training with depth-shuffled negatives should not be expected to raise AUC
here. Planted ink, on the other hand, loses much of its signal once that response is gone while real ink does not. So part
of what makes planted ink visible to ink_9um is a depth cue that real ink in place does not need, and the planted-letter
test above should be read with that limit in mind.

**Limits.** Transductive (the fine-tune saw the scored crops' inputs, never their labels), one seed, 1500 steps, four crops
from two scrolls, 113 keV scans only. The control stays within {kb:.1e} AUC of the released model (its gradient is near zero
from the start), so "treatment minus control" is in effect "treatment minus released".

Code: [bench/spec/](bench/spec/). Scores: [results/spec/](results/spec/).
"""

discord = f"""Follow-up to the planted-letter post: I fine-tuned ink_9um (no labels, self-distillation) so that it calls depth-shuffled input "no ink". It learned that completely ({rng(perm_T)} confident ink on shuffled input, control {rng(perm_K)}), and real-label AUC on the same four eligible-type crops did not move (median {delta:+.4f}, pre-registered verdict NULL). So the stronger response to depth-shuffled ink is not what holds the model back on these scans. The same fine-tune also responds less to the planted letters (stroke AUC down on {st_fell} of 4 crops) while real-label AUC holds, so part of what makes planted ink visible is a depth cue that real ink does not need. Details and code: {SEC}
"""

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "PUBLIC_SECTION.md").write_text(section)
(OUT / "DISCORD_SPEC.md").write_text(discord)
for n, t in (("PUBLIC_SECTION.md", section), ("DISCORD_SPEC.md", discord)):
    assert not any(ch in t for ch in (chr(0x2014), chr(0x2013))), n
    print(f"wrote {OUT / n} ({len(t)} chars)")
