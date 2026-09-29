## Does teaching ink_9um to ignore depth-shuffled input help it read the eligible scans?

**No.** Registered verdict NULL: the median change in real-label AUC (fine-tuned minus control) is -0.0047, and the
fine-tuned model is higher on 1 of 4 crops. A win needed at least +0.010 and 3 of 4.

The planted-letter test above found that ink_9um responds more to ink whose depth planes are shuffled than to the same ink
in the right order. If that response is part of what hides letters, removing it should raise the AUC on real labels. This
tests that directly.

**What was run.** Registered before any fine-tuned weights existed ([registration](results/spec/PREREGISTRATION_SPEC.md),
sha256 `4df26a9543340f7e`). The released ink_9um seed42 step-075000 was fine-tuned without labels on the inputs of
the four eligible-type crops scored above: the frozen released model's own output is the target (self-distillation),
BatchNorm frozen, 1500 steps, on an M1 Max GPU. Treatment: 4 of every 20 patches have their depth planes shuffled, with
target 0 (no ink). Control: the same 4 patches in the right order, with the released model's output as the target. The
labels are read only by the scorer.

| crop | released | control | treatment | treatment minus control | confident ink on depth-shuffled input, control / treatment | planes rolled by half (never trained), control / treatment |
|---|---|---|---|---|---|---|
| PHerc0841 w00 | 0.7680 | 0.7680 | 0.7562 | -0.0118 | 0.1783 / 0.0000 | 0.1897 / 0.0315 |
| PHerc0841 ag144 | 0.7599 | 0.7599 | 0.7521 | -0.0078 | 0.1118 / 0.0000 | 0.1558 / 0.0451 |
| PHerc0841 ag174 | 0.7614 | 0.7615 | 0.7809 | +0.0195 | 0.1147 / 0.0000 | 0.1349 / 0.0182 |
| PHerc0500P2 | 0.7537 | 0.7537 | 0.7520 | -0.0017 | 0.0324 / 0.0000 | 0.2562 / 0.0767 |

**The fine-tune did what it was told.** On depth-shuffled input the treatment calls 0.0000 of the pixels confident
ink on every crop (control 0.0324 to 0.1783), and it also stops responding to a disorder it never saw in training (planes
rolled by half: 0.0182 to 0.0767 against 0.1349 to 0.2562). **Real-label AUC did not move.**

**Planted letters, the same two models** (the test above, same inputs, same scorer):

| crop | stroke AUC, control / treatment | counter contrast C, control / treatment | shuffled-ink AUC, control / treatment |
|---|---|---|---|
| PHerc0841 w00 | 0.7059 / 0.6767 | 0.7040 / 0.6955 | 0.8109 / 0.5077 |
| PHerc0841 ag144 | 0.6903 / 0.6193 | 0.7349 / 0.8254 | 0.8982 / 0.3856 |
| PHerc0841 ag174 | 0.6205 / 0.6268 | 0.9419 / 0.8296 | 0.8116 / 0.4044 |
| PHerc0500P2 | 0.6148 / 0.4962 | 1.2688 / undefined | 0.8787 / 0.3855 |

On depth-shuffled planted ink the treatment's AUC falls from between 0.8109 and 0.8982 to between 0.3855 and 0.5077, so it now rates shuffled ink below
the non-ink transplant on 3 of 4 crops. It also responds less to the planted ink in the right order: stroke AUC falls
on 3 of 4 crops (median change -0.0501) and drops below the 0.55 floor on PHerc0500P2, where C is undefined.
Its AUC on real labels, above, stays where it was.

**What it means.** The response to depth-shuffled input can be removed with no measurable change in real-label AUC, so it
is not what limits ink_9um on these scans, and training with depth-shuffled negatives should not be expected to raise AUC
here. Planted ink, on the other hand, loses much of its signal once that response is gone while real ink does not. So part
of what makes planted ink visible to ink_9um is a depth cue that real ink in place does not need, and the planted-letter
test above should be read with that limit in mind.

**Limits.** Transductive (the fine-tune saw the scored crops' inputs, never their labels), one seed, 1500 steps, four crops
from two scrolls, 113 keV scans only. The control stays within 2.3e-05 AUC of the released model (its gradient is near zero
from the start), so "treatment minus control" is in effect "treatment minus released".

Code: [bench/spec/](bench/spec/). Scores: [results/spec/](results/spec/).
