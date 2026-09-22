# ARM X pre-registration, copied verbatim from the project's PREREGISTRATION.md

Written before any inference; the first inference record in `infer_runs.jsonl` is later. Corrections made after it are recorded in the project's STATE.md, not here.

## AMENDMENT 2026-09-22 23:3x, ARM X. WRITTEN BEFORE ANY INFERENCE. DOES THE FIRST LETTERS INSTRUMENT TRANSFER TO SCROLLS IT NEVER SAW?

**The gap.** `scrollprize/ink_9um` is the model the organisers' First Letters
workflow runs at the eligible 8.6 to 9.4 um resolution, and it is the instrument
behind every published First Letters negative: flummoxjr's corpus screen (0 of 71
scorable Grand Prize segments), TAUIL's PHerc1447 survey, SurgeFok's PHerc0800 run.
It was trained on four scrolls only (PHerc0139, PHerc1667, PHercParis4, PHerc0814),
and its card reports no number on any other scroll. The prizes page says "We don't
yet know whether our existing ink models will work on these scrolls." Prior-art
search this session (villa issues and PRs, GitHub repos, the August winners page)
found no cross-scroll evaluation of it. Russo's August benchmark of the 14
checkpoints and Gaona de Stefani's checkpoint averaging are the nearest work; both
are about choosing among checkpoints, not about transfer.

**Data, fixed now.** The organisers' own labels in
`huggingface.co/buckets/scrollprize/datasets`, tree `ink/`, surveyed by
`ops/xs_labels_survey.py` into `results/xs/labels_survey.json` BEFORE this text was
written. Every segment carries a 65 plane CT render, a 2D `_inklabels.tif` (0/255)
and a 2D `_supervision_mask.tif`.

    PRIMARY SET, native spacing known from meta.json, none of it in ink_9um training:
      ink/841/w00                                PHerc0841   4.681 um
      ink/841/auto_grown_20260220144552896       PHerc0841   4.681 um
      ink/841/auto_grown_20260220174252405       PHerc0841   4.681 um
      ink/unused/500p2a                          PHerc0500P2 4.32 um
    SECONDARY, spacing NOT recorded, so any number from them is labelled exploratory:
      ink/man5/MAN5_outer_3, ink/0009b/..._inp_hr_2um, ink/0500p2/-1
    EXCLUDED: ink/unused/P343p (every supervised pixel is labelled ink, AUC undefined)

**Resampling, fixed now.** 2x2x2 mean pooling of the render, so 4.681 um becomes
9.362 um (exactly the eligible PHerc volumes' grid) and 4.32 um becomes 8.64 um
(exactly the other eligible grid). 65 planes pool to 32. The labels sit at plane 32
of 65, so pooled plane 16; the 21 slice window is `[6, 27)`, which puts the labelled
plane at index 10 of 21, the same place the model's own training labels sit.
Labels pool as (mean of 2x2 >= 0.5), supervision as (all four supervised).

**Inference, fixed now.** villa's own `vesuvius.ink_detection.inference.infer`, at
the villa commit recorded in the run log, never a reimplementation. The only change
is the device: the non CUDA branch returns MPS, the same four line change as nerln's
villa PR #1865. Autocast stays off (its own guard is CUDA only), so this is fp32.
Both depth directions are run.

**Direction rule, label free, fixed now:** per segment and checkpoint, keep the
direction whose output has the larger separation, mean(p | p>0.5) minus
mean(p | p<=0.5), computed inside the supervision mask with NO label. The oracle
(better AUC of the two) is reported separately and labelled oracle.

**PRIMARY ENDPOINT, one: the median, over the 4 primary segments, of pixel ROC AUC
inside the supervision mask, for checkpoint `hybrid_3d2d-seed42/step-075000.pth`**
(the one TAUIL and SurgeFok used for published First Letters runs), direction by
the rule above.

    TRANSFERS   median >= 0.80 and no primary segment below 0.70
    DEGRADES    median in [0.65, 0.80)
    FAILS       median < 0.65

**Controls, and C1 gates everything.**

    C1 positive control, SAME pipeline, in distribution: pherc0139-w016, the ink_9um
       dataset's validation mask, input built from the public 2.399 um surface volume
       of 20250108000004-w029 by the card's recipe (XY level 2, centred 84 planes,
       4 plane mean). Required: AUC >= 0.85 on the validation pixels. kadenpool
       reports 0.912 on the challenge's own input for this segment (villa #1845).
       Below 0.85 the pipeline is broken and every other number is void.
    C2 shuffle floor: labels permuted in 64 px blocks inside the mask, 20 seeds.
       Required |AUC - 0.5| < 0.03, else the metric leaks structure.
    C3 orientation: the rejected direction's AUC is reported next to the chosen one,
       so a reader can see how much the direction matters on each segment.

**Secondary, reported but not the verdict:** all 14 checkpoints; the seed42 and
seed43 finals averaged; `hecate_9.6um` (ink/841 is not in the `unused` folder, so
hecate may have trained on PHerc0841; only 500p2a is clean for hecate, and that
caveat is printed beside every hecate number).

**What each outcome means for the public record, written now so it cannot be
fitted afterwards.** TRANSFERS: the published First Letters negatives rest on an
instrument that works off its training scrolls, which strengthens them. FAILS: those
negatives cannot distinguish "no ink" from "the model does not transfer", which is
the organisers' own open question, and a per scroll fine tune is the first thing to
try on any unread scroll. DEGRADES: both, quantified.

**Abandon rule.** If C1 fails twice after fixing a found defect, stop and report the
pipeline problem instead of a transfer number.
