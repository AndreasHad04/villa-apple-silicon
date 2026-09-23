# ARM Z: ink_9um on the ELIGIBLE native 9.366 um scan of the same PHerc0841 segments

Written 2026-09-23 14:0x local, after ARM Y's production-window result and BEFORE
any ARM Z data is read. ARM Y found ink_9um at 0.8176 to 0.8531 on the published
2.403 um (0.22 m, 77 keV) surface volumes pooled to about 9.6 um, the kind of input
it was trained on, against 0.6743 to 0.6989 on the organisers' 4.681 um (1.2 m,
113 keV) renders. First Letters evidence must come from the eligible 8.6 to 9.4 um
volumes, which for PHerc0841 is the 1.2 m, 113 keV scan. The ink_9um card: trained
on 2.399 um volumes pooled to about 9.6 um, plus 5 native 9.362 um segments, all
PHerc0139.

## Arrays

- Input: `s3://vesuvius-challenge-open-data/PHerc0841/segments/<seg>/surface-volumes/9.366um-1.2m-113keV-volume-20250821151531.zarr`,
  level 0, 28 layers, for w00, ag144 and ag174 (the ARM Y segments).
- Window: 21 layers, `centered_slice(28, 21)` from villa's production module
  (ceil, so [4, 25)). No pooling: the volume is already at the eligible spacing.
- Labels: ARM Y's level-2 labels (published 20260918, 2.403 um grid, 9.612 um per
  pixel), resampled nearest-neighbour onto the 9.366 um canvas at scale 9.366/9.612,
  zero offset. The canvases differ by 3.9005 and 3.8992 against a voxel ratio of
  3.8976, so a pure scale is expected.
- Inference and scoring: ARM X's path (ops/xs_infer.py via ops/ys_infer.py) and
  ARM Y's scorer, both directions, label-free rule.

## Control, fixed now

- Z-ALIGN: normalised cross-correlation between the 9.366 um volume's window-mean
  image and ARM Y's production-window mean image resampled onto the same canvas,
  both high-passed, over integer shifts within +/-8 px. Zero offset is used when the
  peak lies within 2 px of zero; otherwise the peak shift is used when the peak NCC
  is at least 3x the median absolute NCC over the search window; otherwise the
  segment is excluded as unaligned.

## Endpoint, fixed now

- Z1: PHerc0841 median AUC, primary checkpoint (seed42 step-075000), label-free
  direction, on the eligible array. Against ARM Y's 0.8294 (same labels, 2.403 um
  array) and ARM X's 0.6931 (4.681 um renders, same scan family as the eligible one).
  At or below 0.6931 + 0.03: "the loss is the scan: ink_9um reads the eligible
  1.2 m scan worse than the 0.22 m scan it was trained on". At or above
  0.8294 - 0.03: "the eligible scan reads as well as the training kind; ARM X's
  loss was the render". Otherwise reported as measured, between the two.
- Direction on the eligible array is reported per segment, not scored as an
  endpoint.

## Amendment 2026-09-23 14:0x, BEFORE any ARMZ14 inference: all 14 checkpoints on the eligible array

Same arrays, window, labels, offsets and scorer as Z1; all 14 ink_9um checkpoints plus
the averaged finals, both directions, label-free choice. Reported, descriptive only:
- Z2: Spearman, over the 14 checkpoints, between in-distribution AUC (ARM X C1) and the
  eligible PHerc0841 median. ARM X measured 0.160 on the renders.
- Z3: Spearman between the eligible median and ARM Y's 2.403 um median, and between the
  eligible median and ARM X's render median: does the cheaper array rank checkpoints the
  way the eligible one does?
- Z4: the best single checkpoint and the averaged finals on the eligible median.
No bar and no verdict: this informs which checkpoint a hunter should run on eligible data.
