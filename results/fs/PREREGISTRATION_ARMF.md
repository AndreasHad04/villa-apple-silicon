# ARM F: can a fixed, label-free input transform recover what ink_9um loses on the eligible 1.2 m scans?

Written 2026-09-25 23:1x local (typo "23:5x" corrected 23:2x, before any result), after the label-free characterisation in
`results/fs/profile_*.json` (ops/fs_profile.py, no labels and no model output read)
and BEFORE any inference on a transformed input.

## Why

ARM Y and ARM Z scored ink_9um (primary checkpoint seed42 step-075000, label-free
direction) on the same three PHerc0841 segments in two array families: the 2.403 um
0.22 m 77 keV surface volume pooled by villa's production recipe (median 0.8294), and
the 9.366 um 1.2 m 113 keV surface volume (median 0.7614). Every First Letters
eligible volume is a 1.2 m scan: 12 of 22 are 9.362 um 113 keV from the same
2025-08-21 scan batch as PHerc0841's, 10 are 8.640 um 116 keV
(`s3://vesuvius-challenge-open-data/metadata.json`, read 2026-09-25).

## What the characterisation found (no labels)

- Depth geometry is NOT the difference: per 128 px tile, the native depth profile
  matches the 2.403 um profile at a shift of +3.5 to +5.5 um median (p10 to p90 within
  -7.5 to +12.1 um), profile correlation 0.989 to 0.993. Under half a native layer.
- Resolution IS: relative to the pooled input, the native window mean has 1.14 to 1.40x
  the radial power at mid frequencies and 0.17 to 0.30x at the highest, consistently on
  all three segments; adjacent native layers correlate 0.79 to 0.84 against 0.65 to 0.74
  pooled (more depth blur).
- Intensity shape differs at the dark tail (robust quantile 1%: -1.33 to -1.77 native
  against -0.97 to -1.43 pooled).

## Transforms, fitted WITHOUT labels, leave one segment out

For a segment X, every parameter below is estimated from the OTHER two segments'
paired production-window inputs (native w04 against pooled w13 resampled onto the
native crop with ARM Z's offset), never from X, and never from labels.

- SPEC (in plane): per layer 2D filter with radial amplitude
  G(f) = sqrt(P_pooled(f) / P_native(f)), f in cycles per micrometre, power spectra
  averaged over layers on the largest in-support central box, radially smoothed,
  G capped to [1/3, 3], G(0) = 1. Applied by FFT with reflect padding.
- DEPTH: the same construction along z (21 layers, per-column 1D spectra averaged).
- HIST: after any filter, a monotone quantile map (1001 levels) from the transformed
  native window values to the pooled window values, both within support.
- Output clipped to [0, 255], rounded, uint8, 21 layers, same crop and chunks as ARM Z.

## Conditions (primary checkpoint, production window, both directions, label-free choice)

- N0: native, unmodified (C0 control: must reproduce ARM Z's stored AUCs exactly).
- NS: SPEC. NH: HIST. NSH: SPEC then HIST. NSDH: SPEC, DEPTH, then HIST.
- NINV (dose control): 1/G applied to native, then no HIST.
- P0: pooled, unmodified (must reproduce ARM Y's production AUCs exactly).
- PDEG (causal test): 1/G applied in plane to the pooled input (its own pixel scale),
  then HIST mapped onto the NATIVE distribution.

Scoring: ARM Y's `metrics` and `finish` (ops/ys_score.py), labels and support of ARM Z
for N conditions and of ARM Y for P conditions.

## Controls, fixed now

- C0: N0 and P0 reproduce the stored ARM Z and ARM Y primary AUCs within 1e-9.
- C-ID: G = 1 with an identity map through the same code path writes an input
  byte-identical to N0 (checked on arrays, no inference).
- C-DOSE: NINV must not raise the chosen AUC by more than 0.005 on any segment; if it
  does, the sharpening premise is wrong and F1 is reported as uninterpretable.
- The shuffled-label floor (ARM X C2) is recomputed for the best N condition.

## Endpoints, fixed now

- **F1 (primary): median over the three segments of AUC(NSH) minus AUC(N0).**
  >= +0.020: "the family gap is partly repairable with a fixed label-free transform".
  <= +0.005: "not repairable this way". Otherwise "small, as measured".
  Also reported per segment as the fraction of the gap closed,
  (NSH - N0) / (P0 - N0).
- F2 (causal): median of AUC(P0) minus AUC(PDEG). >= half the median gap
  (0.5 x (median P0 - median N0), read from the JSON): "the spectral and intensity
  difference is sufficient to explain at least half the gap".
- F3 (attribution, descriptive): NS, NH and NSDH against N0.
- F4 (descriptive, run only if F1 >= +0.020): all 14 checkpoints plus the averaged
  finals on N0 against the best N condition; count of the 42 pairs improved.

## Honest limits, stated before the result

Three segments of one scroll. The transform is fitted on two and tested on the third,
so segment-level leakage is excluded but scroll-level generality is not tested here.
The 8.640 um 116 keV configuration is not represented in any labelled pair.

## Amendment 2026-09-25 23:2x, BEFORE any F1 number and before any PHerc0139 data is fetched: F5, a second scroll

- Segments: PHerc0139 `20250108000005-w030` and `20260126000000-w045` (in no ink_9um
  training representation, pooled or native) and `20260112000000-w043` (its pooled render
  is a training representation, its native render is not; reported separately).
- Arrays: `9.362um-1.2m-113keV-volume-20250728140407` level 0, centred 21 layers (N), and
  `2.399um-0.22m-78keV-volume-20260102150214` level 2 through villa's production 9 um
  recipe (P), labels `ink-labels/2.399um-volume-20260102150214/20260918` level 2, moved to
  the native canvas by ARM Z's Z-ALIGN rule.
- Transform: fitted on ALL THREE PHerc0841 pairs (no PHerc0139 data), applied unchanged.
- Conditions: N0, NSH, P0. Primary checkpoint, both directions, label-free choice.
- F5 endpoint: median over w030 and w045 of AUC(NSH) minus AUC(N0). Same bars as F1
  (>= +0.020 transfers; <= +0.005 does not). w043 descriptive.
- Run whether or not F1 passes.
- Z-ALIGN search radius for F5 is 32 px instead of 8: PHerc0139's native canvas exceeds the
  pure voxel-ratio size by 19 to 27 px (7140 x 7460 against 6940 x 7260 at 9.596 / 9.362),
  so an 8 px search could not contain the offset. The accept/exclude rule is unchanged.
  Fixed 23:3x from array SHAPES only, before any F5 pixel or label value was read.

## Amendment 2, 2026-09-25 23:4x, AFTER F1 to F3 were read (F1 failed: median -0.0018) and BEFORE any F5 inference

F3 found NSDH (in-plane SPEC, DEPTH, HIST) ahead of N0 on all three PHerc0841 segments
(+0.068, +0.017, +0.009, median +0.017). That is the best of five conditions chosen after
seeing them, so it is a hypothesis, not a result. It gets a fresh test:

- **F5b (confirmatory for NSDH): median over PHerc0139 w030 and w045 of AUC(NSDH) minus
  AUC(N0)**, with the in-plane and depth gains and the intensity map fitted on all three
  PHerc0841 pairs only. Bars as F1. w043 descriptive.
- Exploratory on PHerc0841, labelled exploratory wherever reported: ND (DEPTH only),
  NDH (DEPTH then HIST), PDEGZ (the pooled input degraded in plane AND in depth by the
  inverse gains, then mapped onto the native distribution), which asks whether depth
  resolution explains the part of the gap that in-plane degradation (F2) did not.

## Amendment 3, 2026-09-25 23:5x, AFTER the exploratory PHerc0841 conditions were read and BEFORE any F5 inference

Exploratory on PHerc0841: ND +0.029/+0.020/+0.022, NDH +0.044/+0.024/+0.029 (median
+0.029, all three positive), PDEGZ drops P0 by 0.033/0.034/0.045 against PDEG's
0.010/0.011/0.008: depth resolution, not in-plane resolution, carries the larger share.
NDH was chosen after seeing these, so it gets the same fresh test as NSDH:

- **F5c (confirmatory for NDH): median over PHerc0139 w030 and w045 of AUC(NDH) minus
  AUC(N0)**, depth gain and intensity map fitted on all three PHerc0841 pairs only. Bars
  as F1. Two confirmatory tests (F5b, F5c) now share these segments; both are reported,
  and a pass is quoted with that multiplicity stated.
- Sensitivity, not an endpoint: PHerc0139's canvases carry a known 0.22% voxel-size
  discrepancy (villa #1381) and w030's four quadrant peaks disagree by up to 10 px, so the
  F5 endpoints are also reported with labels resampled by an affine (per-axis scale and
  offset) fitted to the quadrant peaks. The pre-registered pure-shift labels stay primary.

## Amendment 4, 2026-09-26 00:3x, BEFORE any F6 data is fetched: F6, two more held-out scrolls, one in the other eligible configuration

Why, stated honestly: after w030 and w043 of F5 were read (w045 not yet), I checked the
training manifest again. PHerc0139's native scan supplied 5 of ink_9um's 29 training
representations, so F5 tests a scan the model has adapted to; PHerc0841's scan, and every
eligible scan, supplied none. F5 stands as pre-registered and is reported in full. F6 adds
the only other labelled scrolls with a native render that are in no training
representation at all (metadata.json, read 2026-09-26):

- PHerc0009B `20250919125754-auto_grown_20250919055754487_inp_hr`: native
  `8.64um-1.2m-116keV-volume-20250521125136` (31 layers, centred 21 = [5, 26)), the
  8.640 um 116 keV configuration of 10 of the 22 eligible volumes; labels
  `ink-labels/2.401um-volume-20250820154339/20260918`.
- PHerc0500P2 `20250825181859--1`: native `9.362um-1.2m-113keV-volume-20250820143440`
  (28 layers, [4, 25)); labels `ink-labels/2.215um-volume-20250526151718/20260918`.
- Labels at level 2 of their own grid, moved onto the native canvas by the voxel ratio and
  ARM Z's Z-ALIGN rule (search 32 px), the reference image being the mean of the central 21
  planes of the fine surface volume at level 2 over the labelled box.
- Conditions: N0, NDH and ND exactly as `bench/depth_sharpen_9um.py` computes them, with
  `--layer-um` set to the native layer spacing (8.64 or 9.362). Primary checkpoint, both
  directions, label-free choice.
- **F6: mean over the two scrolls of AUC(NDH) minus AUC(N0).** Bars as F1. Each scroll is
  also reported alone; a scroll excluded by Z-ALIGN is reported as excluded.
