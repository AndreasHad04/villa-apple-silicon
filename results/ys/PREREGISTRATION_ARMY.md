# ARM Y: is the depth direction a property of the segment or of the array, and does the input array explain ARM X's held-out gap?

Written 2026-09-23 13:4x local, BEFORE any ARM Y inference. Prompted by
liliandevarieux on villa #1867 (issuecomment-5793152131): PHerc0841 ag144 and
ag174 are "published with the depth axis reversed relative to what ink_detection
expects", and the choice of which 65 of the 109 layers is taken matters.

## What ARM X already measured (fixed, not re-run)

On the organisers' 65-plane 4.681 um renders in the Hugging Face label bucket
(`ink/841/...`), ink_9um prefers the REVERSED order on all three PHerc0841
segments, w00 included, for 15 of 15 checkpoint rows each (gap 0.07 to 0.21).
On PHerc0139 w016, built from a published surface volume through villa's
production recipe, the STORED order wins 15 of 15.

## Arrays and recipe

- Input: the PUBLISHED 2.403 um surface volumes,
  `s3://vesuvius-challenge-open-data/PHerc0841/segments/<seg>/surface-volumes/2.403um-0.22m-77keV-volume-20260319124803.zarr`,
  level 2 (9.612 um in-plane), 109 planes, for w00 (20260220213127),
  ag144 (20260220214732-auto_grown_20260220144552896) and ag174
  (20260221022814-auto_grown_20260220174252405).
- Labels: the published `ink-labels/2.403um-volume-20260319124803/20260918/`
  `inklabels.zarr` and `supervision.zarr`, level 2, same grid as the volume.
  Support = supervision nonzero; ink = inklabels nonzero inside support. If level
  2 labels are not binary, ink = at least half of the level-0 pixels in the 4x4
  block and support = all 16 supervised.
- Recipe: villa's production `prepare_9um_isotropic_input` (level2-zmean4-21slice-v1),
  called, never copied. Production window = its own `centered_slice(109, 84)`.
  Sweep windows by handing it planes [z0, z0+84) so its centred slice is the
  whole input; z0 in {1, 5, 9, 13, 17, 21, 25}.
- Inference: `vesuvius.ink_detection.inference.infer`, fp32 on MPS, both
  `--direction forward` and `reverse`, exactly as ARM X (ops/xs_infer.py).
- Scoring: ARM X's own functions from ops/xs_score.py (exact histogram AUC,
  64 px block bootstrap, label-free direction = larger separation).

## Checkpoints

Primary: seed42 step-075000 (ARM X's primary). Unit 3, after the primary
answers: all 14 checkpoints plus the averaged finals at the production window,
both directions.

## Controls (a failing control is reported, never dropped)

- C4 production equivalence: our z0 = 13 prepared input must equal
  `prepare_isotropic_input` run on the full 109-plane crop, bit for bit, or the
  sweep is void.
- C3 label alignment: the organisers' own published prediction for each segment
  (`ink-detection/...new_canon_autoresearch_recipe-tile256-stride128.tif`,
  2.403 um), block-mean downsampled 4x onto the level-2 grid, must score AUC at
  least 0.75 against the labels, or that segment is excluded as misaligned.
- C2 block-shuffle floor as in ARM X, reported with ARM X's bar (every seed
  within 0.03 of 0.5).

## Endpoints, fixed now

- Y1 DIRECTION. Per segment, the label-preferred direction at the production
  window, primary checkpoint, on the 2.403 um array. Decisive only when
  |AUC_forward - AUC_reverse| >= 0.05. "Direction is a property of the array,
  not the segment" is SUPPORTED if any segment is decisive on both arrays with
  opposite preferred directions; REFUTED if all three are decisive on both arrays
  with the same direction; otherwise UNDECIDED.
- Y2 ARRAY. PHerc0841 median AUC, label-free direction, production window,
  primary checkpoint, on the 2.403 um array, against ARM X's 0.6931 on the
  renders and C1's 0.7740 in distribution. At or above 0.7740: the ARM X gap is
  an input-array effect for this checkpoint. Within 0.03 of 0.6931: the gap does
  not depend on which organiser array is used. Otherwise reported as measured.
- Y3 SLICE. Per segment, the best window's AUC minus the production window's,
  label-free direction, primary checkpoint. "Slice matters on this array" if the
  gain is at least 0.03 on at least 2 of 3 segments. The label-oracle best is
  reported beside it and never used for the verdict.

## Not claimed in advance

Nothing here tests liliandevarieux's own model. Their arrays are inferred from
"65 of the 109 layers"; if they used another array, Y1 compares ours only.
