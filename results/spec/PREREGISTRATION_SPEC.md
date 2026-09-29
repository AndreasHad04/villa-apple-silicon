# ARM SPEC: does teaching ink_9um that depth-disordered input is not ink raise its real AUC on eligible-type scans?

Registered 2026-09-28, before any fine-tuned weights exist.

## Why

ARM FLINJ (results/flinj/FLINJ_REPORT.md) found that ink_9um keeps the shape of stroke-shaped real ink on 9.36 um /
113 keV scans (median counter contrast 0.8384), but the response is weak (stroke AUC 0.61 to 0.71 against a matched
non-ink transplant), and the same ink with its planes shuffled in depth is detected MORE (AUC 0.81 to 0.90). A model
that fires harder on depth-disordered texture than on ink will paint blobs wherever the surface is disordered. If that
response is part of what buries real letters, suppressing it should raise real-label AUC. If AUC does not move, depth
disorder is not what limits the model on real data, and that closes this lever.

## Data (on disk; NO labels are used for training)

Inputs of the four FLINJ hosts, 21 planes each: PHerc0841 9.366 um w00, ag144, ag174 (`data/zs/<seg>/w04/ct.zarr`)
and PHerc0500P2 9.362 um (`data/f6/p0500p2/N0/ct.zarr`), with each crop's support mask. Labels (`label.npy`) are read
only by the scorer. **Transductive pilot:** the fine-tune sees the scoring crops' inputs (never their labels). A
non-transductive replication (training inputs from other regions) is required before any claim beyond this pilot.

## Arms (same seed, same patch sequence, same schedule; they differ only in the 4 extra patches per step)

- **B**: released `ink_9um/hybrid_3d2d-seed42` step-075000, no training.
- **K (control)**: student initialised from B; each step 16 normal patches with the frozen teacher B's probability as
  a soft target, plus 4 more normal patches, also with teacher targets.
- **T (treatment)**: the same 16 normal patches with teacher targets, plus the same 4 extra patches DEPTH-DISORDERED
  with target 0 everywhere. Disorder, chosen per patch with equal probability: (i) one random permutation of the 17
  planes for the whole patch; (ii) an independent random permutation per 32 x 32 cell (as FLINJ C1).

Common to both: patches 17 x 128 x 128, uniform over the four crops, position uniform among positions whose centre pixel
is in support; z window = 17 consecutive planes starting at a uniform offset 0 to 4 (the recipe's +/-2 jitter around
[2, 19)); the teacher sees the identical window. Normalisation = villa's inference preprocessing for this checkpoint
(`tifxyz_robust`, `normalize_flat_patch`), per patch. Model wrapped in villa's `TargetModel` exactly as inference does.
Loss = BCE with logits over support pixels, each patch weighted equally. SGD momentum 0.9 Nesterov, lr 1e-3 (linear
warmup over 50 steps, then constant), weight decay 3e-5, grad clip 1.0, fp32, **1500 steps**, BatchNorm layers frozen
in eval mode (so K at step 0 equals B exactly), training seed 7, MPS.

## Evaluation (the ARM Z / FLINJ inference path, unchanged)

villa `infer.main` in process (`ops/flinj/common.infer`: forward, fp32, batch 32, no compile, 0 workers) on each crop's
input with each arm's weights; scored with `ops/ys_score.metrics(pred, label, support)`.

**Positive control of the path (required):** B must reproduce the stored real AUCs exactly (w00 0.7680, ag144 0.7599,
ag174 0.7614, p0500p2 0.7537, as printed in FLINJ_REPORT.md S7), else VOID.

**Manipulation check (required):** on each crop with all 21 planes globally permuted (fixed permutation, seed 11), the
fraction of support pixels with P > 0.5 must be lower for T than for K on at least 3 of 4 crops, else VOID (the
treatment did not take).

## Primary endpoint, fixed now

Delta = median over the 4 crops of AUC(T) minus AUC(K).
- **WIN** if Delta >= +0.010 AND AUC(T) > AUC(K) on at least 3 of 4 crops.
- **HARM** if Delta <= -0.010.
- **NULL** otherwise.

## Secondary, descriptive

- AUC(K) minus AUC(B): what self-distillation alone does (frozen BN, 1500 steps).
- Disorder types never trained on: planes reversed, planes rolled by half; P > 0.5 fraction per arm per crop.
- The FLINJ transplant pair (B1 / B2 inputs in `data/flinj/`) re-scored with T and K: counter contrast C, d, stroke AUC;
  and FLINJ C1 (shuffled-ink transplant) AUC, which should fall for T.
- Separation (label-free) and P > 0.5 fraction on the real inputs, per arm.

## Honest limits, stated before the result

Transductive, one seed, 1500 steps, four crops from two scrolls at one scan configuration (113 keV). A WIN says the
lever is real on these crops; it does not yet say it transfers to an unseen scroll or makes letters legible there.
