# PREREGISTRATION: Nieuwlaar PHerc. 343 First Letters image, reproduced on Apple Silicon (NIEUWLAAR_MAC)

Written 2026-10-12, before any ink model was run on this render and before G1 or G2 was evaluated. The 66-layer
render was started (download only) while this was written; no layer of it was inspected. Its sha256 is in
`prereg.sha256`. Changes after that are dated amendments appended at the end, never edits above them.

## Question

Can his published image `outputs/concat_w047-w048_R5B2_z9500-11000.png` (sha256 59307c84...) be reproduced on an
M1 Max (64 GB, MPS) with no NVIDIA GPU and no VC3D, using his MIT code at commit
`e8370f1f88a211166056df62ea6fcfffe4a1063e` (clone: `results/nieuwlaar/repo`)?

## What the image depends on (read from the code, not assumed)

From `make_submission_image.py` (main, lines ~1560-1701) and `ensemble_maps.py`:

1. Ink: the four READING-side (`_reverse`) maps at d = +1 only: `dnative L66s26`, `hybrid_3d2d-seed42 L66s26`,
   `hybrid_3d2d-seed43 L66s26`, `hecate L66zc33.50`. Forward (blind) maps never enter the image or the statistics.
2. Statistics from `--ink-stats` (ensemble_maps.py): per model the pooled mean and SD of the reverse maps over all
   `--stat-depths` -15..15 on valid pixels; the ensemble stretch p1..p99.5 of the mean z-score on the fixed grid
   (every 8th row and column from 4) at every depth where all four models exist. Per-model p1/p99.5 do not enter the
   ensemble image.
3. Each map is the tile-phase merge (`merge_tile_phase.py`): w047 pixels (node source 1 or 4) from the zero-padded
   render (pad 40 rows, 200 columns), all others from the plain render. So both tilings are needed at every depth.
4. CT: layer 34 (`--ct-layer d+1`), stretch p1..p99.5 of valid pixels of layers 18..48 (his sidecar: 26, 164).
5. Overlay (`dimmed-ct`): ct_u8 = trunc(255 clip((L - lo)/(hi - lo), 0, 1)) in float32, 0 on invalid pixels;
   u = ink_u8/255, opacity a = 0.9 clip((u - 0.35)/0.40, 0, 1); pixel = rint(ct_u8 * 0.45 * (1 - a) + c * a),
   c = (R 255, G 235, B 120), float32, invalid pixels 0, reading orientation = raw[::-1, ::-1] (flip_lr and flip_ud).
   Consequence used below: where a = 0 the pixel is grey R = G = B = rint(float32(ct_u8) * float32(0.45)), and
   everywhere R - B = 135 a exactly before rounding, so a is recoverable from any image as a = (R - B)/135.
6. Data panel in the PNG: x 173..5814, y 182..2637 (5641 x 2455, 1 px = 1 render px); raw area rows 105..2560,
   columns 380..6021 of the 2560 x 6720 render. Row lines (cyan, 2 px, antialiased) are drawn inside the panel.

**Minimal faithful set (the PRIMARY run set):** reverse maps only, 4 models x 31 depths (-15..+15) x 2 tilings =
**248 inference runs**. This is exactly what the image and its statistics read; he computed 2 x 496 maps because
`--direction both` also makes the blind side. Nothing smaller reproduces the statistics faithfully.

## Pipeline and declared deviations

Same scripts, same flags (`--overlap 0.5 --blend-mode hann --batch-size 4`, layers s..s+16 with s = d + 25; Hecate:
`hecate_prep.py --native-um 8.64 --native-um-z 8.64 --target-um 9.6 --nz 16 --z-center 32.5+d`, then
`hecate.py --spacing-um 9.6 --batch-size 4 --reverse`, fp32). Deviations, each declared now:

- D1 render: our Python renderer (`ops/nieuwlaar/render_343.py` on `ops/render/render_tifxyz.py`), not VC3D 41701aa.
  Convention taken from his `render_66_layers.sh` and sidecar: layer k = p + (k - 32.5) n,
  n = normalise(dP/dcolumn x dP/drow), nearest-vertex normal, identity in-plane. Gated by G2.
- D2 device: MPS. villa's autocast is CUDA-only, so villa runs fp32 on MPS; Hecate fp32 (its default). His maps
  were CUDA (RTX 4090/5090).
- D3 software: villa ink CLI from `villa-inkcopy` (descendant of his villa 0fb38c45; the ink-inference diff is device
  handling only: accelerator selection incl. MPS, a compile-failure eager fallback, non_blocking only on CUDA) under
  `scratch_envs/env212` (torch 2.12.1, the vesuvius lock); Hecate under `env` (torch 2.14.0). Python 3.11.
- D4 dnative: rebuilt from the public safetensors and config (`dense_native-016000.rebuilt.pth`), as his
  fetch_checkpoints.sh does; the exact training checkpoint is not public (he reports r >= 0.99999 between the two).
- D5 reverse only (see above). Unit grouping and concurrency may be chosen for speed; they change no map.

Precondition already measured (no model involved): seed42, seed43, Hecate weights and hecate.py match
`ink/models.tsv` sha256; the public dnative weights and config match theirs (`models/nieuwlaar_ckpt/fetch_record.json`).

## G1 mesh gate

Run `mesh/build_concat.py` exactly as in reproduce.sh step 1. PASS iff x.tif, y.tif and z.tif are sha256-equal to
`outputs/SHA256SUMS` (8f31c8d7..., 7d416197..., 04feacf2...). meta.json equality is reported, not gated. FAIL stops
everything until explained.

## G2 render gate (defined from his published PNG)

Ours: L = our layer 34 over the raw area, V = all 66 layers > 0 there. Predicted grey
P = rint(float32(ct_u8(L; 26, 164)) * float32(0.45)) with ct_u8 = 0 off V, both mapped to reading orientation.
His: the PNG data panel. Comparison set G = V and (his R = G = B) and not (row lines drawn from the sidecar's
`rows_area_px` with his `draw_rows`, aa off, 2 px, dilated by 2 px).

- **PASS iff f34 = share of G with |his R - P| <= 1 is >= 0.99, and |G| >= 0.5 |V|.**
- Discrimination control: f for layers 33 and 35 (d = 0, +2) must each be < 0.99; otherwise G2 does not
  discriminate one layer and is INCONCLUSIVE. f for layer 31 (= the opposite normal sign) is reported; if it beats
  f34 the sign is wrong.
- Reported, not gated: sha256 of our layer-34 area bytes vs his df1a3e8d..., sha256(np.packbits(V)) vs his f6054921...
- If G2 fails: diagnose orientation, layer convention (sign, half-layer centre), normal mode, interpolation and
  chunk alignment before any ink run. No ink run is queued on a failed G2.

## Primary endpoint (arm P: full 248-run set, his statistics recomputed by us)

Our image is made with his exact `make_submission_image.py` command (reproduce.sh step 5) from our merged maps and
our `ensemble_maps.py --stat-depths -15..15` statistics. Comparison set C = our valid area pixels minus the dilated
row mask above; a = (R - B)/135 per image; boxes = the 18 `letters_area_px` boxes of his sidecar.

**REPRODUCED iff all four hold:**
- P-a: Pearson r(a_ours, a_his) over C >= 0.98.
- P-b: Pearson r(a_ours, a_his) over C inside the union of the 18 letter boxes >= 0.98.
- P-c: ink above threshold (R - B >= 1) per box: in at least 16 of 18 boxes 0.8 <= f_ours / f_his <= 1.25
  (a box with f_his < 0.02 counts as agreeing when f_ours < 0.05).
- P-d: Pearson r of luminance (0.299 R + 0.587 G + 0.114 B) over C >= 0.98.

Depth control (required for the verdict to stand): the same P-a statistic for our images from the d = 0 and d = +2
maps (same statistics, same command) must both be below the d = +1 value; otherwise the endpoint is INCONCLUSIVE.
Reported, not gated: our statistics (per-model mean and SD, ensemble p1/p99.5, CT p1/p99.5) against his sidecar.
Any primary failure is reported as a failure; no threshold moves after data.

## Secondary arms (never replace the primary)

- S2 (maps only, first readout, 8 runs): our d = +1 maps (4 models x 2 tilings) with HIS published statistics
  (sidecar `ink.normalisation`: per-model mean and SD, ensemble p1 -0.7182422280311584, p99.5 3.862934286594383; CT
  26, 164). Same endpoint and thresholds. Isolates the maps from the statistics.
- S1 (reduced statistics, 56 runs): statistics from d = -3..+3 only. Same endpoint and thresholds.

## Run plan

Resumable per-unit runner (`ops/nieuwlaar/run_unit.py`): one unit = (tiling, model, d), reverse only; skips a unit
whose output exists; writes into a private temp dir and renames into place only when complete. Order: d = +1 first,
then 0, -1, +2, -2, ..., +15, -15; within a depth the four models, plain then padded. Finalize units (merge,
statistics, image, comparison) for S2 after the first 8 units, S1 after 56, P after all 248. Long compute goes only
through `~/bin/nightqueue` (power-gated), never detached by hand. Timing of one villa run and one Hecate run at
d = +1 on the full plain render is measured in this session, and the ETA is derived from those timings.

## Amendment 1, 2026-10-12 13:38 (after G1, G2 and the two timing runs; no endpoint, threshold or arm changed)

- G1 PASS and G2 PASS before any ink run (numbers in `g2.json`, `g2_extra.json`, `logs/g1_build_concat.log`).
- Compatibility, no effect on any value: (1) his `hecate_prep.py` catches only AttributeError when zarr's
  `create_dataset` is unusable; under zarr 3.1.6 it raises TypeError, so `ops/nieuwlaar/hecate_prep_compat.py` is
  his file with that one `except` widened to (AttributeError, TypeError), which runs the same `create_array` branch
  his zarr 3.2.1 ran. (2) TMPDIR for the ink processes is a short /tmp path, because torch_shm_manager failed under a
  long one (macOS 104-byte socket path limit). (3) One queue line = one (tiling, d) with the four models in turn;
  run_unit.py still skips every finished map, so resume granularity stays one map. (4) run_unit.py refuses to start
  when the villa python cannot see MPS, so no unit can silently fall back to CPU.
- Plumbing test of finalize.py (arm S2 code path) with STAND-IN maps (the real seed42 map copied for dnative and
  seed43, real Hecate, padded copies made by zero padding): the pipeline ran end to end and the endpoint rejected the
  wrong model set (r_alpha 0.70). Those outputs were discarded; they are not a result of any arm.
