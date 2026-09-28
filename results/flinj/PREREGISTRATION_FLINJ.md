# ARM FLINJ, phase 1: can ink_9um draw a letter it is shown, on the eligible scan type?

Registered 2026-09-28, before any injected input exists and before any inference on one.

## Question

On the eligible 1.2 m scans, ink_9um finds ink on scrolls it never saw but draws it as blobs, never as letters
(villa #1867, #1898, #1907; Bullo27's 65-patch survey). Two explanations predict different things:

- **The model cannot draw strokes at this signal strength** (its receptive field and training make blobs whenever the
  ink signal is as weak as real ink is here). Then ink of a KNOWN letter shape, added to real eligible-type scans at the
  strength real ink has, also comes out as a blob.
- **The model can draw strokes, and real ink at 9 um is not stroke-shaped in the data** (surface misplacement, ink
  concentrated in spots, labels traced from 2.4 um maps). Then injected letters come out as strokes.

Injecting letters of known shape separates the two, and it also yields unlimited exact shape labels for phase 2.

## Data (all already on disk, nothing new is downloaded)

Hosts, 21-plane surface-volume crops through villa's own preparation:
- PHerc0841 9.366 um 113 keV: w00, ag144, ag174 (`data/zs/<seg>/w04/ct.zarr`, labels `data/zs/<seg>/label.npy`).
- PHerc0009B 8.64 um 116 keV and PHerc0500P2 9.362 um 113 keV (`data/f6/<seg>/N0/ct.zarr`, labels alongside).

Model: scrollprize/ink_9um hybrid_3d2d seed42 step-075000 (the primary checkpoint of every earlier arm), frozen,
direction chosen as in ARM Z / F6 (forward for all five, recorded there), through the same inference script.

## Injection

1. **Ink residual library, real ink only.** For every labelled ink pixel inside the support mask of a DONOR crop,
   the residual r(z, y, x) = I(z, y, x) minus b(z, y, x), where b is the mean over non-ink support pixels in an annulus
   of radius 16 to 48 px at the same plane. Residual patches of 32 x 32 px are cut where at least 90% of pixels are
   labelled ink. Donors never equal the host: PHerc0841 hosts take donors from the other two PHerc0841 crops;
   PHerc0500P2 (same 113 keV batch as PHerc0841's eligible scan) takes all three PHerc0841 crops; PHerc0009B (the only
   116 keV labels here) is split into left and right halves, each half hosting injections built from the other half.
2. **Glyphs.** Greek capital letters rasterised from a system font, skeletonised and re-drawn at a fixed physical
   stroke width, cap height 2.5 mm and stroke width 0.35 mm (primary), converted to pixels with each host's own
   spacing. Rows of 7 letters drawn at random from the 24, letter pitch 1.2 x cap height, row pitch 1.8 x cap
   height, rotation uniform in +/- 5 degrees, 64 px clear of crop borders. Glyph edges softened by a Gaussian of
   sigma 0.7 px (the measured in-plane blur of the eligible scan, `results/fs/blur_calibration.json`).
3. **Injection.** I' = clip(I + a x G(y, x) x R(z, y, x), 0, 255), where G is the soft glyph mask and R tiles
   residual patches drawn at random from the donor library. One fixed seed per host.
4. **Amplitude calibration.** a* is the amplitude at which the frozen model's AUC for injected stroke pixels against
   pixels farther than 2 cap heights from any glyph equals that host's REAL held-out AUC for the same checkpoint
   (ARM Z and F6 values), found on the grid a in {0.25, 0.5, 1, 2, 4, 8, 16} with linear interpolation in log a.

## Primary endpoint, fixed now

**Counter contrast C** on the difference map D = P(injected) minus P(same crop, no injection), which removes any
response to real ink already in the host:

    C = (mean D on strokes  minus  mean D on counters) / (mean D on strokes  minus  mean D on far background)

strokes = glyph pixels; counters = the glyph's convex hull minus the glyph dilated by half a stroke width; far
background = farther than 2 cap heights from every glyph. A blob fills its counters (C near 0); a drawn letter
leaves them empty (C near 1).

**Primary: the median over the 5 hosts of C at a*.**
- **STROKES** if the median C at a* is 0.5 or more.
- **BLOBS** if it is 0.2 or less.
- **MIXED** otherwise.

## Metric checks, run first, model-free (the thresholds are only meaningful if these pass)

- M1, ceiling: C of the soft glyph mask itself, and of the glyph mask blurred by the full measured eligible-scan
  blur, must each be 0.8 or more.
- M2, blob floor: C of the glyph mask blurred to letter scale (Gaussian sigma = cap height / 3) must be 0.2 or less.
- If M1 or M2 fails, the metric is redefined and this pre-registration is amended BEFORE any model output is scored.

## Controls

- C0, no injection: D is zero by construction; the model run twice on the same input must give max |D| = 0
  (determinism, as measured for this pipeline before).
- C1, depth-shuffled residual at a*: the planes of R permuted per patch. Reported with its AUC; if it is detected as
  well as the real residual, the model is responding to brightness, not to ink structure.
- C2, flat residual at a*: R replaced by its own per-patch mean over depth and space (a plain brightness bump).
- C3, positive control: at a = 16 x a*, C must reach 0.5 on at least 4 of 5 hosts, or the pipeline cannot show a
  letter even at a strength far beyond real ink and the primary is VOID.

## Secondary, descriptive

- S1: C on REAL held-out letters, with the same geometry rules applied to the labelled tracings (strokes = label,
  counters = label hull minus the dilated label), on P(no injection). This says whether real ink behaves like the
  injection.
- S2: the smallest amplitude, as a multiple of a*, at which C reaches 0.5.
- S3: the primary repeated with the 14-checkpoint mean.
- S4: stroke widths 0.25, 0.5 and 0.7 mm at a*.
- Every figure panel shows glyph mask, P(no injection), P(injected) and D side by side for the same pixels.

## Honest limits, stated before the result

The injection is a model of ink, not ink. Its residuals come from 5 labelled crops whose labels were traced from
2.4 um maps and are about 1 mm wide, so they carry both ink and some papyrus. A STROKES verdict says the model can
draw this injection, not that real ink at 9 um is findable as letters. A BLOBS verdict says the model blurs even a
known letter at realistic strength, which is the case for training on injected letters (phase 2).
