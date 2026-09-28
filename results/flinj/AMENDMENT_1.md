# ARM FLINJ, amendment 1

Written after the implementation and a plumbing smoke run on a 768 px sub-crop with 72 px letters (too small to say
anything about the question), and BEFORE any full-size injected input was scored. Every change below answers a defect
measured in that build, not a result.

## A1. Layout (the registered row of 7 letters does not fit w00 or ag144)

One row per host, along x as in the real text, letters upright, as many letters as fit at the registered pitch, at most
7 and at least 4 (w00 fits 4, ag144 fits 5), placed at the first position 64 px clear of the borders and of no-data
pixels. One row leaves far background on every host.

## A2. Residual: remove the donor's voxel noise (primary), keep the registered raw residual as a secondary

Measured: r = I minus b carries the donor's full voxel noise (mean +33, std 40 grey levels on stroke cores at a = 1),
so a raw injection also RAISES local noise, a texture change that is not ink. **Primary injection is now the residual
patch smoothed in plane by a Gaussian of sigma 1.5 px (depth untouched)**, which keeps texture at the 4 px scale and up
against 37 px strokes. Secondaries at a* of their own: S5 the registered raw residual; S6 the donor's mean depth
profile only (no in-plane texture).

## A3. A floor on the stroke response before C means anything

Measured: at a = 1 the strokes moved +1.40 and the counters -3.87 grey levels, giving C = 3.77 from a near-null
response. **C is UNDEFINED unless (mean D on strokes minus mean D on far background) is at least 0.02 in probability
(about 5 grey levels) AND the stroke-versus-far-background AUC on P(injected) is at least 0.55.** An undefined C counts
as neither STROKES nor BLOBS.

## A4. When the amplitude grid does not reach a host's real AUC

The grid is extended by doubling up to a = 64. A host whose stroke AUC never reaches its real held-out AUC has no a*
and is left out of the primary with the reason stated. **If fewer than 3 of the 5 hosts have an a*, the primary verdict
is NOT DETECTABLE AT REALISTIC STRENGTH**, reported as a finding (the model does not see this injection as it sees
real ink), not as STROKES or BLOBS.

## A5. Clipping, and the positive control redefined

Measured: at a = 16, 85% of stroke voxels clip, so the top of the grid tests clipping. Every unit now reports the
fraction of stroke voxels clipped; units above 5% are flagged and excluded from a*. **C3 becomes a clean positive
control:** the donor's mean depth profile times the soft glyph mask, scaled so the mean change on stroke voxels is +20
grey levels with at most 1% clipped. The primary is VOID unless C (with the A3 floor) reaches 0.5 on at least 4 of 5
hosts in C3.

## A6. Stroke widths with no metric headroom

At 0.7 mm the model-free blob floor M2 is 0.28 (w00) and 0.25 (ag144), above 0.2, so the 0.2 threshold cannot tell a
blob from a wide stroke there. S4 at 0.7 mm is reported with its own M2 beside every C, and no verdict is drawn from it.

## A7. Decisions the registration left open, fixed by the implementation before any injected input existed

Arial glyph skeletons (serif skeletons grow spurs); cap height = outer height of the redrawn Eta; rotation per letter;
residual patches from a non-overlapping 32 px grid, patches touching pixels where b is undefined (deep inside wide
traced strokes, 0.3 to 2.2% of ink pixels) dropped, non-ink pixels of kept patches set to r = 0; PHerc0009B split at
W // 2 with the donor half chosen per pixel; the calibration AUC computed on P(injected); S1 hulls per connected label
component; S3 reuses the primary checkpoint's a*; S4 keeps the primary layout.

Everything else in the registration stands, including the thresholds 0.5 and 0.2 and the primary being the median C
over hosts at a*.
