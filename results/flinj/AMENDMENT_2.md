# ARM FLINJ, amendment 2: transplant real ink instead of adding it

Written after amendment 1 was implemented and measured without any scored model output: a model-free precheck
(`results/flinj/a5_precheck.json`) and one full-size smoke unit on ag174 whose C came out UNDEFINED under A3.

## Why the additive injection cannot answer the question

- Measured: the donor residual r = I minus b has a standard deviation of 41 to 44 grey levels, 25 to 40 times its mean;
  in-plane smoothing at 1.5 px only lowers it to 37 to 40, so the variance is papyrus texture at scales above 4 px,
  not voxel noise. The ink signal at 9 um is a small shift buried in that texture.
- Adding r therefore adds texture on top of the host's own and clips: keeping clipping at or under 5% leaves only
  a in {0.25, 0.5} on three hosts and {0.25, 0.5, 1} on two, and at a = 0.5 on ag174 the stroke AUC was 0.4788
  against the host's real 0.7614.
- The clean positive control of A5 (+20 grey levels, at most 1% clipped) clips 5.91% (w00), 9.06% (ag174), 1.41%
  (p0009b), 1.68% (p0500p2); only ag144 meets it.

## B1. Primary injection: background-matched transplant of real ink columns

Under the glyph mask, the host's voxel column is REPLACED, not added to:
I'(z, y, x) = b_host(z, y, x) + [I_donor(z, y', x') minus b_donor(z, y', x')], with (y', x') taken from spatially
coherent 32 x 32 donor patches of labelled ink (at least 90% ink, inside support), tiled under the glyph, b from the
same 16 to 48 px annulus of non-ink support pixels as before. The glyph mask is hard (the transplant itself carries the
scan's blur). Donor and host pairing rules are unchanged. There is no amplitude: this is real ink at real strength.

## B2. Matched control: the same transplant of non-ink papyrus

The identical transplant, same glyph mask, same patch grid logic, but from donor patches with at least 90% NON-ink,
inside support, farther than 48 px from any label. Both transplants replace the host's texture with a donor's and share
every seam; only the donor's ink differs.

## B3. Primary endpoint on the difference of the two transplants

D = P(ink transplant) minus P(non-ink transplant). Counter contrast C on D exactly as registered (strokes, counters,
far background), with the A3 floor restated for this design: C is UNDEFINED unless (mean D on strokes minus mean D on
far background) is at least 0.02 AND the AUC of P(ink transplant) against P(non-ink transplant) over stroke pixels is
at least 0.55. Primary = median C over hosts with a defined C: STROKES at 0.5 or more, BLOBS at 0.2 or less, MIXED
between; fewer than 3 hosts defined gives NOT DETECTABLE (the model does not respond to real ink in letter shape above
the floor). A4's amplitude grid and a* are retired; the far background of both runs is the untouched host, so D there
must be exactly 0 (checked).

## B4. Positive control, and realism

- C3 stays the clean signal of A5 (donor mean depth profile times the glyph mask, +20 grey levels mean change) with the
  clipping bound relaxed to 10% for this control only, scored on P(C3) minus P(host). VOID unless C reaches 0.5 (with
  the floor) on at least 4 of 5 hosts.
- S7, realism: the ink-versus-non-ink stroke AUC from B3 beside each host's real held-out AUC. Much lower means the
  transplant loses what the model uses on real ink, and any verdict carries that caveat.
- Kept as descriptive secondaries: S5 (raw additive residual) and S6 (mean profile) at a = 0.5 only; S3 (14-checkpoint
  mean) and S4 (stroke widths) on the B1 transplant.

Everything else in the registration and amendment 1 that this does not replace stands, including the 0.5 and 0.2
thresholds and A1's layout.
