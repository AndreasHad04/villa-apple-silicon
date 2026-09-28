# ARM FLINJ, amendment 4: the positive control and the stroke-width arm under amendment 3's geometry

Written after amendment 3 was implemented and before any C or AUC under it was computed. Measured under C2: p0009b fits
no row of 4 letters 48 px clear of its own labelled ink and is excluded (layouts: w00 4 letters, ag144 6, ag174 5,
p0500p2 4); on ag174's new layout C3 clips 11.55% of stroke voxels, over B4's 10%; S4 at 0.7 mm on the primary layout
brings glyph pixels to 43.5 px (ag174) and 43.6 px (p0500p2) from host ink, breaking C2.

## D1. C3 is judged against the hosts that exist

The primary is VOID unless C3 reaches C of 0.5 or more (with the B3 floor) on at least three quarters of the included
hosts, rounded up: 3 of the 4 hosts here. The primary still needs at least 3 hosts with a defined C.

## D2. No clipping bound for the positive control

C3 is a check that the pipeline can show a letter at all, not a realistic input, so its +20 grey level mean change is
kept on every host and the clipped fraction is reported beside its C, without a bound. The realism of the primary is
carried by B1 and B2 and checked by S7, not by C3.

## D3. The stroke-width arm gets its own layouts

S4 widths are laid out with the same A1 and C2 rules as the primary, per width; a width where no row of 4 fits on a host
is skipped there, with the reason stated. S4 stays descriptive, with M2 reported beside every C (A6).

## D4. A limit this creates, stated now

With p0009b excluded, no 8.64 um, 116 keV host is in the primary; the result speaks for the 9.36 um, 113 keV
configuration (PHerc0841, PHerc0500P2) only. p0009b still serves as a donor for its own halves in the secondaries it can
run.

Everything else in the registration and amendments 1 to 3 stands.
