# ARM FLINJ, amendment 3: one background definition for donor and host, and rows kept clear of the host's own ink

Written after amendment 2 was implemented and before any C or AUC under it was computed (the smoke pair's C and AUC were
deliberately not computed). Measured: B1's b_host, defined over the host's non-ink SUPPORT pixels, exists at only 16 to
80% of glyph pixels (w00 28.0%, ag144 44.1%, ag174 16.3%, p0009b 79.6%, p0500p2 38.1%), because A1 places rows mostly
outside the labelled support.

## C1. One background definition, used identically for donor and host

b(z, y, x) = the mean over pixels in the 16 to 48 px annulus that are valid (inside the scan, not no-data) and not
labelled ink, at the same plane, whether or not they are in the supervision support. Used for b_donor AND b_host, and
for the ink transplant (B1) AND the non-ink transplant (B2). Any bias from unlabelled ink in an annulus is then the same
in both transplants and cancels in D.

## C2. Rows clear of the host's own labelled ink

A1's placement gains one condition: every glyph pixel is at least 48 px from any pixel the host labels as ink, so no
annulus touches the host's traced letters and no transplant overwrites them. Same pitch, same cap of 7 letters and
floor of 4; a host where no row of 4 fits is left out of the primary with the reason stated.

Everything else in the registration and amendments 1 and 2 stands.
