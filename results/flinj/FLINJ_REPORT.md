# ARM FLINJ phase 1 report (as amended)

Generated 2026-09-28T22:36:03 by ops/flinj/report.py (sha256 273c06dee9181f38). Pre-registration 73e70a5cb0fbb3d07db96e944475914bc788d4c36ff0cb99af81311ca84d1dd6; amendment 1 c4d455a9393f158b94dbf10705c1faa8b46c1f19d4abd6ef9f1b8767c43b5e51; amendment 2 f85930005a1bc207147f5f74f394335821f0898283cbf0e167ab8ff1429c487b; amendment 3 1d34cb73b9007142fd4d05e0f3c93916cfec6c51e6899be7673bec4faf93867f; amendment 4 9b48be610260a3e4b17e4ec35e7487e65b98b0dbd02377b108e8ab19aa3b1a75. Amendment 2 replaces the additive injection with a background-matched transplant of real ink (B1) against a matched non-ink transplant (B2), scores C on their difference with the B3 floor, retires the amplitude grid, keeps C3 with a 10% clip bound (B4) and adds S7. Amendment 3 uses one background (valid, not-labelled-ink annulus pixels) for donor and host (C1) and keeps rows more than 48 px from host ink (C2). Amendment 4 judges C3 on ceil(0.75 x included hosts) (D1) with no clip bound (D2), gives each S4 width its own layout (D3) and states the limit below (D4). Every number below is computed from the result files.

**Limit (D4):** excluded from the primary: p0009b (8.64 um, 116 keV). The primary speaks only for 9.362 um / 113 keV, 9.366 um / 113 keV scans.

## Status

| host | letters | units done | of |
|---|---|---|---|
| w00 | ΘΥΚΕ | 41 | 41 |
| ag144 | ΠΦΓΕΑΦ | 41 | 41 |
| ag174 | ΑΑΨΒΧ | 41 | 41 |
| p0009b |  | 6 | 6 |
| p0500p2 | ΣΨΩΩ | 41 | 41 |

## Model-free checks

M1 / M2: PASS: model runs may proceed.

| host | M1 soft | M1 sigma 0.99 | M2 | M2 0.25 mm | M2 0.5 mm | M2 0.7 mm |
|---|---|---|---|---|---|---|
| w00 | 1.0000 | 1.0000 | 0.0127 | -0.0158 | 0.1308 | 0.1547 |
| ag144 | 1.0000 | 1.0000 | 0.0622 | 0.0192 | 0.1989 | 0.3084 |
| ag174 | 1.0000 | 1.0000 | 0.1120 | 0.0671 | 0.2090 | 0.2319 |
| p0009b | C2: no row of 4 letters fits 64 px clear of borders and no-data and more than 48 px from host ink |  |  |  |  |  |
| p0500p2 | 1.0000 | 1.0000 | -0.0690 | -0.0542 | -0.0956 | -0.0416 |

b over 'valid_not_ink' annulus pixels for donor and host (C1); rows more than 48 px from host ink (C2); C3 clipped fraction reported without a bound (D2).

| host | letters 0.35 / 0.25 / 0.5 / 0.7 mm | b_host coverage | min distance to host ink px | clipped ink / non-ink transplant | C3 clipped | S5 clipped | S6 clipped |
|---|---|---|---|---|---|---|---|
| w00 | 4 / 4 / 4 / 4 | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 48.0 / 48.0 / 48.1 / 48.0 | 0.0172 / 0.0047 | 0.0682 | 0.0162 | 0.0002 |
| ag144 | 6 / 6 / 6 / 6 | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 48.1 / 48.1 / 48.1 / 48.1 | 0.0186 / 0.0062 | 0.0017 | 0.0122 | 0.0001 |
| ag174 | 5 / 5 / 5 / 5 | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 48.0 / 48.0 / 48.0 / 48.0 | 0.0230 / 0.0112 | 0.1155 | 0.0103 | 0.0009 |
| p0009b | none / none / 4 / 4 | 1.0000 / 1.0000 | 48.0 / 48.1 | not run / not run | not run | not run | not run |
| p0500p2 | 4 / 4 / 4 / 4 | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 48.1 / 48.1 / 48.0 / 48.3 | 0.0196 / 0.0068 | 0.0191 | 0.0106 | 0.0001 |

## Plumbing smoke pair (unregistered host background, outcome not computed)

ag174 whole crop: far-background max |D| = 0 over 2152774 px; inputs identical outside the glyph mask: yes; seconds per inference 36.6 to 40.9. Full-size C0 (smoke_full): max |D| = 0.

## C0, determinism

| host | max |D| run 1 vs run 2 | max |diff| vs stored ARM Z / F6 |
|---|---|---|
| w00 | 0 | 0 |
| ag144 | 0 | 0 |
| ag174 | 0 | 0 |
| p0009b | 0 | 0 |
| p0500p2 | 0 | 0 |

## Primary (B3): D = P(ink transplant) - P(non-ink transplant)

| host | C | C with the B3 floor | d (prob) | AUC ink vs non-ink on strokes | far max |D| | clipped ink | clipped non-ink |
|---|---|---|---|---|---|---|---|
| w00 | 0.7040 | 0.7040 | 0.1010 | 0.7059 | 0 | 0.0172 | 0.0047 |
| ag144 | 0.7348 | 0.7348 | 0.0686 | 0.6903 | 0 | 0.0186 | 0.0062 |
| ag174 | 0.9420 | 0.9420 | 0.0517 | 0.6205 | 0 | 0.0230 | 0.0112 |
| p0009b | excluded (C2) |  |  |  |  |  |  |
| p0500p2 | 1.2689 | 1.2689 | 0.0296 | 0.6149 | 0 | 0.0196 | 0.0068 |

- p0009b: excluded from the primary, C2: no row of 4 letters fits more than 48 px from host ink and 64 px clear of borders and no-data.
C3 positive control: C (with the B3 floor) >= 0.5 on 3 of 4 included hosts (D1 needs 3).
Median C over 4 hosts with a defined C: 0.8384. STROKES if >= 0.5, BLOBS if <= 0.2, MIXED otherwise.

**Primary verdict: STROKES.**

## S7, realism

| host | ink vs non-ink stroke AUC | real held-out AUC | difference |
|---|---|---|---|
| w00 | 0.7059 | 0.7680 | -0.0621 |
| ag144 | 0.6903 | 0.7599 | -0.0696 |
| ag174 | 0.6205 | 0.7614 | -0.1410 |
| p0009b | not run | 0.8150 |  |
| p0500p2 | 0.6149 | 0.7537 | -0.1389 |

## Controls

| host | C1 (A3) | C1 AUC | C2 (A3) | C2 AUC | C3 C (floor) | C3 AUC | C3 clipped | C3 far max |D| |
|---|---|---|---|---|---|---|---|---|
| w00 | 0.9888 | 0.8108 | UNDEFINED (C nan, d -0.0148, AUC 0.4258) | 0.4258 | 0.5732 | 0.5949 | 0.0682 | 0 |
| ag144 | 0.9183 | 0.8983 | UNDEFINED (C nan, d -0.0228, AUC 0.3968) | 0.3968 | UNDEFINED (C -1.2991, d 0.0069, AUC 0.5316) | 0.5316 | 0.0017 | 0 |
| ag174 | 1.1018 | 0.8116 | UNDEFINED (C nan, d -0.0435, AUC 0.3175) | 0.3175 | 1.2308 | 0.5716 | 0.1155 | 0 |
| p0009b | not run | not run | not run | not run | not run | not run | not run | not run |
| p0500p2 | 1.0168 | 0.8787 | UNDEFINED (C nan, d -0.0235, AUC 0.4557) | 0.4557 | 1.0293 | 0.6514 | 0.0191 | 0 |

## Secondary

S1, C on the REAL labelled letters, on P(host):

| host | C | mean P strokes | mean P counters | mean P far |
|---|---|---|---|---|
| w00 | nan | 122.93 | 86.84 | nan |
| ag144 | nan | 128.19 | 87.77 | nan |
| ag174 | nan | 115.66 | 77.77 | nan |
| p0009b | nan | 117.77 | 72.35 | nan |
| p0500p2 | nan | 116.22 | 87.21 | nan |

S3, 14-checkpoint mean on the transplant pair: w00 1.0253; ag144 0.9849; ag174 1.0154; p0009b not run; p0500p2 1.0277.

S4 on the transplant pair, each width on its own A1 + C2 layout (D3), each C beside its own M2 (A6: no verdict from S4):

| host | 0.25 mm C / M2 | 0.35 mm C / M2 | 0.5 mm C / M2 | 0.7 mm C / M2 |
|---|---|---|---|---|
| w00 | 0.6299 / -0.0158 | 0.7040 / 0.0127 | 1.2004 / 0.1308 | 1.5730 / 0.1547 |
| ag144 | 0.6322 / 0.0192 | 0.7348 / 0.0622 | 0.5716 / 0.1989 | 0.8989 / 0.3084 |
| ag174 | 0.5969 / 0.0671 | 0.9420 / 0.1120 | 0.8372 / 0.2090 | 0.9533 / 0.2319 |
| p0009b | no row of 4 fits under A1 and C2 at this width | excluded (C2) | UNDEFINED (C 2.0645, d 0.0141, AUC 0.5768) / -0.0614 | 1.2539 / -0.2220 |
| p0500p2 | 0.7490 / -0.0542 | 1.2689 / -0.0690 | 0.5349 / -0.0956 | 0.2507 / -0.0416 |

S5 (raw additive residual) and S6 (mean depth profile) at a = 0.5, against the host, A3 floor:

| host | S5 C | S5 AUC | S5 clipped | S6 C | S6 AUC | S6 clipped |
|---|---|---|---|---|---|---|
| w00 | UNDEFINED (C 1.9642, d 0.0119, AUC 0.5884) | 0.5884 | 0.0162 | UNDEFINED (C nan, d -0.0004, AUC 0.5025) | 0.5025 | 0.0002 |
| ag144 | UNDEFINED (C 1.8090, d 0.0146, AUC 0.4869) | 0.4869 | 0.0122 | UNDEFINED (C nan, d -0.0009, AUC 0.4454) | 0.4454 | 0.0001 |
| ag174 | UNDEFINED (C 0.4069, d 0.0136, AUC 0.5428) | 0.5428 | 0.0103 | UNDEFINED (C 2.2809, d 0.0011, AUC 0.4752) | 0.4752 | 0.0009 |
| p0009b | not run | not run | not run | not run | not run | not run |
| p0500p2 | UNDEFINED (C nan, d -0.0193, AUC 0.5252) | 0.5252 | 0.0106 | UNDEFINED (C 0.9305, d 0.0038, AUC 0.5147) | 0.5147 | 0.0001 |

## Decisions (fixed before any registered model output was scored)

- Amendment 1, A7: Arial skeletons; cap height = outer height of the redrawn Eta; rotation per letter; 32 px patch grid; PHerc0009B split at W // 2 with the donor half chosen per pixel; S1 hulls per connected label component; S4 keeps the layout.
- A1 as implemented fits 4 letters on w00, 6 on ag144, 7 on ag174, p0009b and p0500p2 (the first n of each host's 7 seeded draws).
- B1 / B2 patches: grid cells WHOLLY inside support with b defined at every pixel; B1 >= 90% labelled ink; B2 every pixel farther than 48 px from any label. The transplant carries r = I - b at every pixel of a patch (the donor column as it is).
- C1 / C2 kept from the registration and moved onto the transplant (the ink residual's planes permuted per patch / replaced by its patch mean), each scored against the B2 non-ink transplant.
- The B3 floor applied to C3 as a pair (C3 against the untouched host): d >= 0.02 and AUC of P(C3) against P(host) over strokes >= 0.55. S5 and S6 keep amendment 1's A3 floor (stroke-versus-far AUC).
- Verdict precedence: VOID (C3 below D1's ceil(0.75 x included hosts)) first, then NOT DETECTABLE (fewer than 3 defined), then the 0.5 / 0.2 thresholds.
- C2 read as MORE than 48 px from host ink, so no 16 to 48 px annulus (inclusive at 48) reaches it; that also satisfies 'at least 48 px'.
- C1 applies to the transplants (B1, B2) as it names them; S5 and S6 / C3 keep the registered residual library.
- D4: the C2-excluded host still runs base_r1 / base_r2 (C0, S1) and each S4 width whose own layout fits, with its own halves as donors.
- C3 scale by bisection to +20 grey levels mean change on stroke voxels; clipped fraction reported with no bound (D2).
- Clipped fraction: stroke pixels x all 21 planes whose rounded value fell outside [0, 255]. D5 counters over several glyphs: union of hulls minus the union of glyphs dilated by half a stroke. D6 masks restricted to pixels where the surface volume holds data.

Primary layouts: w00 4 letters ΘΥΚΕ at [590, 251]; ag144 6 letters ΠΦΓΕΑΦ at [597, 61]; ag174 5 letters ΑΑΨΒΧ at [642, 624]; p0009b excluded (C2); p0500p2 4 letters ΣΨΩΩ at [550, 1265].
S4 layouts (letters per width): w00 {'0.25': 'ΘΥΚΕ', '0.5': 'ΘΥΚΕ', '0.7': 'ΘΥΚΕ'}; ag144 {'0.25': 'ΠΦΓΕΑΦ', '0.5': 'ΠΦΓΕΑΦ', '0.7': 'ΠΦΓΕΑΦ'}; ag174 {'0.25': 'ΑΑΨΒΧ', '0.5': 'ΑΑΨΒΧ', '0.7': 'ΑΑΨΒΧ'}; p0009b {'0.25': None, '0.5': 'ΟΙΒΠ', '0.7': 'ΟΙΒΠ'}; p0500p2 {'0.25': 'ΣΨΩΩ', '0.5': 'ΣΨΩΩ', '0.7': 'ΣΨΩΩ'}.

## Figures

- figs/ag144_c3.png
- figs/ag144_primary.png
- figs/ag174_c3.png
- figs/ag174_primary.png
- figs/p0500p2_c3.png
- figs/p0500p2_primary.png
- figs/w00_c3.png
- figs/w00_primary.png
