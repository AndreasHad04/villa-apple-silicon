Two things, one about which PR should carry this and one measurement that answers the half layer question in #1765.

**On #1766 being a draft.** That changes my position, so I am stating the new one plainly rather than leaving the old one standing. My comment above said I was happy for this PR to be closed in favour of #1766. I wrote that believing #1766 was the merge candidate. A draft cannot be merged, so as things stand this PR is the only one of the two that a maintainer can act on today. I am therefore withdrawing the offer to close it and leaving it open and ready. If #1766 comes out of draft I have no objection to it being the one that merges and this one being closed; the point is the README line, not whose PR fixes it.

Priority is unchanged and I want it on the record in this comment rather than only in an earlier one: flummoxjr published the centred window with r = 0.9999997 on 2026-09-03, @kadenpool filed #1765 and #1766 on 2026-09-11, and this PR is from 2026-09-16 and was reached independently without knowledge of either.

**On 23 against 24, and first a correction to my own evidence.** #1765 now records that 23 beats 24 on all 8 text windows there, median match 0.905 against 0.874. I went back to check what my own sweeps actually said about that, and they said nothing: of the 15 sweep grids behind this PR, **14 contained `START_LAYER=24` and not 23, and the one that contained 23 did not contain 24**. So "best `START_LAYER` 21 to 25" in the PR body never separated the two, and any impression that my data preferred 24 was a property of the grid I chose, not a measurement.

So I ran the paired comparison: the same 15 crops, 4 scrolls, each crop scored at 23 and at 24 and nothing else changed. 1024 px crops, `TILE_SIZE=256`, `STRIDE=128`, `resnet3d-152-3d-decoder`, `r152_3ddec_v2_l5_epoch13.ckpt`, scored as Pearson against the published `new_canon_autoresearch_recipe` map of the same segment.

| scroll | crop y,x | r at `START_LAYER=23` | r at `START_LAYER=24` | 23 minus 24 |
|---|---|---:|---:|---:|
| PHerc0139 | 0, 15910 | 0.9900 | 0.9863 | +0.0037 |
| PHerc0139 | 3432, 23954 | 0.9520 | 0.9479 | +0.0041 |
| PHerc0139 | 3492, 3432 | 0.8220 | 0.8525 | -0.0305 |
| PHerc0814 | 6591, 20828 | 0.8095 | 0.8708 | -0.0613 |
| PHerc0814 | 14292, 18224 | 0.9059 | 0.8865 | +0.0194 |
| PHerc0814 | 14903, 92184 | 0.9256 | 0.9239 | +0.0017 |
| PHerc1667 | 5072, 9822 | 0.9725 | 0.9731 | -0.0006 |
| PHerc1667 | 10338, 48536 | 0.9714 | 0.9731 | -0.0017 |
| PHerc1667 | 15312, 38001 | 0.9668 | 0.9714 | -0.0046 |
| PHerc1667 | 20288, 0 | 0.9540 | 0.9601 | -0.0061 |
| PHerc1667 | 20480, 16384 | 0.9836 | 0.9815 | +0.0021 |
| PHerc1667 | 35504, 49110 | 0.9418 | 0.8886 | +0.0532 |
| PHercParis4 | 11646, 18694 | 0.9876 | 0.9831 | +0.0045 |
| PHercParis4 | 16928, 24064 | 0.9876 | 0.9803 | +0.0073 |
| PHercParis4 | 43379, 22110 | 0.9682 | 0.9698 | -0.0016 |

**23 wins 8, 24 wins 7, no ties. Exact two sided sign test p = 1.0000.** Median paired difference +0.0017, mean -0.0007, range -0.0613 to +0.0532. Worst correlation against a shuffled copy of the reference across every one of these runs: 0.0012. Every scroll produces both signs.

**The scale is the part I would act on.** Over the same 15 crops, median r is 0.7474 at the README's `START_LAYER=1` and 0.9698 at the better of 23 and 24. So the window fix is worth +0.2224 in median r and the half layer is worth +0.0017 at p = 1.0000, a factor of about 131. Crop to crop, r ranges 0.8095 to 0.9900, a spread of 0.1805, against a largest single crop half layer effect of 0.0613.

**Where this disagrees with #1765, stated as a disagreement rather than smoothed over.** On PHerc0139, the scroll #1765 measured, my three crops split 2 to 1 for 23 (-0.0305, +0.0037, +0.0041), so I do not reproduce 8 of 8. Two differences could explain it and I have not tested either: crop size, 1024 px here against 2048 px there, and crop selection, the single most ink rich window per segment here against random positions filtered on published ink fraction there. 8 of 8 is itself unlikely under a coin flip, so I am not calling that result noise; I am saying it does not hold when the same comparison is run across four scrolls.

**What I think follows for the README, and it is the cheap conclusion.** Both PRs already write 23, which is what `(109 - 62) // 2` gives. On this evidence there is no reason to change that and no reason for either PR to argue the half layer, because the floor formula is already defensible and the remaining difference is not resolvable at this crop size. If anything is worth adding to the README it is the rule rather than the constant, which #1766 already does in its Tips line.

Raw per crop numbers are in `results/halflayer.json` and the 15 per crop files `results/window_generalise_hl_*.json` in https://github.com/AndreasHad04/villa-apple-silicon , MIT, free to lift into #1766.

One more correction while I am here: one row of my published `results/window_tally.json` carried the string `20240304141531-w013 (offset sweep)` in its `seg` field, which is a label and not a segment id, so that row could not be re-run programmatically. It is now the real id, `20240304141531-w013_20240304141531_flatboi`, with the previous file kept beside it.
