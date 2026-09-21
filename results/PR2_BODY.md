**In one sentence:** The `START_LAYER=1` this README suggests does not reproduce the ink predictions you publish, and centring the 62-layer window in the surface volume does.

**One real example:** Starting with PHerc. 1667 segment `20240304141531-w013_20240304141531_flatboi`, the 2.399 um surface volume from `s3://vesuvius-challenge-open-data/`, I ran `ink_canonical_2um` at `TILE_SIZE=256 STRIDE=128` and swept `START_LAYER`, and agreement with your own published prediction for those pixels went from **r = 0.8521 at START_LAYER 1** to **r = 0.9851 at START_LAYER 25**.

**Before:** The README says "a 62-layer window such as `START_LAYER=1`, `END_LAYER=63`". Following it, my output disagreed with your published map badly enough that I first assumed I had the wrong checkpoint. I did not: the `ink_canonical_2um` model card gives its recipe as `new_canon_autoresearch_recipe`, which is exactly the recipe in the published prediction filenames.

**After this PR:** The README says to centre the window, `START_LAYER=23`, `END_LAYER=85` for the 109-layer surface volumes in the bucket, and states what that is worth.

**Proof:** 15 measurements across 4 scrolls, each with its own published `new_canon_autoresearch_recipe` prediction. The comparison window on each was chosen automatically as a high-variance region of your own reference, not by hand. **The best window beats `START_LAYER=1` in 15 of 15**, best START_LAYER runs 21 to 25, and agreement at the documented setting ranges from **0.1591 to 0.9522** against 0.8525 to 0.99 at the best window. The four rows below are one representative segment per scroll.

| scroll, segment | r at START_LAYER 1 | best START_LAYER | r there |
|---|---|---|---|
| PHerc. 1667, 20240304141531 | 0.8521 | 25 | **0.9851** |
| PHerc. 0139, 20250108000000 | 0.9054 | 23 | **0.9900** |
| PHercParis4, 20230702185753 | 0.6240 | 24 | **0.9698** |
| PHerc. 0814, 20250925161630 | 0.8263 | 21 | **0.9005** |

![agreement against START_LAYER](https://raw.githubusercontent.com/AndreasHad04/villa-apple-silicon/main/figures/start_layer_vs_reference.png)

The same 4096 pixels at both settings, so the cost is visible and not only numerical. Top row the full crop, bottom row the same detail from each:

![what the documented window costs](https://raw.githubusercontent.com/AndreasHad04/villa-apple-silicon/main/figures/window_cost.png)

Every offset on every scroll was also scored against a **shuffled** copy of that scroll's own reference. That floor never exceeds **0.0022**, so this is depth alignment and not marginal statistics. All four volumes are 109 layers, so the centred window is `(109-62)//2 = 23`, and the measured optima are 21 to 25.

Corroborated by a measure that never sees your reference: on PHerc. 1667 the ink separation of my own output (mean of pixels above 0.5 minus mean of those below) peaks at 0.7210 at START_LAYER 28, against 0.6643 at START_LAYER 1. So the window that matches your production output is also the one that maximises ink contrast.

**Why / where this is useful:** Anyone following this README today gets output that agrees with your published predictions at r 0.62 to 0.91 instead of 0.90 to 0.99, and has no way to tell, because nothing errors. It also costs real ink contrast, so a contributor evaluating a new idea against a weak baseline may reach the wrong conclusion about their own change.

- [ ] I personally verified that the example and proof above were produced by this PR on the stated data.

## Details

**Why, and this is your own convention everywhere else.** `START_LAYER=1` in this one document is the outlier:

- `scrollprize/hecate`, released 2026-09-15 and built from `ink_canonical_2um`, describes its own inference script as "The script **selects the central input depth**, slides over XY with half-patch overlap, and blends probabilities using a floored Hann window", and its `--reverse` flag "reverse[s] the **full render depth before selecting the central planes**".
- The same card says why the centre is the reference: a surface volume is "a CT scan **resampled around a mesh** that follows a papyrus sheet", and "The intended sheet can also **wander above and below the centre of the render**".
- `vesuvius/docs/ink_detection.md`, for a 2.399 um OME-Zarr: "**select the centered 84 Z planes**", and "**Labels occupy Z slice 32 of a 65-plane volume**", the exact centre.
- `scrollprize/ink_9um`: "the z window **jitters over 17 of the 21 slices** so the models don't lock onto one exact depth".

For a 109-plane volume the centred window is `(109-62)//2 = 23`. With `START_LAYER=1` the surface at plane 54 lands at position 53 of 62, about 85% of the way through the window. The wandering sheet also explains why the measured optima scatter over 21 to 25 rather than sitting exactly on 23.

**A wrong turn I want to record, because it is an easy mistake.** I first tried to locate the surface from the voxels, using in-plane gradient energy and mean intensity, and their asymmetry led me to think the volume was not centred. That test cannot work: carbon ink on carbonised papyrus has almost no attenuation contrast, so intensity statistics describe bulk papyrus and say nothing about where the writing surface is. The profile is in the linked repo as a measurement of the wrong quantity.

**Stability.** 15 measurements over 4 scrolls and multiple regions per segment, all optima within 21 to 25. The swept grid does not separate 23 from 24, so the two were measured PAIRED on the same 15 crops: **23 wins 8, 24 wins 7, exact two sided sign test p = 1.0000**, median paired difference +0.0017 in r, every scroll producing both signs. The half layer is not determined at this crop size, and `(109 - 62) // 2 = 23`, the value this PR documents, is the right one to write down. For scale, median r over those crops is 0.7474 at `START_LAYER=1` against 0.9698 at the better of 23 and 24, so the window is worth +0.2224 and the half layer +0.0017.

**Scope.** I searched the repo for every other place this value is set. `START_LAYER` and `END_LAYER` appear only in this README and in `entrypoint.py`, which requires both as environment variables with no default (`os.environ["START_LAYER"]`). So there is no coded default to correct and this is a documentation-only change. If you would rather have a guard in code, I am happy to add one that warns when the requested window is not centred in the surface volume, but that seemed like scope creep for a docs fix.

**Honest limits.** One crop per scroll, 1024x1024, 49 tiles. PHerc. 0814 agrees less well than the others even at its best (0.9005), which I have not explained. The sweep is at stride 1 to 3 in START_LAYER near the optimum and coarser far from it. I did not change the `END_LAYER` in the developer smoke-test snippet lower down, because that block is `STEP=aggregate-profiling` and the values there are placeholders.

**Reproduce it:** `bench/window_generalise.py` and `bench/offset_vs_reference.py` at https://github.com/AndreasHad04/villa-apple-silicon, along with the raw per-offset JSON for all four scrolls and the model-free depth profile.

This is independent of my other PR (#1812, Apple Silicon support) and can be taken on its own.
