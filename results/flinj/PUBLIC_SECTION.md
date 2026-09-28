## Can ink_9um draw a letter on the eligible scan type?

On the eligible 1.2 m scans, ink_9um finds ink but draws it as blobs, never as letters. Two causes are possible and
they call for different work: the model cannot draw strokes when ink is as faint as it is at 9 um, or the ink in these
scans is not stroke-shaped where the model looks.

To separate them I planted real ink in the shape of Greek capitals into eligible-type scans and checked whether the
model draws the letters back. The ink is real: voxel columns cut from labelled ink on PHerc0841's 9.366 um, 113 keV
scan, moved into other crops under a letter-shaped mask (cap height 2.5 mm, strokes 0.35 mm). The control is the same
transplant made of non-ink papyrus, so both runs share every seam and differ only in the ink; the difference of the two
predictions is what gets scored. Four host crops, each receiving ink from other crops than itself: three PHerc0841
segments and PHerc0500P2 (9.362 um, 113 keV). The test, its metric and its thresholds were registered before any input
existed; four amendments fixed design problems found by model-free checks and by one smoke run of the first design,
all before the final design's outcome was computed. Everything is in the same folder.

The score is counter contrast C: how much emptier the inside of a letter (the hole in O, A, Theta) is than its
strokes, measured against plain background. 1 is a drawn letter, 0 a filled blob, and above 1 the inside of the
letter reads even lower than plain background. Registered bars: 0.5 or more reads as strokes, 0.2 or less as blobs.

| host | C | response on strokes (probability) | AUC, ink vs non-ink transplant, stroke pixels | same model on the crop's real labelled ink |
|---|---|---|---|---|
| PHerc0841 w00 | 0.7040 | 0.1010 | 0.7059 | 0.7680 |
| PHerc0841 ag144 | 0.7348 | 0.0686 | 0.6903 | 0.7599 |
| PHerc0841 ag174 | 0.9420 | 0.0517 | 0.6205 | 0.7614 |
| PHerc0500P2 | 1.2689 | 0.0296 | 0.6149 | 0.7537 |

**The model keeps the letter shape: median C 0.8384, the registered verdict is STROKES.** The positive control (a
clean letter-shaped bump) is drawn as a letter on 3 of 4 hosts, as the registration requires. The mean of all
14 released checkpoints gives C 0.98 to 1.03.

**What hides the letters is how weakly it responds.** On stroke pixels the planted ink is told apart from the non-ink
transplant with an AUC of 0.61 to 0.71, which is 0.06 to 0.14 below the same model on the real labelled ink of the same crops.
In the prediction itself the planted letters barely rise above the model's response to the host papyrus (figure:
mask, prediction with the non-ink transplant, prediction with the ink transplant, difference).

![FLINJ w00](figures/flinj_w00.png)

**And it responds more to disorder than to ink.** The same planted ink with its depth planes shuffled is detected more
than the real ink (AUC 0.81 to 0.90), while a plain brightness bump is not detected at all (AUC 0.32 to 0.46). A model that
fires harder on depth-disordered texture than on ink paints blobs wherever a surface is disordered.

So on this scan type what stands between ink_9um and letters is not its ability to draw strokes but its sensitivity,
and its response to depth disorder. A fine-tune aimed at that second part is running.

Limits: 113 keV scans only (the one 116 keV crop had no room for a row of letters clear of its own ink); the planted
ink is real but tiled from 32-pixel patches of tracings about 1 mm wide, and the model detects it less well than real
labelled ink, so the transplant loses part of what the model uses; the primary is one checkpoint (seed 42, step 75,000).

Code: `bench/flinj/`. Registration, amendments and the full generated report: `results/flinj/`.
