"""Side-by-side evidence: villa's published prediction vs the same pixels
produced on Apple Silicon through the ported path.
"""
import argparse, pathlib, numpy as np, tifffile, zarr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--y0", type=int, default=20480)
ap.add_argument("--x0", type=int, default=16384)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--tag", default="ink")
ap.add_argument("--out", default="prediction_vs_reference.png")
a = ap.parse_args()

mine = np.load(R/"results"/f"{a.tag}_mps_{a.y0}_{a.x0}_{a.size}.npy").astype(np.float64)
full = zarr.open(tifffile.imread(R/"reference"/"their_pred_2399um.tif", aszarr=True), mode="r")
theirs = np.asarray(full[a.y0:a.y0+a.size, a.x0:a.x0+a.size]).astype(np.float64)/255.0
r = float(np.corrcoef(mine.ravel(), theirs.ravel())[0, 1])

fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.4))
panels = [(theirs, "villa published prediction (reference)", "gray"),
          (mine,   "same pixels, Apple Silicon via this patch", "gray"),
          (np.abs(mine-theirs), "absolute difference", "magma")]
for axis, (img, title, cmap) in zip(ax, panels):
    axis.imshow(img, cmap=cmap, vmin=0, vmax=1)
    axis.set_title(title, fontsize=12, pad=10)
    axis.set_xticks([]); axis.set_yticks([])
fig.subplots_adjust(top=0.88, bottom=0.10, wspace=0.05)
fig.text(0.5, 0.955, f"PHerc. 1667  segment 20240304141531  2.399 um  "
         f"crop y={a.y0} x={a.x0} {a.size}x{a.size}  |  Pearson r = {r:.3f}",
         ha="center", fontsize=13)
fig.text(0.5, 0.035, "Same checkpoint (ink_canonical_2um), TILE_SIZE=256, STRIDE=128, "
         "layers 1 to 63. Reference read from the open-data bucket.",
         ha="center", fontsize=10, color="0.35")
(R/"public"/"figures").mkdir(parents=True, exist_ok=True)
fig.savefig(R/"public"/"figures"/a.out, dpi=105, bbox_inches="tight")
print(f"wrote {a.out}  r={r:.4f}  crop={a.size}")
