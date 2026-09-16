"""Side-by-side evidence for the PR: villa's published prediction vs the same
pixels produced on Apple Silicon by the ported path.
"""
import pathlib, numpy as np, tifffile, zarr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parent.parent
Y0, X0, SZ = 20480, 16384, 1024
mine = np.load(R/"results"/f"ink_mps_{Y0}_{X0}_{SZ}.npy").astype(np.float64)
full = zarr.open(tifffile.imread(R/"reference"/"their_pred_2399um.tif", aszarr=True), mode="r")
theirs = np.asarray(full[Y0:Y0+SZ, X0:X0+SZ]).astype(np.float64)/255.0
r = float(np.corrcoef(mine.ravel(), theirs.ravel())[0, 1])

fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
for a, (img, t) in zip(ax, [
        (theirs, "villa published prediction\n(reference, from the open-data bucket)"),
        (mine,  "same pixels, Apple Silicon (MPS)\nvia this patch"),
        (np.abs(mine-theirs), "absolute difference")]):
    a.imshow(img, cmap="gray" if "difference" not in t else "magma", vmin=0, vmax=1)
    a.set_title(t, fontsize=11); a.axis("off")
fig.suptitle(f"PHerc. 1667 segment 20240304141531, 2.399 um, crop y={Y0} x={X0} {SZ}x{SZ}   "
             f"Pearson r = {r:.3f}", fontsize=12)
fig.tight_layout()
fig.savefig(R/"public"/"figures"/"prediction_vs_reference.png", dpi=110, bbox_inches="tight")
print("wrote prediction_vs_reference.png  r =", round(r, 4))
