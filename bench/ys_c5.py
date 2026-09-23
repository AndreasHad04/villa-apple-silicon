"""ARM Y, control C5 (added AFTER the Y1/Y2 result, not pre-registered): are the
two label sets equally related to the organisers' canonical model? Scores the
organisers' own w00 predictions stored in the label bucket against ARM X's bucket
labels on the render grid, with ARM X's pooling (2x2 mean) and ARM X's scorer.
If the canonical prediction scores as high on the render labels as on the
published 20260918 labels, the render-versus-published gap for ink_9um is not a
label artefact.

    env/bin/python3 ops/ys_c5.py
"""
import json, pathlib, sys, tempfile, urllib.request
import numpy as np, tifffile
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import xs_score as XS  # noqa: E402
HF = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/841/w00/preds/"
FILES = ["w00_canonical_2um_20250807020208.tif", "w00_canonical_030726_reverse_070326.tif", "ps48_640_640_smooth_0.1_w00_ckpt_130000_forward_210326.tif"]
sd = ROOT / "data" / "xs" / "p0841_w00"; meta = json.load(open(sd / "meta.json"))
label = np.load(sd / "label.npy"); support = np.load(sd / "support.npy")
y0, y1, x0, x1 = meta["bbox_native"]; out = {}
for f in FILES:
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / f; urllib.request.urlretrieve(HF + f, p); a = tifffile.imread(p)
    rec = dict(shape=list(a.shape), dtype=str(a.dtype), render_shape=meta["render_shape"][1:])
    if a.ndim == 3: a = a[..., 0] if a.shape[-1] in (1, 3, 4) else a[0]
    if a.dtype != np.uint8: a = np.rint(np.clip(a.astype(np.float64) / (255.0 if a.max() > 1 else 1.0), 0, 1) * 255).astype(np.uint8)
    if list(a.shape) != meta["render_shape"][1:]:
        rec["note"] = "not on the render canvas, not scored"; out[f] = rec; print(f, rec, flush=True); continue
    c = a[y0:y1, x0:x1].astype(np.float32); h, w = label.shape
    q = np.rint(c[: 2 * h, : 2 * w].reshape(h, 2, w, 2).mean(axis=(1, 3))).astype(np.uint8)
    pos, neg = XS.block_hists(q, label, support)
    rec.update(auc=XS.auc_from(pos.sum(0), neg.sum(0)), ci95=XS.bootstrap(pos, neg, n=300))
    out[f] = rec; print(f, {k: rec[k] for k in ("shape", "auc", "ci95")}, flush=True)
json.dump(out, open(ROOT / "results" / "ys" / "c5_render_labels.json", "w"), indent=1)
