"""ARM X data survey: enumerate the organisers' labelled ink segments in
huggingface.co/buckets/scrollprize/datasets (ink/*), download only the small 2D
label and mask TIFs plus meta.json, and report per segment: native spacing,
render shape, supervised area, ink fraction. No CT is downloaded here."""
import json, re, sys, urllib.request, pathlib
import numpy as np, tifffile
API = "https://huggingface.co/api/buckets/scrollprize/datasets/tree/"
RES = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "xs_labels"
def ls(p):
    return json.load(urllib.request.urlopen(API + urllib.parse.quote(p) + "?recursive=false", timeout=60))
def get(p, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0: return dest
    tmp = dest.with_suffix(dest.suffix + ".partial")
    with urllib.request.urlopen(RES + urllib.parse.quote(p), timeout=300) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    tmp.replace(dest); return dest
import urllib.parse
def seg_dirs(root):
    """A segment dir is one that directly holds a *_inklabels.tif."""
    out = []
    for x in ls(root):
        name = x["path"].split("/")[-1]
        if x["type"] == "file" and name.endswith("_inklabels.tif"):
            return [root]
    for x in ls(root):
        if x["type"] == "directory" and not x["path"].endswith(".zarr"):
            out += seg_dirs(x["path"])
    return out
rows = []
roots = sys.argv[1:] or ["ink/unused", "ink/841", "ink/man5", "ink/0009b", "ink/0500p2"]
for root in roots:
    for d in seg_dirs(root):
        files = {x["path"].split("/")[-1]: x for x in ls(d)}
        lab = sorted(k for k in files if k.endswith("_inklabels.tif"))
        sup = sorted(k for k in files if k.endswith("_supervision_mask.tif"))
        zarrs = sorted(k for k in files if k.endswith(".zarr") and "labels" not in k and "mask" not in k and "max_" not in k)
        meta = {}
        if "meta.json" in files:
            meta = json.load(open(get(d + "/meta.json", OUT / d / "meta.json")))
        r = {"dir": d, "labels": lab, "masks": sup, "ct_zarr": zarrs, "volume": meta.get("volume"), "scroll": meta.get("scroll_source"),
             "has_validation": any("validation_mask" in k for k in files)}
        if lab and sup:
            L = tifffile.imread(get(d + "/" + lab[0], OUT / d / lab[0]))
            M = tifffile.imread(get(d + "/" + sup[0], OUT / d / sup[0]))
            L = L if L.ndim == 2 else L[..., 0]; M = M if M.ndim == 2 else M[..., 0]
            sm = M > 0; pos = (L > 0) & sm
            ys, xs = np.nonzero(sm)
            r.update(shape=list(L.shape), sup_px=int(sm.sum()), ink_px=int(pos.sum()),
                     ink_frac=round(float(pos.sum() / max(sm.sum(), 1)), 4),
                     sup_bbox=[int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1] if len(ys) else None,
                     label_values=sorted(int(v) for v in np.unique(L))[:6])
        rows.append(r); print(json.dumps(r), flush=True)
json.dump(rows, open(pathlib.Path(__file__).resolve().parents[1] / "results" / "xs" / "labels_survey.json", "w"), indent=1)
