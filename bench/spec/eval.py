"""ARM SPEC evaluation and verdict, results/spec/PREREGISTRATION_SPEC.md (sha 4df26a95...).

    python ops/spec/eval.py            # infer whatever is missing, then score and write results/spec/spec_scores.json

Arms B (released weights), K, T (results/spec/<arm>/ckpt.pth). Every prediction goes through ops/flinj/common.infer
(the ARM Z / FLINJ path, unchanged) and is scored with ops/ys_score.metrics. Nulls: all 21 planes permuted (seed 11),
reversed, rolled by 10. Resumable: an existing prediction tif is reused. Labels are read here and nowhere in training.
"""
import json
import pathlib
import sys

import numpy as np
import tifffile
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops")); sys.path.insert(0, str(ROOT / "ops" / "flinj"))
import common as C  # noqa: E402
import ys_score as YS  # noqa: E402

OUT = ROOT / "results" / "spec"
CKPTS = {"B": ROOT / "models/ink_9um/hybrid_3d2d-seed42/step-075000.pth",
         "K": OUT / "K" / "ckpt.pth", "T": OUT / "T" / "ckpt.pth"}
CROPS = {"w00": ("data/zs/w00/w04/ct.zarr", "data/zs/w00"), "ag144": ("data/zs/ag144/w04/ct.zarr", "data/zs/ag144"),
         "ag174": ("data/zs/ag174/w04/ct.zarr", "data/zs/ag174"),
         "p0500p2": ("data/f6/p0500p2/N0/ct.zarr", "data/f6/p0500p2")}
STORED = {"w00": 0.7680, "ag144": 0.7599, "ag174": 0.7614, "p0500p2": 0.7537}   # FLINJ_REPORT.md S7, registered
NULLS = ("permuted", "reversed", "rolled")


def null_input(k, kind):
    path = ROOT / "data" / "spec" / k / kind / "ct.zarr"
    if not path.exists():
        a = zarr.open_array(str(ROOT / CROPS[k][0]), mode="r")[:]
        b = {"permuted": lambda: a[np.random.default_rng(11).permutation(a.shape[0])],
             "reversed": lambda: a[::-1], "rolled": lambda: np.roll(a, 10, axis=0)}[kind]()
        C.write_zarr(path, np.ascontiguousarray(b))
    return path


def pred(arm, k, kind):
    tif = OUT / "pred" / arm / k / f"{kind}.tif"
    if not tif.exists():
        ct = ROOT / CROPS[k][0] if kind == "real" else null_input(k, kind)
        print(f"infer {arm} {k} {kind}: {C.infer(ct, CKPTS[arm], tif):.1f} s", flush=True)
    return tifffile.imread(tif)


def main():
    arms = [a for a in CKPTS if CKPTS[a].exists()]
    rows = {}
    for arm in arms:
        for k, (_, d) in CROPS.items():
            label, support = np.load(ROOT / d / "label.npy"), np.load(ROOT / d / "support.npy")
            r = {"real": YS.metrics(pred(arm, k, "real"), label, support)}
            for kind in NULLS:
                r[kind] = float((pred(arm, k, kind)[support] > 127).mean())
            rows.setdefault(arm, {})[k] = r
    out = dict(registration=open(OUT / "prereg.sha256").read().split()[0], arms=arms, rows=rows)
    if {"B", "K", "T"} <= set(arms):
        auc = {a: {k: rows[a][k]["real"]["auc"] for k in CROPS} for a in arms}
        pc = {k: round(auc["B"][k], 4) == STORED[k] for k in CROPS}
        manip = {k: rows["T"][k]["permuted"] < rows["K"][k]["permuted"] for k in CROPS}
        diff = {k: auc["T"][k] - auc["K"][k] for k in CROPS}
        delta = float(np.median(list(diff.values()))); wins = sum(v > 0 for v in diff.values())
        if not all(pc.values()):
            verdict = "VOID (positive control: B does not reproduce the stored AUCs)"
        elif sum(manip.values()) < 3:
            verdict = "VOID (manipulation check: T does not lower the permuted-input ink fraction on 3 of 4)"
        elif delta >= 0.010 and wins >= 3:
            verdict = "WIN"
        elif delta <= -0.010:
            verdict = "HARM"
        else:
            verdict = "NULL"
        out.update(positive_control=pc, manipulation=manip, auc=auc, t_minus_k=diff, delta_median=delta,
                   t_beats_k=wins, k_minus_b={k: auc["K"][k] - auc["B"][k] for k in CROPS}, verdict=verdict)
        print(f"VERDICT {verdict}: median AUC(T) - AUC(K) {delta:+.4f}, T > K on {wins} of 4", flush=True)
    C.atomic_json(OUT / "spec_scores.json", out)
    for arm in arms:
        print(arm, {k: (round(rows[arm][k]["real"]["auc"], 4), {n: round(rows[arm][k][n], 4) for n in NULLS})
                    for k in CROPS}, flush=True)


if __name__ == "__main__":
    main()
