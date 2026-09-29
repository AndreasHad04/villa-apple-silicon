#!/usr/bin/env python3
"""ARM SPEC registered secondary (results/spec/PREREGISTRATION_SPEC.md, "Secondary, descriptive"): the FLINJ transplant
pair re-scored with the fine-tuned arms K and T.

For each FLINJ primary host (w00, ag144, ag174, p0500p2) and arm: B1 ink transplant, B2 non-ink transplant and C1
shuffled-ink transplant, each one inference of the full host crop through FLINJ's own run_all.run_host / run_unit
(inputs rebuilt deterministically, their sha256 compared with the ones FLINJ recorded), scored with FLINJ's own
report.host_results (metric.score_pair: counter contrast C, d, stroke AUC for the primary pair; the same for C1).
Nothing under results/flinj is written: units, predictions and figures go to results/spec/flinj/<arm>/.

Positive control, run first and gating everything else: arm B (the released weights) on w00 through this path must
reproduce FLINJ's stored b1_ink, b2_non and c1_shuffle predictions bit for bit.

    /Users/andreashad04/money/vesuv/env/bin/python3 /Users/andreashad04/money/vesuv/ops/spec/flinj_rescore.py
"""
import json, sys, time
from pathlib import Path

import numpy as np, tifffile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops" / "flinj"))
import run_all as RA  # noqa: E402
import report as RP  # noqa: E402
from common import PRIMARY_CKPT, atomic_json, banner  # noqa: E402

OUT = ROOT / "results" / "spec" / "flinj"
CK = dict(B=PRIMARY_CKPT, K=ROOT / "results/spec/K/ckpt.pth", T=ROOT / "results/spec/T/ckpt.pth")
HOSTS = ["w00", "ag144", "ag174", "p0500p2"]
NAMES = ("b1_ink", "b2_non", "c1_shuffle")
ORIG_UNITS, ORIG_PRED, ORIG_UNITS_DIR = RA.units, RA.PRED, RA.UNITS


def arm_units(arm):
    def units(h):
        base = {n: s for n, s in ORIG_UNITS(h) if n in NAMES}
        assert set(base) == set(NAMES), (h, sorted(base))
        assert all(s["ckpt"] == PRIMARY_CKPT for s in base.values()), h
        return [(n, dict(base[n], ckpt=CK[arm])) for n in NAMES]
    return units


def point(arm):
    d = OUT / arm
    RA.UNITS, RA.PRED, RA.units = d / "units", d / "pred", arm_units(arm)
    RP.UNITS, RP.PRED, RP.FIGS, RP.units = d / "units", d / "pred", d / "figs", arm_units(arm)
    assert RA.PRED != ORIG_PRED and RP.FIGS != ROOT / "results" / "flinj" / "figs"


def input_matches(arm, h):
    """sha256 of every rebuilt input against the one FLINJ recorded for the same unit."""
    out = {}
    for n in NAMES:
        mine = json.load(open(OUT / arm / "units" / h / f"{n}.json"))["input"]["sha256"]
        theirs = json.load(open(ORIG_UNITS_DIR / h / f"{n}.json"))["input"]["sha256"]
        out[n] = mine == theirs
    return out


def run(arm, hosts, need):
    point(arm)
    for h in hosts:
        t = time.time()
        assert RA.run_host(h, need), f"run_host refused {arm} {h}"
        m = input_matches(arm, h)
        print(f"{arm} {h}: 3 units in {time.time() - t:.0f} s, inputs equal FLINJ's: {m}", flush=True)
        assert all(m.values()), f"{arm} {h}: a rebuilt input differs from FLINJ's recorded sha256"


def main():
    banner(__file__)
    for ck in CK.values(): assert Path(ck).exists(), ck
    need = RA.need_gb()[0]
    OUT.mkdir(parents=True, exist_ok=True)
    # Positive control: B on w00 must reproduce FLINJ's stored predictions exactly.
    run("B", ["w00"], need)
    ctrl = {n: int(np.abs(tifffile.imread(OUT / "B" / "pred" / "w00" / f"{n}.tif").astype(np.int32)
                          - tifffile.imread(ORIG_PRED / "w00" / f"{n}.tif")).max()) for n in NAMES}
    print(f"POSITIVE CONTROL max |diff| vs FLINJ stored predictions: {ctrl}", flush=True)
    atomic_json(OUT / "control.json", dict(max_abs_diff=ctrl, passed=all(v == 0 for v in ctrl.values())))
    if not all(v == 0 for v in ctrl.values()):
        print("VOID: this path does not reproduce FLINJ's stored predictions", flush=True); return 2
    for arm in ("T", "K"):
        run(arm, HOSTS, need)
    rows = {}
    for arm in ("B", "K", "T"):
        point(arm)
        for h in (["w00"] if arm == "B" else HOSTS):
            R = RP.host_results(h)
            keep = ("C", "C_b3", "b3_defined", "d_prob", "auc_pair", "max_abs_D_far")
            rows.setdefault(arm, {})[h] = {k: {q: R["pairs"][k][q] for q in keep} for k in ("primary", "c1")}
    flinj = json.load(open(ROOT / "results" / "flinj" / "report_data.json"))
    flinj_B = {h: {k: {q: flinj[h]["pairs"][k][q] for q in ("C", "C_b3", "b3_defined", "d_prob", "auc_pair")} for k in ("primary", "c1")}
               for h in HOSTS}
    for q in ("C", "d_prob", "auc_pair"):  # the path reproduces FLINJ's published numbers for B on w00
        for k in ("primary", "c1"):
            assert rows["B"]["w00"][k][q] == flinj_B["w00"][k][q], (k, q, rows["B"]["w00"][k][q], flinj_B["w00"][k][q])
    atomic_json(OUT / "rescore.json", dict(control=ctrl, rows=rows, flinj_report_B=flinj_B,
                                           script_mtime=time.ctime(Path(__file__).stat().st_mtime)))
    for arm, hs in rows.items():
        for h, p in hs.items():
            print(f"{arm} {h:8s} primary C {p['primary']['C']:.4f} d {p['primary']['d_prob']:.4f} stroke AUC {p['primary']['auc_pair']:.4f}"
                  f" | C1 AUC {p['c1']['auc_pair']:.4f} C1 C {p['c1']['C']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
