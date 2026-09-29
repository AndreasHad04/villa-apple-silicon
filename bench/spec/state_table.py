#!/usr/bin/env python3
"""Print the ARM SPEC result table for STATE.md from results/spec/spec_scores.json (generated, never typed)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
d = json.loads((ROOT / "results/spec/spec_scores.json").read_text())
crops = ["w00", "ag144", "ag174", "p0500p2"]
print("| crop | AUC B | AUC K | AUC T | T - K | P>0.5 permuted K / T | reversed K / T | rolled K / T |")
print("|---|---|---|---|---|---|---|---|")
for c in crops:
    r = {a: d["rows"][a][c] for a in ("B", "K", "T")}
    print(f"| {c} | {r['B']['real']['auc']:.4f} | {r['K']['real']['auc']:.4f} | {r['T']['real']['auc']:.4f} | "
          f"{d['t_minus_k'][c]:+.4f} | {r['K']['permuted']:.4f} / {r['T']['permuted']:.4f} | "
          f"{r['K']['reversed']:.4f} / {r['T']['reversed']:.4f} | {r['K']['rolled']:.4f} / {r['T']['rolled']:.4f} |")
km = sorted(d["k_minus_b"].values())
print(f"\nverdict {d['verdict']}: median T - K {d['delta_median']:+.4f}, T > K on {d['t_beats_k']} of 4; "
      f"positive control {sum(d['positive_control'].values())} of 4, manipulation check {sum(d['manipulation'].values())} of 4; "
      f"K - B within [{km[0]:+.1e}, {km[-1]:+.1e}]")
R = lambda arm, key: [d["rows"][arm][c][key] for c in crops]
rng = lambda xs: f"{min(xs):.4f} to {max(xs):.4f}"
print(f"\nT's P>0.5 on the permuted-planes null (trained disorder): {rng(R('T', 'permuted'))} (K {rng(R('K', 'permuted'))}); "
      f"rolled by half (never trained): {rng(R('T', 'rolled'))} (K {rng(R('K', 'rolled'))}); "
      f"reversed (never trained): {rng(R('T', 'reversed'))} (K {rng(R('K', 'reversed'))}).")
print("95% CI per crop, K then T: " + "; ".join(
    f"{c} [{d['rows']['K'][c]['real']['ci95'][0]:.4f}, {d['rows']['K'][c]['real']['ci95'][1]:.4f}] / "
    f"[{d['rows']['T'][c]['real']['ci95'][0]:.4f}, {d['rows']['T'][c]['real']['ci95'][1]:.4f}]" for c in crops))
