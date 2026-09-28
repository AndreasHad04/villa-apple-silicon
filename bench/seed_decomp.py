"""Seed versus training-step decomposition of the 14 ink_9um checkpoints.

Answers PedroR4321 on villa #1867 (2026-09-27): is the spread the checkpoint
ranking is asked to explain mostly SEED or mostly STEP?  No new inference; reads
the ARM Y and ARM Z score files that the published numbers came from.

Arrays: 2.403 um published volume (ARM Y), 9.366 um eligible volume (ARM Z),
4.681 um render (ARM X column carried in ARM Y), and the in-distribution C1
score with the direction chosen correctly (c1_oracle).

Two-way additive model on the 2 seeds x 7 steps grid, one value per cell:
SS_total = SS_seed (df 1) + SS_step (df 6) + SS_resid (df 6).
Writes results/seed_decomp.json.
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
Y = json.loads((ROOT / "results/ys/army_scores.json").read_text())["ALLCKPT"]
Z = json.loads((ROOT / "results/zs/armz_scores.json").read_text())["ALLCKPT"]
PAT = re.compile(r"^hybrid_3d2d-seed(42|43)_step-(\d{6})$")

ARRAYS = {
    "published_2p403um": Y["published_median"],
    "eligible_9p366um": Z["eligible_median"],
    "render_4p681um": Y["render_median"],
    "c1_in_distribution": Y["c1_oracle"],
}


def grid(d):
    cells = {}
    for k, v in d.items():
        m = PAT.match(k)
        if m:
            cells[(int(m.group(1)), int(m.group(2)))] = float(v)
    seeds = sorted({s for s, _ in cells})
    steps = sorted({t for _, t in cells})
    assert seeds == [42, 43], seeds
    assert len(steps) == 7, steps
    assert len(cells) == 14, len(cells)
    return np.array([[cells[(s, t)] for t in steps] for s in seeds]), steps


def decompose(g):
    mu = g.mean()
    seed_eff = g.mean(axis=1) - mu
    step_eff = g.mean(axis=0) - mu
    fit = mu + seed_eff[:, None] + step_eff[None, :]
    ss_tot = float(((g - mu) ** 2).sum())
    ss_seed = float(7 * (seed_eff ** 2).sum())
    ss_step = float(2 * (step_eff ** 2).sum())
    ss_res = float(((g - fit) ** 2).sum())
    assert abs(ss_tot - (ss_seed + ss_step + ss_res)) < 1e-12
    return {
        "range_all_14": float(g.max() - g.min()),
        "seed42_mean": float(g[0].mean()), "seed43_mean": float(g[1].mean()),
        "seed42_median": float(np.median(g[0])), "seed43_median": float(np.median(g[1])),
        "between_seed_gap_mean": float(g[1].mean() - g[0].mean()),
        "within_seed_range": [float(np.ptp(g[0])), float(np.ptp(g[1]))],
        "within_seed_sd": [float(g[0].std(ddof=1)), float(g[1].std(ddof=1))],
        "same_step_diff_43_minus_42": [float(x) for x in (g[1] - g[0])],
        "share_seed": ss_seed / ss_tot, "share_step": ss_step / ss_tot,
        "share_resid": ss_res / ss_tot,
        "step_effect_corr_between_seeds": float(np.corrcoef(g[0], g[1])[0, 1]),
    }


def main():
    grids, steps = {}, None
    for name, d in ARRAYS.items():
        grids[name], st = grid(d)
        assert steps is None or st == steps
        steps = st
    out = {"steps": steps, "arrays": {n: decompose(g) for n, g in grids.items()}}

    # Does in-distribution rank checkpoints once the seed offset is removed?
    c1 = grids["c1_in_distribution"]
    rank = {}
    for name in ("published_2p403um", "eligible_9p366um"):
        h = grids[name]
        within = [float(spearmanr(c1[i], h[i]).statistic) for i in range(2)]
        c1d = (c1 - c1.mean(axis=1, keepdims=True)).ravel()
        hd = (h - h.mean(axis=1, keepdims=True)).ravel()
        rank[name] = {
            "spearman_all_14": float(spearmanr(c1.ravel(), h.ravel()).statistic),
            "spearman_within_seed42_n7": within[0],
            "spearman_within_seed43_n7": within[1],
            "spearman_seed_demeaned_n14": float(spearmanr(c1d, hd).statistic),
        }
    out["c1_ranking"] = rank

    # Controls: the full-14 Spearman must reproduce the published figures.
    assert abs(rank["published_2p403um"]["spearman_all_14"] - Y["spearman_c1oracle_vs_published"]) < 1e-12
    assert abs(rank["eligible_9p366um"]["spearman_all_14"] - Z["Z2_spearman_c1oracle_vs_eligible"]) < 1e-12
    # Positive control: a grid built as pure seed offset must be attributed 100% to seed.
    fake = np.array([[0.80] * 7, [0.85] * 7])
    assert abs(decompose(fake)["share_seed"] - 1.0) < 1e-12
    # Negative control: a pure step effect shared by both seeds gives 0% seed.
    ramp = np.linspace(0.7, 0.8, 7)
    assert decompose(np.array([ramp, ramp]))["share_seed"] < 1e-12

    (ROOT / "results/seed_decomp.json").write_text(json.dumps(out, indent=1))
    for n, r in out["arrays"].items():
        print(f"{n:20s} range14 {r['range_all_14']:.4f}  seed42 {r['seed42_mean']:.4f}  seed43 {r['seed43_mean']:.4f}"
              f"  gap {r['between_seed_gap_mean']:+.4f}  within-range {r['within_seed_range'][0]:.4f}/{r['within_seed_range'][1]:.4f}"
              f"  shares seed {r['share_seed']:.3f} step {r['share_step']:.3f} resid {r['share_resid']:.3f}"
              f"  step-profile corr {r['step_effect_corr_between_seeds']:+.3f}")
    for n, r in rank.items():
        print(n, {k: round(v, 4) for k, v in r.items()})


if __name__ == "__main__":
    main()
