"""Writes results/ys/REPLY_1867_pedro.md from results/seed_decomp.json (every number generated, none typed)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "results/seed_decomp.json").read_text())
a, r = d["arrays"], d["c1_ranking"]
P, E, R, C = a["published_2p403um"], a["eligible_9p366um"], a["render_4p681um"], a["c1_in_distribution"]
pc = lambda x: f"{100 * x:.1f}%"
row = lambda lab, x: f"| {lab} | {pc(x['share_seed'])} | {pc(x['share_step'])} | {pc(x['share_resid'])} | {x['between_seed_gap_mean']:+.4f} |"
URL = "https://github.com/AndreasHad04/villa-apple-silicon/blob/main/bench/seed_decomp.py"
text = f"""Done from the published score files, no new inference ([`bench/seed_decomp.py`]({URL}), output `results/seed_decomp.json`). Two seeds by seven steps, split into a seed part, a step part and what is left:

| scores | seed part | step part | left over | seed43 minus seed42 |
|---|---|---|---|---|
{row("2.403 µm volumes, held out", P)}
{row("9.366 µm eligible volumes, held out", E)}
{row("4.681 µm renders, held out", R)}
{row("in-distribution (C1)", C)}

On the held-out scores the seed gap is small, and the two seeds do not agree on which steps are better: their step profiles correlate {P['step_effect_corr_between_seeds']:+.2f} on 2.403 µm and {E['step_effect_corr_between_seeds']:+.2f} on 9.366 µm. The large seed gap is in the in-distribution score, seed43 {C['between_seed_gap_mean']:+.3f}, and the held-out scores do not share it, which is part of why pooling both seeds misranks. Removing each seed's mean does not rescue the ranking either: Spearman {r['published_2p403um']['spearman_seed_demeaned_n14']:+.2f} on 2.403 µm, {r['eligible_9p366um']['spearman_seed_demeaned_n14']:+.2f} on 9.366 µm.

So it is not a seed offset. Most of the spread is variation the two seeds do not share, which supports your restatement in spirit: differences of this size between checkpoints do not repeat across runs, so no validation set would rank them reliably. Two seeds make this a weak estimate, as you say.
"""
(ROOT / "results/ys/REPLY_1867_pedro.md").write_text(text)
print(text)
