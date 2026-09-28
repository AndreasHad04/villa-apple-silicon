Done from the published score files, no new inference ([`bench/seed_decomp.py`](https://github.com/AndreasHad04/villa-apple-silicon/blob/main/bench/seed_decomp.py), output `results/seed_decomp.json`). Two seeds by seven steps, split into a seed part, a step part and what is left:

| scores | seed part | step part | left over | seed43 minus seed42 |
|---|---|---|---|---|
| 2.403 µm volumes, held out | 4.7% | 29.1% | 66.2% | -0.0059 |
| 9.366 µm eligible volumes, held out | 10.9% | 48.2% | 40.9% | -0.0137 |
| 4.681 µm renders, held out | 0.2% | 60.8% | 39.0% | -0.0016 |
| in-distribution (C1) | 49.7% | 17.4% | 32.9% | +0.0723 |

On the held-out scores the seed gap is small, and the two seeds do not agree on which steps are better: their step profiles correlate -0.63 on 2.403 µm and +0.09 on 9.366 µm. The large seed gap is in the in-distribution score, seed43 +0.072, and the held-out scores do not share it, which is part of why pooling both seeds misranks. Removing each seed's mean does not rescue the ranking either: Spearman +0.15 on 2.403 µm, -0.02 on 9.366 µm.

So it is not a seed offset. Most of the spread is variation the two seeds do not share, which supports your restatement in spirit: differences of this size between checkpoints do not repeat across runs, so no validation set would rank them reliably. Two seeds make this a weak estimate, as you say.
