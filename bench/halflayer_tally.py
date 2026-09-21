"""23 against 24 on the same crop: the paired table and the sign test.

Generated, never typed. Every number in the PR comment comes from here.
"""
import json, glob, pathlib, math
R = pathlib.Path(__file__).resolve().parent.parent
rows = []
for f in sorted(glob.glob(str(R / "results" / "window_generalise_hl_*.json"))):
    d = json.load(open(f))
    r = {x["start_layer"]: x["pearson_vs_reference"] for x in d["rows"]}
    sh = max(abs(x["pearson_vs_shuffled"]) for x in d["rows"])
    if 23 not in r or 24 not in r:
        print(f"SKIP {f}: missing a window {sorted(r)}")
        continue
    rows.append(dict(tag=pathlib.Path(f).stem.replace("window_generalise_hl_", ""),
                     scroll=d["scroll"], seg=d["seg"], crop=d["crop"],
                     r23=r[23], r24=r[24], diff=round(r[23] - r[24], 4), shuffle=sh))
rows.sort(key=lambda x: (x["scroll"], x["crop"]))

w23 = [x for x in rows if x["diff"] > 0]
w24 = [x for x in rows if x["diff"] < 0]
tie = [x for x in rows if x["diff"] == 0]
diffs = sorted(x["diff"] for x in rows)
n = len(rows)
mean = sum(diffs) / n
med = diffs[n // 2] if n % 2 else (diffs[n // 2 - 1] + diffs[n // 2]) / 2

# exact two sided sign test on the non tie crops
m = len(w23) + len(w24)
k = min(len(w23), len(w24))
p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0

print(f"\n{'crop':34s} {'scroll':13s} {'r@23':>8s} {'r@24':>8s} {'23 - 24':>9s}  winner")
for x in rows:
    print(f"{x['tag'][:20]:20s} {str(x['crop'][:2]):13s} {x['scroll']:13s} "
          f"{x['r23']:8.4f} {x['r24']:8.4f} {x['diff']:+9.4f}  "
          f"{'23' if x['diff']>0 else '24' if x['diff']<0 else 'tie'}")

print(f"\nn = {n} paired crops on {len(set(x['scroll'] for x in rows))} scrolls")
print(f"23 wins {len(w23)}, 24 wins {len(w24)}, ties {len(tie)}")
print(f"paired difference (23 minus 24): mean {mean:+.4f}, median {med:+.4f}, "
      f"min {diffs[0]:+.4f}, max {diffs[-1]:+.4f}")
print(f"largest |difference| {max(abs(d) for d in diffs):.4f}; "
      f"worst shuffle floor {max(x['shuffle'] for x in rows):.4f}")
print(f"exact two sided sign test on {m} non tie crops: p = {p:.4f}")

# spread BETWEEN crops, for scale: is the half layer big or small next to it?
allr = [x["r23"] for x in rows] + [x["r24"] for x in rows]
print(f"crop to crop spread of r: {min(allr):.4f} to {max(allr):.4f} "
      f"= {max(allr)-min(allr):.4f}, against a largest half layer effect of "
      f"{max(abs(d) for d in diffs):.4f}")

per = {}
for x in rows:
    per.setdefault(x["scroll"], []).append(x["diff"])
print("\nper scroll, paired difference 23 minus 24:")
for s, v in sorted(per.items()):
    print(f"  {s:13s} n={len(v)}  {', '.join(f'{d:+.4f}' for d in sorted(v))}  "
          f"mean {sum(v)/len(v):+.4f}")

# SCALE: what the window fix is worth, against what the half layer is worth.
# r at START_LAYER=1 comes from the existing tally, joined on the crop.
tal = {(x["scroll"], tuple(x["crop"])): x["r_at_1"]
       for x in json.load(open(R / "results" / "window_tally.json"))["rows"]}
r1 = sorted(tal[(x["scroll"], tuple(x["crop"]))] for x in rows)
best = sorted(max(x["r23"], x["r24"]) for x in rows)
mid = lambda v: v[len(v)//2] if len(v) % 2 else (v[len(v)//2-1]+v[len(v)//2])/2
m1, mb = mid(r1), mid(best)
print(f"\nSCALE, median r over the same {n} crops:")
print(f"  START_LAYER=1 (README)      {m1:.4f}")
print(f"  the better of 23 and 24     {mb:.4f}")
print(f"  the window fix is worth     {mb-m1:+.4f}")
print(f"  the half layer is worth     {med:+.4f} (median), p = {p:.4f}")
print(f"  ratio                       {abs(mb-m1)/max(abs(med),1e-9):.0f}x")

out = dict(rows=rows, n=n, wins_23=len(w23), wins_24=len(w24), ties=len(tie),
           median_r_at_1=round(m1, 4), median_r_at_best_half=round(mb, 4),
           window_fix_worth=round(mb - m1, 4),
           mean_diff=round(mean, 4), median_diff=round(med, 4),
           min_diff=diffs[0], max_diff=diffs[-1],
           sign_test_p=round(p, 4), n_nontie=m,
           worst_shuffle_floor=round(max(x["shuffle"] for x in rows), 4),
           r_min=round(min(allr), 4), r_max=round(max(allr), 4),
           scrolls=sorted(set(x["scroll"] for x in rows)))
(R / "results" / "halflayer.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote results/halflayer.json")
