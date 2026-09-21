"""Score every reference free selector against the pre-registered endpoint.

PRIMARY, fixed before the run: median REGRET in Pearson r of S4 across crops.
    S4 WORKS   median regret <= 0.02 AND strictly below S1's median regret
Controls C1 (random), C2 (README window), C3 (shuffle floor) are reported
whatever they say.
"""
import json, glob, pathlib, statistics as st
import numpy as np
R = pathlib.Path(__file__).resolve().parent.parent
BAR = 0.02
# name -> (field, direction) ; direction +1 maximise, -1 minimise
SEL = {"S1_separation": ("S1_separation", +1),
       "S2_frac_mushy": ("S2_frac_mushy", -1),
       "S3_pred_std": ("S3_pred_std", +1),
       "S4_null_collapse": ("S4_null_collapse", -1),
       "S5_null_margin": ("S5_null_margin", +1)}

crops = [json.load(open(f)) for f in sorted(glob.glob(str(R/"results"/"window_selector_*.json")))]
if not crops:
    raise SystemExit("no window_selector_*.json yet")

reg = {k: [] for k in SEL}; reg["C1_random"] = []; reg["C2_readme_sl1"] = []
picks = {k: [] for k in SEL}
per_crop = []
rng = np.random.default_rng(0)
floor = 0.0
for c in crops:
    rows = c["rows"]
    r = {x["start_layer"]: x["pearson_vs_reference"] for x in rows}
    floor = max(floor, max(abs(x["pearson_vs_shuffled"]) for x in rows))
    best_sl = max(r, key=r.get); best = r[best_sl]
    rec = {"scroll": c["scroll"], "crop": c["crop"][:2], "best_sl": best_sl, "r_best": best}
    for name, (fld, d) in SEL.items():
        pick = max(rows, key=lambda x: d * x[fld])["start_layer"]
        reg[name].append(round(best - r[pick], 4)); picks[name].append(pick)
        rec[name] = pick
    grid = list(r)
    draws = rng.choice(grid, size=1000)
    reg["C1_random"].append(round(best - float(np.mean([r[int(g)] for g in draws])), 4))
    reg["C2_readme_sl1"].append(round(best - r[1], 4))
    per_crop.append(rec)

print(f"{len(crops)} crops on {len(set(c['scroll'] for c in crops))} scrolls, "
      f"grid {crops[0]['grid']}")
print(f"\nC3 worst |Pearson vs SHUFFLED reference| across every window: {floor:.4f}"
      f"   {'PASS' if floor < 0.01 else 'FAIL, ground truth suspect'}")

print(f"\n{'scroll':13s} {'crop':16s} {'best':>5s} {'r_best':>7s}  " +
      "  ".join(f"{k.split('_')[0]:>4s}" for k in SEL))
for x in per_crop:
    print(f"{x['scroll']:13s} {str(x['crop']):16s} {x['best_sl']:5d} {x['r_best']:7.4f}  " +
          "  ".join(f"{x[k]:4d}" for k in SEL))

print(f"\n{'selector':18s} {'median regret':>14s} {'mean':>8s} {'max':>8s}  verdict")
order = list(SEL) + ["C1_random", "C2_readme_sl1"]
med = {k: st.median(reg[k]) for k in order}
for k in order:
    v = reg[k]
    note = ""
    if k == "S4_null_collapse":
        ok = med[k] <= BAR and med[k] < med["S1_separation"]
        note = "PRIMARY: " + ("WORKS" if ok else "FAILS")
    print(f"{k:18s} {med[k]:14.4f} {st.mean(v):8.4f} {max(v):8.4f}  {note}")

best_sel = min(SEL, key=lambda k: med[k])
print(f"\nbest selector by median regret: {best_sel} at {med[best_sel]:.4f}")
print(f"C1 random median regret {med['C1_random']:.4f}; any selector at or above "
      f"that is worthless")
print(f"C2 the README window START_LAYER=1 costs {med['C2_readme_sl1']:.4f} in median r")
for k in SEL:
    if med[k] >= med["C1_random"]:
        print(f"  WORTHLESS: {k} does not beat random")

out = dict(n_crops=len(crops), grid=crops[0]["grid"], bar=BAR,
           shuffle_floor=round(floor, 4), per_crop=per_crop,
           median_regret={k: round(med[k], 4) for k in order},
           mean_regret={k: round(st.mean(reg[k]), 4) for k in order},
           max_regret={k: round(max(reg[k]), 4) for k in order},
           regret=reg, picks=picks,
           primary_S4_works=bool(med["S4_null_collapse"] <= BAR and
                                 med["S4_null_collapse"] < med["S1_separation"]),
           best_selector=best_sel)
(R/"results"/"window_selector_tally.json").write_text(json.dumps(out, indent=2))
print("\nwrote results/window_selector_tally.json")
