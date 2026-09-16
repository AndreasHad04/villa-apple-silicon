"""One table over every window measurement, generated from the raw JSON."""
import json, glob, pathlib, statistics as st
R = pathlib.Path(__file__).resolve().parent.parent
rows = []
for f in sorted(glob.glob(str(R/"results"/"window_generalise_*.json"))):
    d = json.load(open(f)); v = d["verdict"]
    if v.get("readme_start_layer_1_pearson") is None: continue
    rows.append({"scroll": d.get("scroll"), "seg": d.get("seg"), "crop": d.get("crop"),
                 "layers": d.get("layers"), "r_at_1": v["readme_start_layer_1_pearson"],
                 "best_start_layer": v["best_start_layer"], "r_at_best": v["best_pearson"],
                 "shuffle_floor": v["max_abs_shuffle_floor"]})
o = json.load(open(R/"results"/"offset_vs_reference_prof.json"))
rows.append({"scroll": "PHerc1667", "seg": "20240304141531-w013 (offset sweep)",
             "crop": o["crop"], "layers": 109,
             "r_at_1": next(r["pearson_vs_reference"] for r in o["rows"] if r["z_window"][0] == 1),
             "best_start_layer": o["verdict"]["best_z_window"][0],
             "r_at_best": o["verdict"]["best_pearson"],
             "shuffle_floor": o["verdict"]["max_abs_shuffle_floor"]})
b = [r["best_start_layer"] for r in rows]; o1 = [r["r_at_1"] for r in rows]; ob = [r["r_at_best"] for r in rows]
summ = {"n": len(rows), "best_min": min(b), "best_max": max(b), "best_median": st.median(b),
        "best_mode": max(set(b), key=b.count), "best_mode_count": b.count(max(set(b), key=b.count)),
        "r_at_1_min": round(min(o1), 4), "r_at_1_max": round(max(o1), 4), "r_at_1_median": round(st.median(o1), 4),
        "r_at_best_min": round(min(ob), 4), "r_at_best_max": round(max(ob), 4), "r_at_best_median": round(st.median(ob), 4),
        "best_beats_one": sum(1 for r in rows if r["r_at_best"] > r["r_at_1"]),
        "max_shuffle_floor": round(max(r["shuffle_floor"] for r in rows), 4),
        "scrolls": sorted({r["scroll"] for r in rows})}
(R/"results"/"window_tally.json").write_text(json.dumps({"rows": rows, "summary": summ}, indent=2))
print(json.dumps(summ, indent=2))
