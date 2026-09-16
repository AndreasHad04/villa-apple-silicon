"""Agreement with villa's published prediction as a function of START_LAYER,
across every scroll tested. One line per scroll, plus the shuffled-reference
floor so the reader can see the null.
"""
import json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = pathlib.Path(__file__).resolve().parent.parent
series = []
# PHerc1667 comes from the coarse+fine offset sweeps
o = json.loads((R/"results"/"offset_vs_reference_prof.json").read_text())
series.append(("PHerc. 1667  seg 20240304141531",
               [(r["z_window"][0], r["pearson_vs_reference"]) for r in o["rows"]],
               o["verdict"]["max_abs_shuffle_floor"]))
for tag, label in (("g0139", "PHerc. 0139  seg 20250108000000"),
                   ("gparis", "PHercParis4  seg 20230702185753"),
                   ("g0814", "PHerc. 0814")):
    f = R/"results"/f"window_generalise_{tag}.json"
    if not f.exists(): continue
    d = json.loads(f.read_text())
    series.append((label + f"  ({d['layers']} layers)",
                   [(r["start_layer"], r["pearson_vs_reference"]) for r in d["rows"]],
                   d["verdict"]["max_abs_shuffle_floor"]))

fig, ax = plt.subplots(figsize=(10.5, 6.2))
cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
for i, (lab, pts, floor) in enumerate(series):
    pts = sorted(pts)
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    ax.plot(x, y, "o-", color=cols[i % 4], label=lab, lw=2, ms=4)
    b = max(pts, key=lambda p: p[1])
    ax.plot([b[0]], [b[1]], "*", color=cols[i % 4], ms=18, zorder=5)
ax.axvline(1, color="0.3", ls="--", lw=1.4)
ax.text(2.2, 0.60, "START_LAYER=1\n(the value the README suggests)",
        fontsize=10, color="0.25", va="bottom")
ax.axvspan(23, 25, color="0.85", zorder=0)
ax.text(24, 1.005, "23 to 25", ha="center", fontsize=10, color="0.3")
ax.axhline(0, color="0.6", lw=1)
ax.text(46, 0.015, "shuffled-reference floor (|r| < 0.0014)", fontsize=9,
        color="0.45", ha="right", va="bottom")
ax.set_xlabel("START_LAYER  (62-channel window, END_LAYER = START_LAYER + 62)", fontsize=11)
ax.set_ylabel("Pearson r against villa's published prediction", fontsize=11)
ax.set_title("The documented inference window does not reproduce villa's own production output",
             fontsize=12.5, pad=14)
ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
ax.grid(alpha=0.25); ax.set_ylim(-0.05, 1.04)
fig.tight_layout()
fig.savefig(R/"public"/"figures"/"start_layer_vs_reference.png", dpi=115, bbox_inches="tight")
print("wrote start_layer_vs_reference.png with", len(series), "scrolls")
for lab, pts, _ in series:
    b = max(pts, key=lambda p: p[1]); one = [p[1] for p in pts if p[0] == 1]
    print(f"  {lab:46s} best START_LAYER {b[0]:3d} r={b[1]:.4f}"
          + (f"   at START_LAYER 1: r={one[0]:.4f}" if one else ""))
