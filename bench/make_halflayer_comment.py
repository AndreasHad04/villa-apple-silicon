"""Generate the PR 1813 comment. Every number comes from results/halflayer.json."""
import json, pathlib
R = pathlib.Path(__file__).resolve().parent.parent
d = json.load(open(R / "results" / "halflayer.json"))
rows = d["rows"]
by = {}
for x in rows:
    by.setdefault(x["scroll"], []).append(x)

tbl = ["| scroll | crop y,x | r at `START_LAYER=23` | r at `START_LAYER=24` | 23 minus 24 |",
       "|---|---|---:|---:|---:|"]
for s in sorted(by):
    for x in sorted(by[s], key=lambda z: z["crop"]):
        tbl.append(f"| {s} | {x['crop'][0]}, {x['crop'][1]} | {x['r23']:.4f} | "
                   f"{x['r24']:.4f} | {x['diff']:+.4f} |")

p0139 = sorted(z["diff"] for z in by.get("PHerc0139", []))
w0139 = sum(1 for z in p0139 if z > 0)

txt = f"""Two things, one about which PR should carry this and one measurement that answers the half layer question in #1765.

**On #1766 being a draft.** That changes my position, so I am stating the new one plainly rather than leaving the old one standing. My comment above said I was happy for this PR to be closed in favour of #1766. I wrote that believing #1766 was the merge candidate. A draft cannot be merged, so as things stand this PR is the only one of the two that a maintainer can act on today. I am therefore withdrawing the offer to close it and leaving it open and ready. If #1766 comes out of draft I have no objection to it being the one that merges and this one being closed; the point is the README line, not whose PR fixes it.

Priority is unchanged and I want it on the record in this comment rather than only in an earlier one: flummoxjr published the centred window with r = 0.9999997 on 2026-09-03, @kadenpool filed #1765 and #1766 on 2026-09-11, and this PR is from 2026-09-16 and was reached independently without knowledge of either.

**On 23 against 24, and first a correction to my own evidence.** #1765 now records that 23 beats 24 on all 8 text windows there, median match 0.905 against 0.874. I went back to check what my own sweeps actually said about that, and they said nothing: of the 15 sweep grids behind this PR, **14 contained `START_LAYER=24` and not 23, and the one that contained 23 did not contain 24**. So "best `START_LAYER` 21 to 25" in the PR body never separated the two, and any impression that my data preferred 24 was a property of the grid I chose, not a measurement.

So I ran the paired comparison: the same 15 crops, 4 scrolls, each crop scored at 23 and at 24 and nothing else changed. 1024 px crops, `TILE_SIZE=256`, `STRIDE=128`, `resnet3d-152-3d-decoder`, `r152_3ddec_v2_l5_epoch13.ckpt`, scored as Pearson against the published `new_canon_autoresearch_recipe` map of the same segment.

{chr(10).join(tbl)}

**23 wins {d['wins_23']}, 24 wins {d['wins_24']}, no ties. Exact two sided sign test p = {d['sign_test_p']:.4f}.** Median paired difference {d['median_diff']:+.4f}, mean {d['mean_diff']:+.4f}, range {d['min_diff']:+.4f} to {d['max_diff']:+.4f}. Worst correlation against a shuffled copy of the reference across every one of these runs: {d['worst_shuffle_floor']:.4f}. Every scroll produces both signs.

**The scale is the part I would act on.** Over the same 15 crops, median r is {d['median_r_at_1']:.4f} at the README's `START_LAYER=1` and {d['median_r_at_best_half']:.4f} at the better of 23 and 24. So the window fix is worth {d['window_fix_worth']:+.4f} in median r and the half layer is worth {d['median_diff']:+.4f} at p = {d['sign_test_p']:.4f}, a factor of about {abs(d['window_fix_worth'])/abs(d['median_diff']):.0f}. Crop to crop, r ranges {d['r_min']:.4f} to {d['r_max']:.4f}, a spread of {d['r_max']-d['r_min']:.4f}, against a largest single crop half layer effect of {max(abs(x['diff']) for x in rows):.4f}.

**Where this disagrees with #1765, stated as a disagreement rather than smoothed over.** On PHerc0139, the scroll #1765 measured, my three crops split {w0139} to {len(p0139)-w0139} for 23 ({', '.join(f'{v:+.4f}' for v in p0139)}), so I do not reproduce 8 of 8. Two differences could explain it and I have not tested either: crop size, 1024 px here against 2048 px there, and crop selection, the single most ink rich window per segment here against random positions filtered on published ink fraction there. 8 of 8 is itself unlikely under a coin flip, so I am not calling that result noise; I am saying it does not hold when the same comparison is run across four scrolls.

**What I think follows for the README, and it is the cheap conclusion.** Both PRs already write 23, which is what `(109 - 62) // 2` gives. On this evidence there is no reason to change that and no reason for either PR to argue the half layer, because the floor formula is already defensible and the remaining difference is not resolvable at this crop size. If anything is worth adding to the README it is the rule rather than the constant, which #1766 already does in its Tips line.

Raw per crop numbers are in `results/halflayer.json` and the 15 per crop files `results/window_generalise_hl_*.json` in https://github.com/AndreasHad04/villa-apple-silicon , MIT, free to lift into #1766.

One more correction while I am here: one row of my published `results/window_tally.json` carried the string `20240304141531-w013 (offset sweep)` in its `seg` field, which is a label and not a segment id, so that row could not be re-run programmatically. It is now the real id, `20240304141531-w013_20240304141531_flatboi`, with the previous file kept beside it.
"""
out = R / "results" / "PR1813_COMMENT_halflayer.md"
out.write_text(txt)
bad = sum(txt.count(c) for c in (chr(0x2014), chr(0x2013)))
low = txt.lower()
ai = sum(low.count(w) for w in ("claude", "chatgpt", "copilot", "llm", "generated with"))
print(f"wrote {out}  {len(txt)} chars  dashes={bad}  ai_mentions={ai}")
assert bad == 0 and ai == 0, "hygiene fail"
