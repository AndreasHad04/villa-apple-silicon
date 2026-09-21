"""START_LAYER 23 against 24 on the SAME crop, every crop in the tally.

Our published sweeps never separated these two. 14 of the 15 grids contained
24 and not 23; the one that contained 23 did not contain 24. So "24 is the
empirical optimum" was a property of the grid, not a measurement. kadenpool
measured the half layer on PHerc0139 and found 23 beats 24; this runs the
paired comparison on four scrolls.

Reuses ops/window_generalise.py verbatim as a subprocess rather than
reimplementing the inference path, and hardlinks the already downloaded
reference so nothing re-downloads and no existing result file is overwritten.
"""
import json, glob, pathlib, subprocess, sys, time
R = pathlib.Path(__file__).resolve().parent.parent
PY_ = str(R / "env" / "bin" / "python3")

tal = json.load(open(R / "results" / "window_tally.json"))["rows"]
gen = {}
for f in glob.glob(str(R / "results" / "window_generalise_*.json")):
    d = json.load(open(f))
    gen[(d["scroll"], d["seg"], tuple(d["crop"]))] = pathlib.Path(f).stem.replace("window_generalise_", "")

jobs = []
for r in tal:
    src = gen.get((r["scroll"], r["seg"], tuple(r["crop"])))
    tag = f"hl_{src}" if src else "hl_r1667_0"
    # the unmapped crop is on the same segment as r1667_1..3, so its reference is theirs
    reftag = src if src else "r1667_1"
    jobs.append(dict(scroll=r["scroll"], seg=r["seg"], crop=r["crop"], tag=tag, reftag=reftag))

t0 = time.time()
done = 0
for i, j in enumerate(jobs, 1):
    out = R / "results" / f"window_generalise_{j['tag']}.json"
    if out.exists():
        print(f"[{i}/{len(jobs)}] {j['tag']}: already done, skipping", flush=True)
        done += 1
        continue
    src = R / "reference" / f"ref_{j['reftag']}.tif"
    dst = R / "reference" / f"ref_{j['tag']}.tif"
    if not dst.exists():
        if not src.exists():
            print(f"[{i}/{len(jobs)}] {j['tag']}: NO cached reference {src.name}, will download", flush=True)
        else:
            dst.hardlink_to(src)
    sib = next((c for c in (pathlib.Path(__file__).with_name("window_generalise.py"),
                        R / "ops" / "window_generalise.py") if c.exists()), None)
    assert sib, "window_generalise.py not found beside this script or in ops/"
    cmd = [PY_, str(sib),
           "--scroll", j["scroll"], "--seg", j["seg"],
           "--y0", str(j["crop"][0]), "--x0", str(j["crop"][1]),
           "--size", str(j["crop"][2]), "--offsets=22,23", "--tag", j["tag"]]
    ts = time.time()
    p = subprocess.run(cmd, cwd=str(R), capture_output=True, text=True)
    if p.returncode != 0:
        print(f"[{i}/{len(jobs)}] {j['tag']}: FAILED rc={p.returncode}", flush=True)
        print(p.stdout[-2000:], p.stderr[-2000:], flush=True)
        continue
    done += 1
    d = json.load(open(out))
    rows = {r["start_layer"]: r["pearson_vs_reference"] for r in d["rows"]}
    el = time.time() - ts
    per = (time.time() - t0) / max(done, 1)
    left = (len(jobs) - i) * per
    print(f"[{i}/{len(jobs)}] {j['tag']:18s} {j['scroll']:12s} "
          f"sl23={rows.get(23)} sl24={rows.get(24)} "
          f"({el:.0f}s, ETA {left/60:.1f} min)", flush=True)
print(f"DONE {done}/{len(jobs)} in {(time.time()-t0)/60:.1f} min", flush=True)
