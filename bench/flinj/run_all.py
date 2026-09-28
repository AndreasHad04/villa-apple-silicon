"""ARM FLINJ phase 1 as amended (AMENDMENT_1.md, AMENDMENT_2.md): the full run.

Gates, all checked before any inference (the three registration hashes are checked in common.py):
  1. metric_checks.json: M1 and M2 PASS, and every layout re-planned here equals layouts.json (A1).
  2. b_precheck.json: b_host (C1) is defined at EVERY glyph pixel of every host and stroke width that has a layout.
     A host excluded by C2 runs only base_r1, base_r2 and its S4 widths (amendment 4, D4).
  3. smoke_full.json C0 passed (full-size determinism) and smoke_pair.json found max |D| = 0 on the far background.
  4. On AC power (pmset), and armguard check --need <N, measured> passes before each host (polled 60 s, 30 min).

Units, each one inference of one FULL host crop, primary checkpoint unless named, forward direction:
  base_r1, base_r2                  the untouched host, twice (C0; reference for C3, S1, S5, S6)
  b1_ink, b2_non                    B1 ink transplant and B2 non-ink transplant (primary, B3)
  c1_shuffle, c2_flat               C1 / C2 on the transplant: the ink residual's planes permuted per patch / its mean
  c3_clean                          B4: donor mean depth profile x soft mask at +20 grey levels (no clip bound, D2)
  s4_w<w>_ink, s4_w<w>_non          S4 at 0.25, 0.5, 0.7 mm on the B1 / B2 transplants, each width on its own layout (D3)
  s5_raw, s6_profile                S5 raw additive residual and S6 mean profile at a = 0.5 (scored against the host)
  s3_ink_<ck>, s3_non_<ck>          S3: the 13 other checkpoints on the B1 / B2 transplants
Resumable per unit (results/flinj/units/<host>/<unit>.json marks it done; temp files renamed). Inputs are rebuilt
deterministically when missing (sha256 checked) and deleted when no pending unit needs them.

    /Users/andreashad04/money/vesuv/env/bin/python3 /Users/andreashad04/money/vesuv/ops/flinj/run_all.py plan
    /Users/andreashad04/money/vesuv/env/bin/python3 /Users/andreashad04/money/vesuv/ops/flinj/run_all.py run [--hosts h ...] [--need GB]
"""
import argparse, json, math, resource, shutil, subprocess, sys, time
import numpy as np, tifffile, zarr
from common import (DATA, HOSTS, PRIMARY_CKPT, REG, RES, all_ckpts, armguard, atomic_json, banner, cap_px, ckpt_key,
                    host_seed, infer, load_host, load_volume, mm_px, sha_array, write_zarr)
import glyphs as GL, residuals as RS
from inject import c3_scale, inject, transplant

UNITS, PRED = RES / "units", RES / "pred"
S3_CKPTS = all_ckpts()[1:]
W0 = REG["stroke_mm"]
STREAM = dict(ink=10, non=20)


def units(h):
    """Amendment 4: a host excluded from the primary (C2) keeps base_r1 / base_r2 (C0, S1) and every S4 width whose own
    layout fits (D3, D4); an S4 width with no layout of 4 letters is skipped on that host."""
    excl = bool(json.load(open(RES / "layouts.json"))[h].get("excluded")); L4 = json.load(open(RES / "layouts_s4.json"))[h]
    P = PRIMARY_CKPT
    tp = lambda lib, mode="real", w=W0, ck=P: dict(kind="transplant", lib=lib, mode=mode, stroke=w, ckpt=ck)
    u = [("base_r1", dict(kind="base", ckpt=P)), ("base_r2", dict(kind="base", ckpt=P))]
    if not excl:
        u += [("b1_ink", tp("ink")), ("b2_non", tp("non")), ("c1_shuffle", tp("ink", "shuffle")), ("c2_flat", tp("ink", "flat")),
              ("c3_clean", dict(kind="additive", res="profile", mode="profile", c3=True, stroke=W0, ckpt=P))]
    for w in REG["s4_stroke_mm"]:
        if L4[f"{w:g}"] is not None: u += [(f"s4_w{w:g}_ink", tp("ink", w=w)), (f"s4_w{w:g}_non", tp("non", w=w))]
    if excl: return u
    u += [("s5_raw", dict(kind="additive", res="raw", mode="real", a=REG["s56_a"], stroke=W0, ckpt=P)),
          ("s6_profile", dict(kind="additive", res="profile", mode="profile", a=REG["s56_a"], stroke=W0, ckpt=P))]
    for ck in S3_CKPTS:
        u += [(f"s3_ink_{ckpt_key(ck)}", tp("ink", ck=ck)), (f"s3_non_{ckpt_key(ck)}", tp("non", ck=ck))]
    return u


def done(h, name):
    return (UNITS / h / f"{name}.json").exists()


def in_key(spec):
    if spec["kind"] == "transplant": return f"tp_{spec['lib']}_{spec['mode']}_w{spec['stroke']:g}"
    return "c3" if spec.get("c3") else f"add_{spec['res']}_a{spec['a']:g}"


def mpx(h):
    _, H, W = zarr.open_array(str(HOSTS[h]["ct"]), mode="r").shape; return H * W / 1e6


def timing():
    """(inference s/Mpx, input build + zarr write s/Mpx), both measured by the full-size smoke pair on ag174."""
    s = json.load(open(RES / "smoke_pair" / "smoke_pair.json"))
    build = float(np.median([v["build_s"] + v["zarr_write_s"] for v in s["inputs"].values()]))
    return s["seconds_per_inference"]["median"] / s["mpx"], build / s["mpx"]


def need_gb():
    foot = int([l for l in (RES / "smoke_pair" / "time_l.txt").read_text().splitlines() if "peak memory footprint" in l][0].split()[0]) / 1e9
    extra = max(0.0, max(mpx(h) for h in HOSTS) - json.load(open(RES / "smoke_pair" / "smoke_pair.json"))["mpx"]) * 1e6 * 21 * 8 / 1e9
    return int(math.ceil(foot + extra + 0.5)), foot, extra


def plan():
    spm, bpm = timing(); total = left = 0; eta = 0.0
    for h in HOSTS:
        us = units(h); rem = [n for n, _ in us if not done(h, n)]; m = mpx(h); total += len(us); left += len(rem)
        eta += len(rem) * spm * m + len({in_key(s) for n, s in us if s["kind"] != "base" and not done(h, n)}) * bpm * m
        print(f"{h:8s} units {len(us):3d}, remaining {len(rem):3d}, crop {m:.2f} Mpx")
    print(f"TOTAL {total} units, {left} remaining")
    print(f"ETA from the full-size smoke pair ({spm:.2f} s/Mpx per whole-crop inference incl. model load, plus {bpm:.2f} s/Mpx per distinct input built): "
          f"{eta / 60:.0f} min, finish about {time.strftime('%H:%M', time.localtime(time.time() + eta))} if started now")
    n, foot, extra = need_gb(); print(f"armguard need {n} GB = smoke-pair peak footprint {foot:.2f} GB + largest-host extra {extra:.2f} GB + 0.5")
    return total, left, eta


def gates():
    p = []
    M = json.load(open(RES / "metric_checks.json"))
    if not (M["M1_pass"] and M["M2_pass"]): p.append("M1 or M2 failed")
    B = json.load(open(RES / "b_precheck.json"))
    if B["host_bg_registered"] != REG["host_bg"]: p.append("b_precheck.json was made under another host background: re-run it")
    cov = {h: {w: c[REG["host_bg"]] for w, c in r["coverage"].items()} for h, r in B["hosts"].items()}  # C2-excluded hosts are absent
    bad = {h: {w: round(v, 4) for w, v in c.items() if v < 1.0} for h, c in cov.items() if any(v < 1.0 for v in c.values())}
    if bad: p.append(f"B1 undefined: b_host ({REG['host_bg']}) covers less than every glyph pixel: {bad}")
    s = json.load(open(RES / "smoke_full" / "smoke_full.json")) if (RES / "smoke_full" / "smoke_full.json").exists() else {}
    if not s.get("C0", {}).get("pass_"): p.append("full-size C0 determinism missing or failed")
    q = json.load(open(RES / "smoke_pair" / "smoke_pair.json")) if (RES / "smoke_pair" / "smoke_pair.json").exists() else {}
    if q.get("max_abs_D_far") != 0: p.append("smoke pair: D on the far background missing or not exactly 0")
    if "AC Power" not in subprocess.run(["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True).stdout: p.append("not on AC power")
    return p


def build_input(h, spec, ctx):
    g = ctx["geo"](spec["stroke"]); I = ctx["I"]; seed = host_seed(h)
    if spec["kind"] == "transplant":
        return transplant(I, ctx["bg"], g["binary"], ctx["tlib"][spec["lib"]], seed, spec["mode"], STREAM[spec["lib"]])
    if spec.get("c3"):
        c3 = c3_scale(I, g["G"], ctx["profiles"], g["binary"]); v, info = inject(I, g["G"], c3["scale"], "profile", ctx["profiles"], seed, g["binary"])
        return v, dict(info, a=c3["scale"])
    lib = ctx["raw"] if spec["res"] == "raw" else ctx["profiles"]
    v, info = inject(I, g["G"], spec["a"], spec["mode"], lib, seed, g["binary"]); return v, dict(info, a=spec["a"])


def run_unit(h, name, spec, ctx):
    ck = spec["ckpt"]; out = PRED / h / f"{name}.tif"
    rec = dict(host=h, unit=name, ckpt=ckpt_key(ck), direction="forward", spec={k: v for k, v in spec.items() if k != "ckpt"})
    if spec["kind"] == "base":
        ct = HOSTS[h]["ct"]
    else:
        ct = DATA / h / f"in_{in_key(spec)}" / "ct.zarr"; side = ct.parent / "input.json"
        if ct.exists() and side.exists():
            assert sha_array(load_volume(ct)) == json.load(open(side))["sha256"], f"{ct} does not match its recorded sha256"
        else:
            t = time.time(); v, info = build_input(h, spec, ctx)
            write_zarr(ct, v); atomic_json(side, dict(info, sha256=sha_array(v), key=in_key(spec), build_s=time.time() - t))
        rec["input"] = json.load(open(side))
    secs = infer(ct, ck, out)
    rec.update(secs=secs, input_path=str(ct), pred=str(out), pred_sha256=sha_array(tifffile.imread(out)),
               maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, when=time.strftime("%Y-%m-%dT%H:%M:%S"))
    atomic_json(UNITS / h / f"{name}.json", rec)
    print(f"DONE {h} {name} {secs:.1f}s" + (f" clip {rec['input']['clip_frac']:.4f}" if "input" in rec else ""), flush=True)


def cleanup(h):
    keep = {in_key(s) for n, s in units(h) if s["kind"] != "base" and not done(h, n)}
    for d in (DATA / h).glob("in_*"):
        if d.name[3:] not in keep: shutil.rmtree(d, ignore_errors=True)


def run_host(h, need):
    ok, msg = armguard(need)
    if not ok: print(f"REFUSED {h}: armguard {msg}", flush=True); return False
    I, ink, support, valid = load_host(h); cap = cap_px(h)
    lay = GL.plan_layout(valid, cap, mm_px(h, W0), host_seed(h), border=REG["border_px"], ink=ink)
    lays = {W0: lay} | {w: GL.plan_layout(valid, cap, mm_px(h, w), host_seed(h), border=REG["border_px"], ink=ink) for w in REG["s4_stroke_mm"]}
    ref, ref4 = json.load(open(RES / "layouts.json"))[h], json.load(open(RES / "layouts_s4.json"))[h]
    for w, l in lays.items():
        r_ = ref if w == W0 else ref4[f"{w:g}"]
        if l["axis"] is None: assert r_ is None or r_.get("excluded"), f"{h} {w}: layout differs from the recorded one"
        else: assert (l["letters"], l["origin"], l["n"]) == (r_["letters"], r_["origin"], r_["n"]), f"{h} {w}: layout differs from the recorded one"
    if lay["axis"] is None: print(f"EXCLUDED {h} from the primary (C2); running its base and S4 units only", flush=True)
    assert REG["host_bg"] == "valid_not_ink"; bg = valid & ~ink  # amendment 3, C1
    cache = {}

    def geo(w):
        if w not in cache: cache[w] = GL.build(valid, cap, mm_px(h, w), lays[w], sigma=REG["soft_sigma_px"])  # D3: each width its own layout
        return cache[w]
    ctx = dict(I=I, bg=bg, geo=geo, tlib=dict(ink=RS.host_transplant_libraries(h, "ink"), non=RS.host_transplant_libraries(h, "non")),
               raw=RS.host_libraries(h, "raw"), profiles=RS.host_profiles(h))
    for name, spec in units(h):
        if done(h, name): continue
        run_unit(h, name, spec, ctx); cleanup(h)
    cleanup(h); return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["plan", "run"]); ap.add_argument("--hosts", nargs="*", default=list(HOSTS))
    ap.add_argument("--need", type=int); a = ap.parse_args(); banner(__file__)
    plan()
    if a.cmd == "plan": return 0
    problems = gates()
    if problems:
        for p in problems: print("REFUSED:", p, flush=True)
        return 2
    need = a.need or need_gb()[0]
    for h in a.hosts:
        if not run_host(h, need): return 1
    plan(); return 0


if __name__ == "__main__":
    sys.exit(main())
