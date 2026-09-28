"""ARM FLINJ phase 1 (as amended by AMENDMENT_1.md), shared pieces: the registered constants, the five hosts and their donors, the real
held-out AUC targets (READ from ARM Z's and F6's JSON, never typed), the frozen inference call (ARM Z's
call, unchanged), and atomic IO.

The sha256 of the pre-registration and of amendments 1 to 4 are checked at import, so an edited registration stops
every FLINJ script before it can produce a number.
"""
import hashlib, json, os, pathlib, re, shutil, subprocess, sys, tempfile, time
import numpy as np, zarr

ROOT = pathlib.Path(os.environ.get("VESUV_ROOT") or next((q for q in pathlib.Path(__file__).resolve().parents
                                                     if (q / "villa").is_dir() and (q / "models").is_dir()),
                                                    pathlib.Path(__file__).resolve().parents[2]))  # ops/flinj or public/bench/flinj
PY = str(ROOT / "env" / "bin" / "python3")
ARMGUARD = "/Users/andreashad04/bin/armguard"
RES, DATA = ROOT / "results" / "flinj", ROOT / "data" / "flinj"
PREREG = RES / "PREREGISTRATION_FLINJ.md"
PREREG_SHA = "73e70a5cb0fbb3d07db96e944475914bc788d4c36ff0cb99af81311ca84d1dd6"
_got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
assert _got == PREREG_SHA, f"pre-registration changed: sha256 {_got}"
assert PREREG_SHA in (RES / "prereg.sha256").read_text(), "prereg.sha256 does not record the registered hash"
AMEND = RES / "AMENDMENT_1.md"
AMEND_SHA = "c4d455a9393f158b94dbf10705c1faa8b46c1f19d4abd6ef9f1b8767c43b5e51"
_got = hashlib.sha256(AMEND.read_bytes()).hexdigest()
assert _got == AMEND_SHA, f"amendment 1 changed: sha256 {_got}"
assert AMEND_SHA in (RES / "prereg.sha256").read_text(), "prereg.sha256 does not record amendment 1"
AMEND2 = RES / "AMENDMENT_2.md"
AMEND2_SHA = "f85930005a1bc207147f5f74f394335821f0898283cbf0e167ab8ff1429c487b"
_got = hashlib.sha256(AMEND2.read_bytes()).hexdigest()
assert _got == AMEND2_SHA, f"amendment 2 changed: sha256 {_got}"
assert AMEND2_SHA in (RES / "prereg.sha256").read_text(), "prereg.sha256 does not record amendment 2"
AMEND3 = RES / "AMENDMENT_3.md"
AMEND3_SHA = "1d34cb73b9007142fd4d05e0f3c93916cfec6c51e6899be7673bec4faf93867f"
_got = hashlib.sha256(AMEND3.read_bytes()).hexdigest()
assert _got == AMEND3_SHA, f"amendment 3 changed: sha256 {_got}"
assert AMEND3_SHA in (RES / "prereg.sha256").read_text(), "prereg.sha256 does not record amendment 3"
AMEND4 = RES / "AMENDMENT_4.md"
AMEND4_SHA = "9b48be610260a3e4b17e4ec35e7487e65b98b0dbd02377b108e8ab19aa3b1a75"
_got = hashlib.sha256(AMEND4.read_bytes()).hexdigest()
assert _got == AMEND4_SHA, f"amendment 4 changed: sha256 {_got}"
assert AMEND4_SHA in (RES / "prereg.sha256").read_text(), "prereg.sha256 does not record amendment 4"
assert not (ROOT / "env").exists() or pathlib.Path(sys.prefix).resolve() == (ROOT / "env").resolve(), f"run with {PY}, not {sys.executable}"

# Every number the pre-registration and amendment 1 (A1 to A6) fix, in one place.
REG = dict(cap_mm=2.5, stroke_mm=0.35, letters_max=7, letters_min=4, letter_pitch_caps=1.2, row_pitch_caps=1.8, rot_deg=5.0,  # A1: 4 to 7
           border_px=64, soft_sigma_px=0.7, annulus_px=(16, 48), patch_px=32, patch_min_ink=0.90,
           smooth_sigma_px=1.5,                                                   # A2: primary residual smoothed in plane
           grid=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),                # A4: doubled up to 64
           d_floor_prob=0.02, auc_floor=0.55,                                     # A3: C undefined below either floor
           min_hosts_astar=3,                                                     # A4: fewer -> NOT DETECTABLE AT REALISTIC STRENGTH
           clip_max=0.05, c3_mean_change=20.0, c3_clip_max=0.01,                  # A5
           s4_stroke_mm=(0.25, 0.5, 0.7), far_caps=2.0,
           strokes_bar=0.5, blobs_bar=0.2, m1_bar=0.8, m2_bar=0.2, m2_sigma_caps=1.0 / 3.0, c3_bar=0.5, c3_min_hosts=4,
           # amendment 2: transplant (B1, B2), floor on the pair (B3), C3 clip bound 10% (B4), S5/S6 at a = 0.5
           b2_label_dist_px=48, b3_d_floor=0.02, b3_auc_floor=0.55, b3_min_hosts=3, c3_min_frac=0.75, s56_a=0.5,  # amendment 4: D1 C3 on ceil(0.75 x included), D2 no C3 clip bound
          
           host_bg="valid_not_ink",  # amendment 3, C1: b over valid, not-labelled-ink pixels, donor AND host, B1 AND B2
           ink_clear_px=48)          # amendment 3, C2: every glyph pixel more than 48 px from host-labelled ink

PRIMARY_KEY = "hybrid_3d2d-seed42_step-075000"
PRIMARY_CKPT = ROOT / "models" / "ink_9um" / "hybrid_3d2d-seed42" / "step-075000.pth"
DIRECTION = "forward"  # ARM Z and F6 chose forward on all five hosts; asserted in target_auc()

HOSTS = {
    "w00": dict(arm="zs", dir=ROOT / "data/zs/w00", ct=ROOT / "data/zs/w00/w04/ct.zarr", um=9.366, donors=["ag144", "ag174"]),
    "ag144": dict(arm="zs", dir=ROOT / "data/zs/ag144", ct=ROOT / "data/zs/ag144/w04/ct.zarr", um=9.366, donors=["w00", "ag174"]),
    "ag174": dict(arm="zs", dir=ROOT / "data/zs/ag174", ct=ROOT / "data/zs/ag174/w04/ct.zarr", um=9.366, donors=["w00", "ag144"]),
    "p0009b": dict(arm="f6", dir=ROOT / "data/f6/p0009b", ct=ROOT / "data/f6/p0009b/N0/ct.zarr", um=8.64, donors="halves"),
    "p0500p2": dict(arm="f6", dir=ROOT / "data/f6/p0500p2", ct=ROOT / "data/f6/p0500p2/N0/ct.zarr", um=9.362, donors=["w00", "ag144", "ag174"]),
}


def spacing_um(h):
    """The host's in-plane spacing, read from its meta.json and checked against the table."""
    m = json.load(open(HOSTS[h]["dir"] / "meta.json"))
    v = float(re.search(r"/(\d+\.\d+)um-", m["source"]).group(1)) if HOSTS[h]["arm"] == "zs" else float(m["nat_um"])
    assert abs(v - HOSTS[h]["um"]) < 1e-9, (h, v)
    return v


def cap_px(h, cap_mm=REG["cap_mm"]):
    return cap_mm * 1000.0 / spacing_um(h)


def mm_px(h, mm):
    return mm * 1000.0 / spacing_um(h)


def target_auc(h):
    """That host's REAL held-out AUC for the primary checkpoint in its chosen direction (ARM Z / F6)."""
    if HOSTS[h]["arm"] == "zs":
        r = json.load(open(ROOT / "results/zs/armz_scores.json"))["segments"][h]["rows"][PRIMARY_KEY]
    else:
        r = json.load(open(ROOT / "results/fs/f6_scores.json"))["segments"][h]["N0"]
    assert r["chosen"] == DIRECTION, (h, r["chosen"])
    return float(r["auc_chosen"])


def reference_pred(h):
    """The stored ARM Z / F6 prediction for the same (unaltered) input, used only as a cross-check."""
    if HOSTS[h]["arm"] == "zs":
        return ROOT / "results/zs/pred" / h / "w04" / f"{PRIMARY_KEY}_{DIRECTION}.tif"
    return ROOT / "results/fs/f6pred" / h / "N0" / f"{PRIMARY_KEY}_{DIRECTION}.tif"


def all_ckpts():
    """The 14 ink_9um checkpoints, primary first (S3)."""
    c = sorted((ROOT / "models" / "ink_9um").glob("hybrid_3d2d-seed*/step-*.pth"))
    assert len(c) == 14 and PRIMARY_CKPT in c, len(c)
    return [PRIMARY_CKPT] + [x for x in c if x != PRIMARY_CKPT]


def ckpt_key(ck):
    return f"{ck.parent.name}_{ck.stem}"


def host_config(h):
    """(um, keV) of the host's scan, parsed from its meta.json (D4 statement)."""
    m = json.load(open(HOSTS[h]["dir"] / "meta.json")); u, k = re.search(r"(\d+\.\d+)um-[^/]*?(\d+)keV", m.get("source") or m["nat"]).groups()
    return float(u), int(k)


def host_seed(h):
    """One fixed seed per host (pre-registration, Injection 3)."""
    return int(hashlib.sha256(f"FLINJ-phase1-{h}".encode()).hexdigest()[:8], 16)


def load_volume(ct):
    return np.asarray(zarr.open_array(str(ct), mode="r")[:])


def load_host(h):
    d = HOSTS[h]["dir"]; I = load_volume(HOSTS[h]["ct"])
    label, support = np.load(d / "label.npy"), np.load(d / "support.npy")
    assert I.shape[0] == 21 and I.dtype == np.uint8 and I.shape[1:] == label.shape == support.shape, (h, I.shape)
    valid = ~(I == 0).all(0)  # the surface volume holds data here
    return I, label & support, support, valid


def write_zarr(path, arr):
    """Same layout as every earlier arm's ct.zarr: zarr v2, (21, H, W) uint8, chunks (21, 128, 128); atomic rename."""
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=path.name + ".", dir=path.parent))
    zarr.open_array(str(tmp / "a.zarr"), mode="w", shape=arr.shape, chunks=(arr.shape[0], 128, 128), dtype=np.uint8, zarr_format=2)[:] = arr
    if path.exists(): shutil.rmtree(path)
    (tmp / "a.zarr").rename(path); shutil.rmtree(tmp, ignore_errors=True)


def atomic_json(path, obj):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.partial{os.getpid()}")
    with open(tmp, "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.partial{os.getpid()}")
    with open(tmp, "wb") as f: np.savez(f, **arrays)
    os.replace(tmp, path)


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def infer(ct, ckpt, out_tif):
    """ARM Z's inference call, unchanged: villa infer.main in process via ops/xs_infer.py (MPS patch), fp32,
    forward, no compile, 0 workers, batch 32. Written to a per-process .partial TIFF and renamed."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # ops/ or bench/
    import xs_infer as XI  # noqa: E402
    D, H, W = zarr.open_array(str(ct), mode="r").shape; assert D == 21
    out = pathlib.Path(out_tif); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.partial{os.getpid()}.tif")
    if tmp.exists(): tmp.unlink()
    t = time.time()
    XI.INF.main([str(ct), str(ckpt), str(tmp), "--direction", DIRECTION, "--no-compile", "--num-workers", "0", "--batch-size", "32"])
    dt = time.time() - t
    assert XI.valid_tiff(tmp, (H, W)), f"no valid output at {tmp}"
    os.replace(tmp, out)
    return dt


def armguard(need_gb, poll_s=60, max_wait_s=1800):
    """armguard check --need N; poll every poll_s seconds up to max_wait_s. Returns (ok, last output)."""
    t0 = time.time()
    while True:
        r = subprocess.run([ARMGUARD, "check", "--need", str(need_gb)], capture_output=True, text=True)
        msg = (r.stdout + r.stderr).strip()
        print(f"armguard --need {need_gb}: rc={r.returncode} {msg}", flush=True)
        if r.returncode == 0: return True, msg
        if time.time() - t0 + poll_s > max_wait_s: return False, msg
        time.sleep(poll_s)


def banner(path):
    """Every long-running FLINJ script prints its own source hash, mtime and pid, so a log names its version."""
    p = pathlib.Path(path); b = p.read_bytes()
    print(f"{p.name} sha256 {hashlib.sha256(b).hexdigest()[:16]} mtime {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(p.stat().st_mtime))} pid {os.getpid()}", flush=True)
