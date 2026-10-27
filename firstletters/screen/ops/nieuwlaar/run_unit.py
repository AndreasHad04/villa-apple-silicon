#!/usr/bin/env python3
"""run_unit.py: one pre-registered ink unit (tiling, model, d), READING side (reverse) by default, as his run_ink.sh does it
(results/nieuwlaar/PREREGISTRATION_MAC.md, sha256 607f4ec2...). direction='forward' gives the BLIND side: the same planes
in the stored order (villa --direction forward, hecate without --reverse), saved under his file name for that side (no
suffix, ensemble_maps.map_path); the default call is unchanged byte for byte (ops/fl/control_direction.py).

  run_unit.py --tiling plain|padded --model MODEL [MODEL ...] --d D [D ...]
  (models: dnative hybrid_3d2d-seed42 hybrid_3d2d-seed43 hecate; d in -15..15; runs every (d, model), d-major)

Resumable: a unit whose output exists is skipped. Each unit runs in a private temp dir on the same volume and its
map is renamed into place only after the run finished and the map has the expected shape. Several --d values run
in that order. Timings go to results/nieuwlaar/logs/timings.tsv.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path('/Users/andreashad04/money/vesuv')
REPO = ROOT / 'results/nieuwlaar/repo'
DATA = ROOT / 'data/nieuwlaar'
N = 'concat_w047-w048_R5B2_z9500-11000'
RENDER = {'plain': DATA / f'{N}_66.zarr', 'padded': DATA / f'{N}_66_padded.zarr'}
MAPS = {'plain': DATA / 'maps_plain', 'padded': DATA / 'maps_padded'}
CK = ROOT / 'models/nieuwlaar_ckpt'
CKPT = {'dnative': CK / 'dense_native-016000.rebuilt.pth',
        'hybrid_3d2d-seed42': CK / 'hybrid_3d2d-seed42_step-075000.pth',
        'hybrid_3d2d-seed43': CK / 'hybrid_3d2d-seed43_step-075000.pth',
        'hecate': CK / 'hecate_9.6um.pth'}
PY_VILLA = ROOT / 'scratch_envs/env212/bin/python3'          # torch 2.12.1, the vesuvius lock
VILLA_SRC = ROOT / 'villa-inkcopy/vesuvius/src'               # villa main + the MPS copy-race fix
PY_HEC = ROOT / 'env/bin/python3'                             # torch 2.14.0
TIMINGS = ROOT / 'results/nieuwlaar/logs/timings.tsv'
NL = 66
# Metal aborted or ignored a command buffer: the patches it held were never computed, yet the stitcher still writes a
# full-size map (2026-10-25: 24 PHerc0800 screen maps, up to 708 aborts in one, NaN tiles in the blend)
GPU_ABORT = b'CommandBufferCallbackError'
DIRECTIONS = ('reverse', 'forward')     # reading side (his screen), blind side


def tag(direction):
    """suffix of log and temp names: none for the reading side, so default names are unchanged"""
    assert direction in DIRECTIONS, direction
    return '' if direction == 'reverse' else f'_{direction}'


def log_name(tiling, model, d, direction='reverse'):
    return f'{tiling}_{model}_d{d:+d}{tag(direction)}.log'


def gpu_aborts(log, offset=0):
    """abort lines the child wrote to `log` after byte `offset`, i.e. during this run only"""
    if not Path(log).exists():
        return 0
    with open(log, 'rb') as f:
        f.seek(offset)
        return f.read().count(GPU_ABORT)


def out_path(tiling, model, d, direction='reverse'):
    assert direction in DIRECTIONS, direction
    rev = '_reverse' if direction == 'reverse' else ''      # his naming (ensemble_maps.map_path)
    name = f'{N}_66' if tiling == 'plain' else f'{N}_66_padded'
    if model == 'hecate':
        return MAPS[tiling] / f'{name}__hecate__L66zc{32.5 + d:.2f}{rev}.png'
    return MAPS[tiling] / f'{name}__{model}__L66s{NL // 2 + d - 8}{rev}.tif'


def expected_shape(tiling, model):
    import zarr
    _, H, W = zarr.open_group(str(RENDER[tiling]), mode='r')['0'].shape
    return (int(round(H * 0.9)), int(round(W * 0.9))) if model == 'hecate' else (H, W)


def run(cmd, log, env=None, cwd=None):
    with open(log, 'a') as f:
        f.write(f'$ {" ".join(map(str, cmd))}\n')
        f.flush()
        return subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd)


def one(tiling, model, d, direction='reverse'):
    dst = out_path(tiling, model, d, direction)
    if dst.exists() and dst.stat().st_size > 0:
        print(f'skip (exists) {dst.name}', flush=True)
        return 0
    MAPS[tiling].mkdir(parents=True, exist_ok=True)
    (DATA / 'logs').mkdir(parents=True, exist_ok=True)
    log = DATA / 'logs' / log_name(tiling, model, d, direction)
    log_off = log.stat().st_size if log.exists() else 0
    tmp = DATA / 'tmp' / f'{tiling}_{model}_d{d:+d}{tag(direction)}_{os.getpid()}'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    # TMPDIR must stay short: torch_shm_manager puts a UNIX socket there and macOS caps socket paths at 104
    # bytes (a TMPDIR under data/ failed with 'no response from torch_shm_manager', measured 2026-10-12)
    stmp = Path(f'/tmp/nl{os.getpid()}')
    shutil.rmtree(stmp, ignore_errors=True)
    stmp.mkdir()
    env = dict(os.environ, TMPDIR=str(stmp), LC_ALL='C')
    t0 = time.time()
    z = RENDER[tiling]
    if model != 'hecate':
        s = NL // 2 + d - 8
        e = s + 17
        assert 0 <= s and e <= NL, (s, e)
        env['PYTHONPATH'] = str(VILLA_SRC)
        rc = run([PY_VILLA, '-m', 'vesuvius.ink_detection.inference.infer', z, CKPT[model], tmp / 'out.tif',
                  '--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '4', '--direction', direction,
                  '--layer-start', s, '--layer-end', e], log, env=env, cwd=tmp)
        prod = tmp / 'out.tif'
    else:
        zc = f'{32.5 + d:.2f}'
        rc = run([PY_HEC, ROOT / 'ops/nieuwlaar/hecate_prep_compat.py', '--src', z, '--out', tmp / 'prep.zarr',
                  '--native-um', '8.64', '--native-um-z', '8.64', '--target-um', '9.6', '--z-center', zc, '--nz', '16'],
                 log, env=env, cwd=tmp)
        if rc == 0:
            rev = '_reverse' if direction == 'reverse' else ''
            rc = run([PY_HEC, CK / 'hecate.py', '--checkpoint', CKPT['hecate'], '--input', tmp / 'prep.zarr',
                      '--spacing-um', '9.6', '--output', tmp / f'out{rev}.png', '--device', 'mps',
                      '--batch-size', '4'] + (['--reverse'] if direction == 'reverse' else []), log, env=env, cwd=tmp)
        prod = tmp / f'out{"_reverse" if direction == "reverse" else ""}.png'
    secs = time.time() - t0
    ok = rc == 0 and prod.exists() and prod.stat().st_size > 0
    n_abort = gpu_aborts(log, log_off)
    if n_abort:
        ok, rc = False, 91
        with open(log, 'a') as f:
            f.write(f'GPU ABORT: {n_abort} Metal command buffer errors during this run, output refused\n')
    if ok:
        import cv2, tifffile
        a = cv2.imread(str(prod), cv2.IMREAD_UNCHANGED) if model == 'hecate' else tifffile.imread(prod)
        want = expected_shape(tiling, model)
        if a is None or tuple(a.shape[:2]) != want:
            ok, rc = False, 90
            with open(log, 'a') as f:
                f.write(f'SHAPE MISMATCH {None if a is None else a.shape} != {want}\n')
    if ok:
        js = prod.with_suffix('.json')                     # hecate writes output.with_suffix('.json')
        if model == 'hecate' and js.exists():
            os.replace(js, dst.with_suffix('.json'))
        os.replace(prod, dst)
    TIMINGS.parent.mkdir(parents=True, exist_ok=True)
    with open(TIMINGS, 'a') as f:
        f.write(f'{tiling}\t{model}{"" if direction == "reverse" else ":" + direction}\t{d}\t{secs:.1f}\t{rc}\t{time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))}\n')
    print(f'{"ok" if ok else "FAILED rc=" + str(rc)} {tiling} {model} d={d:+d} {secs:.1f} s -> {dst.name}', flush=True)
    shutil.rmtree(stmp, ignore_errors=True)
    if ok:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else (rc or 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiling', required=True, choices=list(RENDER))
    ap.add_argument('--model', required=True, nargs='+', choices=list(CKPT))
    ap.add_argument('--d', type=int, nargs='+', required=True)
    ap.add_argument('--direction', choices=DIRECTIONS, default='reverse')
    a = ap.parse_args()
    me = Path(__file__).resolve()
    print(f'run_unit pid {os.getpid()} sha256 {hashlib.sha256(me.read_bytes()).hexdigest()} '
          f'mtime {time.ctime(me.stat().st_mtime)}', flush=True)
    # fail fast if Metal is not reachable (e.g. a launchd context): villa would silently fall back to CPU
    chk = subprocess.run([str(PY_VILLA), '-c', 'import torch; print(torch.backends.mps.is_available())'],
                         capture_output=True, text=True)
    if chk.stdout.strip() != 'True':
        raise SystemExit(f'MPS not available to the villa python: {chk.stdout!r} {chk.stderr[-300:]!r}')
    if not (RENDER[a.tiling] / '0').exists():
        raise SystemExit(f'render missing: {RENDER[a.tiling]}')
    bad = 0
    for d in a.d:
        assert -15 <= d <= 15, d
        for m in a.model:
            bad |= one(a.tiling, m, d, a.direction) != 0
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    # Runs only from ops/nieuwlaar/chain.sh, which sets NL_OWN_CHAIN=1: this work moved off the shared
    # nightqueue on 2026-10-12, so a resumed copy of that chain must do nothing.
    if os.environ.get('NL_OWN_CHAIN') != '1':
        print('skipped: runs only inside ops/nieuwlaar/chain.sh', flush=True)
        sys.exit(0)
    main()
