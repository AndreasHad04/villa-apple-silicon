#!/usr/bin/env python3
"""rv3_unit.py: Reader V3 (Domenico Russo, 2026-10-19) as one more ink model of the production runner, WITHOUT editing it.

It registers 'reader_v3' in ops/nieuwlaar/run_unit.py's CKPT table at runtime and then calls run_unit.one() itself, so
the infer command line, env, TMPDIR, private temp dir, shape check, GPU-abort guard (rc 91) and atomic rename are the
production ones. Reader V3 is a vesuvius_unet_3d_stem_2d flat checkpoint for the same command as dnative and
hybrid_3d2d (17 layers, robust_mad 1..99, fp16, label smoothing 0.5); its only contract difference is 384 px patches
(theirs: 128 px). Record: results/readerv3/model_record.json. Pre-registration: results/readerv3/PREREGISTRATION_RV3.md.

  rv3_unit.py --tiling plain --d D [D ...] [--device mps|cpu] [--python villa|env]
              [--render ZARR --maps DIR --name NAME --work DIR]     (another mesh or a crop, as screen_unit.py does)

Maps land at run_unit.out_path(tiling, 'reader_v3', d), e.g. data/nieuwlaar/maps_plain/<N>_66__reader_v3__L66s26_reverse.tif
for d = +1, which ensemble_maps.map_path finds once the model is registered (ops/fl/rv3_with.py).
--device mps (default) refuses while two or more other GPU ink jobs or any A/B training job run (STATE.md RESULT 77).
--device cpu replaces vesuvius' get_accelerator with the CPU in the child and adds --no-compile: validation only.
--python villa (default) = run_unit.PY_VILLA, the interpreter of the other three villa models; env = env/bin/python3.
"""
import argparse, hashlib, os, subprocess, sys, time
from pathlib import Path

ROOT = Path('/Users/andreashad04/money/vesuv')
sys.path.insert(0, str(ROOT / 'ops/nieuwlaar'))
import run_unit as ru   # noqa: E402  production runner; its main() is never called from here

MODEL = 'reader_v3'
CKPT = ROOT / 'models/reader_v3/reader-v3.pth'
CKPT_SHA256 = 'ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a'   # HF LFS oid at revision aee1d64b
INFER = 'vesuvius.ink_detection.inference.infer'
PYS = {'villa': ru.PY_VILLA, 'env': ROOT / 'env/bin/python3'}
CPU_MARK = 'rv3: get_accelerator forced to cpu'


def register():
    """put Reader V3 into run_unit.CKPT after checking the weights are the pinned release"""
    h = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    if h != CKPT_SHA256:
        raise SystemExit(f'REFUSED: {CKPT} sha256 {h} is not the pinned release {CKPT_SHA256}')
    ru.CKPT[MODEL] = CKPT


def point(render, maps, name, work):
    """aim run_unit at another surface volume (a crop or a screen mesh), exactly as screen_unit.py does"""
    ru.N, ru.DATA, ru.RENDER, ru.MAPS = name, Path(work), {'plain': Path(render)}, {'plain': Path(maps)}
    ru.TIMINGS = Path(work) / 'logs' / f'{name}.unit_timings.tsv'


def gpu_jobs():
    """pids of other processes that use the GPU for ink inference or for A/B training"""
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me, jobs = os.getpid(), []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(' ')
        if int(pid) != me and (INFER in cmd or 'hecate.py --checkpoint' in cmd or '/ops/ab/' in cmd):
            jobs.append((int(pid), cmd[:120]))
    return jobs


def cpu_launcher(py):
    """an executable that run_unit.one() calls in place of the interpreter: same argv, CPU forced"""
    d = Path(f'/tmp/rv3cpu{os.getpid()}')
    d.mkdir(exist_ok=True)
    f = d / 'python3'
    f.write_text('#!/bin/sh\n'
                 f'if [ "$1" != "-m" ] || [ "$2" != "{INFER}" ]; then echo "rv3 cpu launcher: unexpected argv: $*"; exit 2; fi\n'
                 'shift 2\n'
                 f'exec "{py}" "{Path(__file__).resolve()}" --cpu-infer "$@"\n')
    f.chmod(0o755)
    return f


def cpu_infer(argv):
    """child side of --device cpu: vesuvius' own infer.main() with the accelerator lookup pinned to the CPU"""
    import functools
    import torch
    real_is_available = torch.backends.mps.is_available

    # functools.wraps keeps __wrapped__: torch 2.14's dynamo reads it at import (a bare lambda raised AttributeError)
    @functools.wraps(real_is_available)
    def no_mps():
        return False
    torch.backends.mps.is_available = no_mps
    from vesuvius.ink_detection.inference import inference_runtime as rt

    def cpu():
        print(CPU_MARK, flush=True)
        return torch.device('cpu')
    rt.get_accelerator = cpu
    from vesuvius.ink_detection.inference import infer
    argv = list(argv) + ([] if '--no-compile' in argv else ['--no-compile'])
    return infer.main(argv)


def one(tiling, d, device='mps', python='villa'):
    """one Reader V3 map, reading side (reverse), through run_unit.one() itself"""
    register()
    saved = ru.PY_VILLA
    ru.PY_VILLA = cpu_launcher(PYS[python]) if device == 'cpu' else PYS[python]
    try:
        return ru.one(tiling, MODEL, d)
    finally:
        ru.PY_VILLA = saved


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--cpu-infer':
        sys.exit(cpu_infer(sys.argv[2:]))
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tiling', default='plain', choices=list(ru.RENDER))
    ap.add_argument('--d', type=int, nargs='+', required=True)
    ap.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    ap.add_argument('--python', choices=list(PYS), default='villa')
    ap.add_argument('--render')
    ap.add_argument('--maps')
    ap.add_argument('--name')
    ap.add_argument('--work')
    a = ap.parse_args()
    me = Path(__file__).resolve()
    print(f'rv3_unit pid {os.getpid()} sha256 {hashlib.sha256(me.read_bytes()).hexdigest()} mtime {time.ctime(me.stat().st_mtime)}; '
          f'run_unit sha256 {hashlib.sha256(Path(ru.__file__).read_bytes()).hexdigest()}', flush=True)
    custom = [a.render, a.maps, a.name, a.work]
    if any(custom):
        if not all(custom) or a.tiling != 'plain':
            raise SystemExit('--render, --maps, --name and --work go together, with --tiling plain')
        point(*custom)
    if not (Path(ru.RENDER[a.tiling]) / '0').exists():
        raise SystemExit(f'render missing: {ru.RENDER[a.tiling]}')
    if a.device == 'mps':
        chk = subprocess.run([str(PYS[a.python]), '-c', 'import torch; print(torch.backends.mps.is_available())'],
                             capture_output=True, text=True)
        if chk.stdout.strip() != 'True':
            raise SystemExit(f'MPS not available to {PYS[a.python]}: {chk.stdout!r} {chk.stderr[-300:]!r}')
    bad = 0
    for d in a.d:
        assert -15 <= d <= 15, d
        if a.device == 'mps':
            jobs = gpu_jobs()
            if len(jobs) >= 2 or any('/ops/ab/' in c for _, c in jobs):
                raise SystemExit(f'REFUSED: other GPU jobs running {jobs}; at most two inference jobs share the GPU '
                                 'and never beside training (STATE.md RESULT 77)')
        bad |= one(a.tiling, d, a.device, a.python) != 0
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
