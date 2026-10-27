#!/usr/bin/env python3
"""rv3_screen_unit.py: screen_unit.py for Reader V3 ALONE, without editing screen_unit.py or run_unit.py.

The Reader V3 second pass (results/readerv3/PREREGISTRATION_RV3.md, "Decision rule"): Reader V3 maps at d = +1 for the
screened meshes, rendered again because renders are deleted after each mesh; the four-model screen is never altered.

  rv3_screen_unit.py --scroll S --mesh M --models reader_v3 --depths 1 [screen_unit.py options] [--device mps|cpu]

What it changes, in this process only:
  - run_unit.CKPT gets 'reader_v3' (rv3_unit.register(), which checks the pinned sha256 of the weights);
  - screen_unit.LOGS becomes results/fl/logs_rv3/, so its log, render log, render record, last_run.json and
    screen_timings.tsv never touch the four-model ones in results/fl/logs/;
  - --device cpu (validation only) runs vesuvius on the CPU through rv3_unit's launcher and skips the MPS check.
Then screen_unit.main() runs unchanged: render, valid-mask equality against the kept mask, run_unit.one() with its
GPU-abort guard (rc 91), render deleted. The map lands next to the four-model maps under its own file name.
Guard: every file already in the mesh's (or crop's) maps directory is read before and compared after; any file that
changed (screen_unit rewrites the CT histogram json) gets its saved bytes back and the run exits 92.
"""
import argparse, hashlib, os, sys, time
from pathlib import Path

ROOT = Path('/Users/andreashad04/money/vesuv')
for p in ('ops/fl', 'ops/nieuwlaar'):
    sys.path.insert(0, str(ROOT / p))
import rv3_unit              # noqa: E402
import run_unit as ru        # noqa: E402
import screen_unit as su     # noqa: E402

LOGS_RV3 = ROOT / 'results/fl/logs_rv3'
RC_CHANGED = 92


def snapshot(d):
    """{file name: bytes} of every file already in d"""
    d = Path(d)
    return {p.name: p.read_bytes() for p in sorted(d.iterdir()) if p.is_file()} if d.is_dir() else {}


def changed(before, d):
    """names from the snapshot that are now missing or hold different bytes"""
    d = Path(d)
    return [n for n, b in before.items() if not (d / n).is_file() or (d / n).read_bytes() != b]


def restore(before, names, d):
    for n in names:
        tmp = Path(d) / (n + '.restore.partial')
        tmp.write_bytes(before[n])
        os.replace(tmp, Path(d) / n)


def maps_dir(scroll, mesh, crop):
    cname = None if not crop else 'crop_' + '_'.join(str(int(v)) for v in crop.split(','))
    return su.names(scroll, mesh, cname)[2]


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    pre.add_argument('--models', nargs='+', required=True)
    pre.add_argument('--scroll', required=True)
    pre.add_argument('--mesh', required=True)
    pre.add_argument('--crop')
    k, rest = pre.parse_known_args()
    if k.models != [rv3_unit.MODEL]:
        raise SystemExit(f'REFUSED: rv3_screen_unit.py runs {rv3_unit.MODEL} only, got --models {" ".join(k.models)}')
    rv3_unit.register()
    su.LOGS = LOGS_RV3
    LOGS_RV3.mkdir(parents=True, exist_ok=True)
    if k.device == 'cpu':
        ru.PY_VILLA = rv3_unit.cpu_launcher(rv3_unit.PYS['villa'])
        su.mps_ok = lambda py: (True, None)
    me = Path(__file__).resolve()
    print(f'rv3_screen_unit pid {os.getpid()} sha256 {hashlib.sha256(me.read_bytes()).hexdigest()} mtime '
          f'{time.ctime(me.stat().st_mtime)}; device {k.device}; screen_unit logs -> {su.LOGS}', flush=True)
    sys.argv = ([str(Path(su.__file__))] + rest + ['--scroll', k.scroll, '--mesh', k.mesh, '--models', rv3_unit.MODEL]
                + (['--crop', k.crop] if k.crop else []))
    maps = maps_dir(k.scroll, k.mesh, k.crop)
    before = snapshot(maps)
    rc = 0
    try:
        su.main()
    except SystemExit as e:
        if isinstance(e.code, int) or e.code is None:
            rc = e.code or 0
        else:
            print(e.code, flush=True)
            rc = 1
    finally:
        bad = changed(before, maps)
        if bad:
            restore(before, bad, maps)
            print(f'REFUSED: screen_unit changed {bad} in {maps}; the original bytes are restored', flush=True)
            rc = RC_CHANGED
    sys.exit(rc)


if __name__ == '__main__':
    main()
