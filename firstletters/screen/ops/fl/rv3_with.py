#!/usr/bin/env python3
"""rv3_with.py: run an ensemble script with Reader V3 APPENDED to its model table, without editing the script.

  rv3_with.py ensemble_maps|make_submission_image|fl_ensemble [that script's arguments ...]

ensemble_maps and make_submission_image are Nieuwlaar's unchanged scripts (results/nieuwlaar/repo/ink/scripts);
fl_ensemble is ops/fl/ensemble.py. The entry goes LAST, so the four original models keep his order (it matters for the
float32 sums) and a run that does not include reader_v3 maps is the script's own run. Reader V3 maps use the villa
naming and value contract (uint8, label smoothing 0.5): <name>__reader_v3__L66s<25 + d>_reverse.tif.
Control: ops/fl/rv3_pc343.py arm B (four models through this wrapper) must equal the archived all4 ablation exactly.
"""
import importlib, sys
from pathlib import Path

ROOT = Path('/Users/andreashad04/money/vesuv')
SCRIPTS = ROOT / 'results/nieuwlaar/repo/ink/scripts'
EM_ENTRY = ('reader_v3', 'reader_v3')          # ensemble_maps: (stack name, map key)
MS_ENTRY = ('reader_v3', '__reader_v3__')      # make_submission_image: (model name, file name tag)
TARGETS = {'ensemble_maps': SCRIPTS / 'ensemble_maps.py', 'make_submission_image': SCRIPTS / 'make_submission_image.py',
           'fl_ensemble': ROOT / 'ops/fl/ensemble.py'}


def load(which):
    sys.path.insert(0, str(SCRIPTS))
    em = importlib.import_module('ensemble_maps')
    if EM_ENTRY not in em.MODELS:
        em.MODELS.append(EM_ENTRY)
    if which == 'ensemble_maps':
        return em
    if which == 'make_submission_image':
        ms = importlib.import_module('make_submission_image')
        if MS_ENTRY not in ms.MODELS:
            ms.MODELS.append(MS_ENTRY)
            ms.MODEL_ORDER.append(MS_ENTRY[0])
        return ms
    sys.path.insert(0, str(ROOT / 'ops/fl'))
    return importlib.import_module('ensemble')   # its MK table is built at import, after the append above


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TARGETS:
        raise SystemExit(f'usage: rv3_with.py {"|".join(TARGETS)} [args ...]')
    which = sys.argv[1]
    mod = load(which)
    assert Path(mod.__file__).resolve() == TARGETS[which].resolve(), (mod.__file__, TARGETS[which])
    sys.argv = [str(TARGETS[which])] + sys.argv[2:]
    rc = mod.main()
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == '__main__':
    main()
