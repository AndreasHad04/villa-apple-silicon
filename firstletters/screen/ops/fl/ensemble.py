#!/usr/bin/env python3
"""ensemble.py: Nieuwlaar's z-score ensemble (results/nieuwlaar/repo/ink/scripts/ensemble_maps.py) for one mesh screened
by screen_unit.py, from its maps and the render byproducts it kept (the render itself is deleted).

  ensemble.py --scroll PHerc0800 --mesh z16464_w020 --models hybrid_3d2d-seed42 --depths 1
              [--orientation reading|raw] [--maps-dir DIR] [--out-dir DIR] [--blind | --stats-side forward]

Statistics are his main(), line for line, with his functions (pct_hist, map_path, read_map, to8, write_png) and
constants: valid = CT > 0 in all 66 layers; per model p1/p99.5 stretch and pooled mean/SD over the valid pixels of all
--depths (reading side = reverse maps); ensemble = mean of the per-model z-scores, ONE stretch p1..p99.5 over the
fixed grid (every 8th row and column from 4) at the depths where every model exists; CT stretch from layers 18..48.
Differences: only the requested models (his MODELS order kept, it matters for the float32 sums); one model gives its
z-score image as the "ensemble" and the stats say single_model (his script writes no ensemble below 2 models); the blind
side only with --blind (his lines: forward maps shown with the READING side's statistics, never their own, suffix
_blind; the screen makes reverse maps only); --stats-side forward treats the blind side as the primary side (its own
statistics and images, no suffix: the blind side at its own scale); orientation 'reading' = high z at the top and theta decreasing to the right,
MEASURED for this mesh by his measure_orientation around the axis from orient.py (his fixed 180 degree rotation was
measured for the official PHerc0343 meshes only).

Writes <out>/<name>__{ensemble,<model>,ct}__d+NN.png for every --depths value plus d = +1, <name>__stats.json, and
previews at most 1600 px wide in <out>/preview/. Default out: results/fl/images/<scroll>/<mesh>/.
"""
import argparse, csv, hashlib, json, math, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path('/Users/andreashad04/money/vesuv')
sys.path.insert(0, str(ROOT / 'results/nieuwlaar/repo/ink/scripts'))
sys.path.insert(0, str(ROOT / 'ops/fl'))
import ensemble_maps as em    # noqa: E402
import orient                 # noqa: E402

PREVIEW_W = 1600
MK = [mk for _, mk in em.MODELS]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scroll', required=True)
    ap.add_argument('--mesh', required=True)
    ap.add_argument('--models', required=True, nargs='+', choices=MK)
    ap.add_argument('--depths', required=True, nargs='+', type=int)
    ap.add_argument('--orientation', choices=('reading', 'raw'), default='reading')
    ap.add_argument('--maps-dir')
    ap.add_argument('--out-dir')
    ap.add_argument('--blind', action='store_true', help='also the blind side (forward maps), his convention')
    ap.add_argument('--stats-side', choices=('reverse', 'forward'), default='reverse',
                    help='forward: statistics and images from the blind side alone (its own scale)')
    a = ap.parse_args()
    if a.blind and a.stats_side == 'forward':
        ap.error('--blind adds the forward side to a reverse-side run; with --stats-side forward it is the primary side')
    import cv2
    t0 = time.time()
    me = Path(__file__).resolve()
    print(f'ensemble pid {os.getpid()} sha256 {hashlib.sha256(me.read_bytes()).hexdigest()} '
          f'mtime {time.ctime(me.stat().st_mtime)}', flush=True)
    name = f'{a.scroll}_{a.mesh}_66'
    maps = Path(a.maps_dir or ROOT / 'data/fl/maps' / a.scroll / a.mesh)
    out = Path(a.out_dir or ROOT / 'results/fl/images' / a.scroll / a.mesh)
    (out / 'preview').mkdir(parents=True, exist_ok=True)
    stat_ds, write_ds = sorted(set(a.depths)), sorted(set(a.depths) | {1})

    valid = cv2.imread(str(maps / f'{name}__valid.png'), cv2.IMREAD_UNCHANGED) > 0
    cth = json.loads((maps / f'{name}__cthist.json').read_text())
    H, W = valid.shape
    assert (H, W) == (cth['H'], cth['W']) and int(valid.sum()) == cth['valid_px'], 'valid mask and histogram disagree'
    ct_lo, ct_hi = em.pct_hist(np.array(cth['ct_hist'], np.int64), np.arange(256, dtype=float), [em.P_LO, em.P_HI])
    sub = np.zeros_like(valid)
    sub[em.SUB // 2::em.SUB, em.SUB // 2::em.SUB] = True
    sub &= valid
    stats = dict(surface=f'{name}.zarr (deleted after the screen; valid mask and CT histogram kept)', H=H, W=W,
                 valid_px=int(valid.sum()), ct=dict(p1=ct_lo, p99_5=ct_hi, layers=cth['ct_layers']), models={})

    if a.orientation == 'reading':
        mt = [m for m in orient.mesh_table(a.scroll) if m['mesh'] == a.mesh][0]
        X, Y, Z, ok, _ = orient.msi.load_mesh(str(mt['dir']))
        _, om = orient.msi.measure_orientation(X, Y, Z, ok, orient.scroll_axis(a.scroll))
        flip_lr, flip_ud = bool(om['flip_lr']), bool(om['flip_ud'])
        stats['orientation'] = dict(mode='reading', flip_lr=flip_lr, flip_ud=flip_ud, measured=om,
                                    axis=str(orient.SCROLLS[a.scroll]['axis']))
    else:
        flip_lr = flip_ud = False
        stats['orientation'] = dict(mode='raw')

    def put(fname, b):
        b = b[::-1] if flip_ud else b
        b = np.ascontiguousarray(b[:, ::-1] if flip_lr else b)
        for d, img in ((out, b), (out / 'preview', None)):
            if img is None:
                img = b if b.shape[1] < 2000 else cv2.resize(
                    b, (PREVIEW_W, max(1, round(b.shape[0] * PREVIEW_W / b.shape[1]))), interpolation=cv2.INTER_AREA)
            tmp = d / (fname[:-4] + '.partial.png')
            em.write_png(str(tmp), img)
            os.replace(tmp, d / fname)

    # per-model statistics from the reading side: his main(), requested models only, his order
    subs, shas = {}, {}
    for sn, mk in em.MODELS:
        if mk not in a.models:
            continue
        villa = mk != 'hecate'
        hist = np.zeros(129 if villa else em.FINE, np.int64)
        s1 = s2 = 0.0; n = 0; ds = []; sv = []
        for d in stat_ds:
            p = em.map_path(str(maps), name, mk, a.stats_side, d)
            if not p:
                continue
            shas[os.path.basename(p)] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
            M = em.read_map(p, (H, W))
            v = M[valid]
            if villa:
                hist += np.bincount(np.rint(v * 128).astype(np.int64), minlength=129)[:129]
            else:
                hist += np.bincount(np.minimum((v * em.FINE).astype(np.int64), em.FINE - 1), minlength=em.FINE)
            s1 += float(v.sum(dtype=np.float64)); s2 += float(np.square(v, dtype=np.float64).sum()); n += v.size
            sv.append(M[sub].astype(np.float32)); ds.append(d)
        if not ds:
            stats['models'][sn] = dict(model=mk, depths=[])
            print(f'{sn}: no maps', flush=True)
            continue
        values = (np.arange(129) / 128.0) if villa else (np.arange(em.FINE) / em.FINE)
        lo, hi = em.pct_hist(hist, values, [em.P_LO, em.P_HI])
        mu = s1 / n
        sd = math.sqrt(max(s2 / n - mu * mu, 0.0)) + 1e-6
        stats['models'][sn] = dict(model=mk, depths=ds, p1=lo, p99_5=hi, mean=mu, sd=sd, n_px=n)
        subs[sn] = np.stack(sv)
        print(f'{sn}: {len(ds)} depths, stretch {lo:.4f}..{hi:.4f}, mean {mu:.4f}, sd {sd:.4f}', flush=True)

    # ensemble statistics on the fixed grid, at depths where every model with maps exists
    ens_models = [sn for sn, _ in em.MODELS if stats['models'].get(sn, {}).get('depths')]
    common = sorted(set.intersection(*[set(stats['models'][sn]['depths']) for sn in ens_models])) if ens_models else []
    ens = dict(models=ens_models, depths=common, single_model=len(ens_models) == 1)
    if ens_models and common:
        E = None
        for sn in ens_models:
            m = stats['models'][sn]
            idx = [m['depths'].index(d) for d in common]
            z = (subs[sn][idx] - m['mean']) / m['sd']
            E = z if E is None else E + z
        E /= len(ens_models)
        ens['p1'], ens['p99_5'] = (float(x) for x in np.percentile(E.ravel(), [em.P_LO, em.P_HI]))
        ens['n_grid'] = int(E.size)
    stats['ensemble'] = ens

    # images
    blind_shas = {}
    for d in write_ds:
        ctp = maps / f'{name}__ct__L{em.MID + d}.png'
        if ctp.exists():
            ct = cv2.imread(str(ctp), cv2.IMREAD_UNCHANGED)
            put(f'{name}__ct__d{d:+03d}.png', em.to8(ct.astype(np.float32), ct_lo, ct_hi, valid, 'raw'))
        for direction, suffix in ((a.stats_side, ''),) + ((('forward', '_blind'),) if a.blind else ()):
            acc = None; nmod = 0
            for sn, mk in em.MODELS:
                m = stats['models'].get(sn, {})
                if 'p1' not in m:
                    continue
                p = em.map_path(str(maps), name, mk, direction, d)
                if not p:
                    print(f'missing: {sn} {direction} d{d:+d}')
                    continue
                if direction == 'forward':
                    blind_shas[os.path.basename(p)] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
                M = em.read_map(p, (H, W))
                put(f'{name}__{sn}{suffix}__d{d:+03d}.png', em.to8(M, m['p1'], m['p99_5'], valid, 'raw'))
                if ens.get('p1') is not None and sn in ens['models']:
                    zc = (M - m['mean']) / m['sd']
                    acc = zc if acc is None else acc + zc
                    nmod += 1
            if ens.get('p1') is not None and nmod == len(ens['models']):
                put(f'{name}__ensemble{suffix}__d{d:+03d}.png', em.to8(acc / nmod, ens['p1'], ens['p99_5'], valid, 'raw'))
            else:
                print(f'no ensemble for {"" if direction == "reverse" else "forward "}d{d:+d} ({nmod} of {len(ens["models"])} models)')
        print(f'd{d:+d} written', flush=True)
    stats['maps_sha256'] = shas
    if a.stats_side != 'reverse':
        stats['stats_side'] = a.stats_side
    if a.blind:
        stats['blind'] = dict(convention='forward maps with the reading side statistics (his ensemble_maps.py)',
                              maps_sha256=blind_shas)
    stats['seconds'] = round(time.time() - t0, 1)
    tmp = out / f'{name}__stats.json.partial'
    tmp.write_text(json.dumps(stats, indent=1))
    os.replace(tmp, out / f'{name}__stats.json')
    print('done', out)


if __name__ == '__main__':
    main()
