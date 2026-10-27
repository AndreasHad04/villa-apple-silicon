#!/usr/bin/env python3
"""screen_unit.py: First Letters screen of ONE mesh, or of one crop of it, with Nieuwlaar's PHerc0343 ink recipe:
render 66 layers with +d pointing away from the scroll axis, run the reading-side (reverse) map of every (model, d),
keep the maps, delete the render.

  screen_unit.py --scroll PHerc0800 --mesh z16464_w020 --models hybrid_3d2d-seed42 --depths 1
                 [--cache-dir DIR [--cache-readonly]] [--crop X0,Y0,X1,Y1 [--crop-margin 256]] [--keep-render]
  (scrolls and meshes: orient.SCROLLS / orient.mesh_table; PHerc0343 is the positive control, his concat mesh with his
  umbilicus. Models: dnative hybrid_3d2d-seed42 hybrid_3d2d-seed43 hecate; depths in -15..15, run d-major.)

Render: render66.py's own functions (render_tifxyz.Surface 'nearest', render_surface, write_zarr with read-back check
and atomic rename), the normal sign from orient.py (refuses a mesh whose chosen sign is below 0.9 outward). The render
goes to data/fl/tmp/<name>/ and is deleted once every requested map exists (kept on a failure, for the resume). The
pieces ensemble.py needs are kept next to the maps first: the valid mask (CT > 0 in all 66 layers, ensemble_maps.py's
definition), the CT histogram of layers 18..48 over it, and CT layer 33 + d per depth (and d = +1).

Chunks: by default read over HTTPS into memory only. --cache-dir DIR uses render66.DiskChunkCache (chunks kept in DIR,
so the windings of one z window share them); with --cache-readonly an existing chunk directory is only read and
missing chunks are fetched into memory, nothing is written there (for data/nieuwlaar/chunks_0343_l0).

Crop (the neighbour-depth check of flagged windows, amendment 1): --crop X0,Y0,X1,Y1 is a rectangle in READING
coordinates of the mesh's d = +1 image (x1, y1 exclusive), widened by --crop-margin on every side for model context,
clipped to the canvas and converted to the raw render frame with his Frame.rect_reading_to_raw; only those canvas
pixels are rendered, sampled exactly as the full render samples them. Outputs go to data/fl/crops/<scroll>/<mesh>/
crop_X0_Y0_X1_Y1/ with crop.json (every box in both frames, and the requested box inside the crop's raw frame).

Maps: ops/nieuwlaar/run_unit.py's own one() with its paths pointed at this mesh or crop, so the CLI calls, env, TMPDIR
trick, shape checks and atomic outputs are the production code, not a copy. One tiling ('plain'). Resumable: an
existing map is skipped; when all exist nothing is rendered. --directions reverse forward also makes the BLIND side
(forward maps, his file names, run_unit.one(..., 'forward')) from the same render; the default is reverse only.

Refuses to start when free disk is under 100 GB or MPS is unavailable. Logs: results/fl/logs/<name>.log (+ .render.log,
per-unit model logs, .last_run.json with this run's render and map seconds), stage timings in screen_timings.tsv.
"""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path('/Users/andreashad04/money/vesuv')
for p in ('ops/fl', 'ops/nieuwlaar', 'public/firstletters'):
    sys.path.insert(0, str(ROOT / p))
import orient                 # noqa: E402
import run_unit as ru         # noqa: E402  production ink runner (main() is not called)
import render66 as r66        # noqa: E402  sets render_tifxyz NUM_LAYERS 66, CENTER 32.5
rt = r66.rt

LOGS = ROOT / 'results/fl/logs'
MIN_FREE = 40e9   # 2026-10-13: was 100e9; the screen holds only transient window caches (a few GB), the disk is being filled by another project
MID = 33


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for f in self.s:
            f.write(x)
            f.flush()

    def flush(self):
        for f in self.s:
            f.flush()


class ReadOnlyDiskCache(rt.ChunkCache):
    """render66.DiskChunkCache's layout, read only: present chunks come from DIR, missing ones over HTTPS into memory."""

    def __init__(self, volume_url, shape, threads, cache_dir):
        super().__init__(volume_url, shape, threads)
        self.dir = Path(cache_dir)

    def _one(self, k):
        cz, cy, cx = rt.unkey(k)
        p = self.dir / str(cz) / str(cy) / str(cx)
        if p.with_name(p.name + '.404').exists():
            return k, None, 0
        if p.exists() and p.stat().st_size == rt.CHUNK_BYTES:
            return k, np.fromfile(p, np.uint8).reshape(rt.CHUNK, rt.CHUNK, rt.CHUNK), 0
        return super()._one(k)


class CropSurface(rt.Surface):
    """canvas pixels [r0:r1, c0:c1] of a full Surface; grid_coords is the full surface's, offset, so every sample is
    the one the full render takes."""

    def __init__(self, full, r0, r1, c0, c1):
        self.__dict__.update(full.__dict__)
        self.full, self.r0, self.c0, self.H, self.W = full, r0, c0, r1 - r0, c1 - c0

    def grid_coords(self, r0, r1, c0, c1):
        return self.full.grid_coords(self.r0 + r0, self.r0 + r1, self.c0 + c0, self.c0 + c1)


def names(scroll, mesh, crop=None):
    name = f'{scroll}_{mesh}' + (f'__{crop}' if crop else '')
    maps = (ROOT / 'data/fl/crops' / scroll / mesh / crop) if crop else (ROOT / 'data/fl/maps' / scroll / mesh)
    return name, ROOT / 'data/fl/tmp' / name, maps


def map_file(scroll, mesh, model, d, crop=None, direction='reverse'):
    """the production file name of a map (run_unit.out_path) of this mesh or crop."""
    name, _, maps = names(scroll, mesh, crop)
    ru.N, ru.MAPS = name, {'plain': maps}
    return ru.out_path('plain', model, d, direction)


def mps_ok(py):
    r = subprocess.run([str(py), '-c', 'import torch; print(torch.backends.mps.is_available())'],
                       capture_output=True, text=True)
    return r.stdout.strip() == 'True', r


def imwrite(path, a):
    import cv2
    tmp = path.with_name(path.stem + '.partial.png')
    if not cv2.imwrite(str(tmp), a, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise IOError(tmp)
    os.replace(tmp, path)


def write_json(path, obj):
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def load_surface(mesh_dir):
    import tifffile
    meta = json.loads((mesh_dir / 'meta.json').read_text())
    X, Y, Z = (tifffile.imread(mesh_dir / f'{c}.tif') for c in 'xyz')
    sc = meta['scale']
    sc = (float(sc[0]), float(sc[1])) if isinstance(sc, (list, tuple)) else (float(sc), float(sc))
    return rt.Surface(X, Y, Z, sc, normal_mode='nearest', px_shift=0.0), X.shape, sc


def render(mesh_dir, out, sign, orient_rec, a, url, label, crop=None):
    """render66.main() with the normal sign, the chunk cache and an optional raw crop (r0, r1, c0, c1) as parameters."""
    rt.RESULTS = out.parent
    rt.LOG = LOGS / f'{label}.render.log'
    me = Path(__file__).resolve()
    rt.log(f'=== screen_unit render pid {os.getpid()} src sha256 {rt.sha256_file(me)}; render66 sha256 '
           f'{rt.sha256_file(Path(r66.__file__))}; renderer sha256 {rt.sha256_file(Path(rt.__file__))}; sign {sign:+d}'
           f'; crop {crop}; cache {a.cache_dir} readonly {a.cache_readonly}')
    shas = {f'{c}.tif': rt.sha256_file(mesh_dir / f'{c}.tif') for c in 'xyz'}
    shape, _ = rt.volume_shape(url)                       # asserts uint8, 128^3 raw chunks
    surf, grid, sc = load_surface(mesh_dir)
    full_hw = [surf.H, surf.W]
    if crop is not None:
        surf = CropSurface(surf, *crop)
    rt.log(f'  mesh {mesh_dir.name} grid {grid} scale {sc} -> canvas {full_hw}, rendering {surf.H} x {surf.W}; '
           f'volume {shape}')
    if a.cache_dir:
        cls = ReadOnlyDiskCache if a.cache_readonly else r66.DiskChunkCache
        cache = cls(url, shape, a.threads, Path(a.cache_dir))
    else:
        cache = rt.ChunkCache(url, shape, a.threads)
    stack, valid, info, _ = rt.render_surface(surf, cache, float(sign), 0.0, a.workers, a.cap_gb * 1e9, label)
    for k in range(r66.N_LAYERS):
        assert not stack[k][~valid].any()
    attrs = dict(render66=dict(convention=dict(layers=r66.N_LAYERS, centre=r66.CENTRE, sign=float(sign),
                                               normal_mode='nearest', normal='normalise(dP/dcolumn x dP/drow)',
                                               transform='identity',
                                               rule='layer k samples p + (k - 32.5) * sign * n; d = k - 33; '
                                                    'sign from orient.py so that +d points away from the axis'),
                               volume=url, mesh=str(mesh_dir), mesh_sha256=shas, scale=sc, orientation=orient_rec,
                               canvas_hw=full_hw, crop_raw_r0_r1_c0_c1=None if crop is None else list(crop),
                               cache=dict(dir=a.cache_dir, readonly=a.cache_readonly),
                               renderer_sha256=rt.sha256_file(Path(rt.__file__)),
                               render66_sha256=rt.sha256_file(Path(r66.__file__)), script_sha256=rt.sha256_file(me),
                               info=info, valid_px=int(valid.sum())))
    r66.write_zarr(stack, out, attrs)
    return stack, attrs


def keep_byproducts(stack, maps, name, depths):
    """valid mask, CT histogram (layers 18..48) and CT layers that ensemble.py needs once the render is gone."""
    H = stack.shape[1]
    valid = np.zeros(stack.shape[1:], bool)
    hist = np.zeros(256, np.int64)
    for r0 in range(0, H, 128):
        blk = stack[:, r0:r0 + 128]
        g = (blk > 0).all(0)                              # ensemble_maps.py: CT > 0 in all 66 layers
        valid[r0:r0 + blk.shape[1]] = g
        hist += np.bincount(blk[MID - 15:MID + 16][:, g].ravel(), minlength=256)
    vp = maps / f'{name}__valid.png'
    v8 = valid.astype(np.uint8) * 255
    if vp.exists():                                       # a re-render must reproduce the first one exactly
        import cv2
        old = cv2.imread(str(vp), cv2.IMREAD_UNCHANGED)
        if old is None or old.shape != v8.shape or not np.array_equal(old, v8):
            raise SystemExit(f'valid mask differs from the kept one: {vp}')
    else:
        imwrite(vp, v8)
    for d in sorted(set(depths) | {1}):
        p = maps / f'{name}__ct__L{MID + d}.png'
        if not p.exists():
            imwrite(p, np.ascontiguousarray(stack[MID + d]))
    write_json(maps / f'{name}__cthist.json', dict(H=int(stack.shape[1]), W=int(stack.shape[2]),
                                                    valid_px=int(valid.sum()), ct_layers=[MID - 15, MID + 15],
                                                    ct_hist=hist.tolist()))


def crop_box(a, mesh_dir, o):
    """--crop (reading, d = +1 image) -> (crop name, raw (r0, r1, c0, c1) with margin, crop.json record)."""
    x0, y0, x1, y1 = (int(v) for v in a.crop.split(','))
    surf, _, _ = load_surface(mesh_dir)
    H, W = surf.H, surf.W
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
        raise SystemExit(f'REFUSED: crop {a.crop} outside the {W} x {H} canvas')
    fr = orient.msi.Frame(H, W, o['flip_lr'], o['flip_ud'])
    m = a.crop_margin
    X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    c0, r0, c1, r1 = (int(v) for v in fr.rect_reading_to_raw(X0, Y0, X1, Y1))
    i0, j0, i1, j1 = (int(v) for v in fr.rect_reading_to_raw(x0, y0, x1, y1))
    cname = f'crop_{x0}_{y0}_{x1}_{y1}'
    rec = dict(name=cname, canvas_hw=[H, W], flip_lr=bool(o['flip_lr']), flip_ud=bool(o['flip_ud']), margin=m,
               box_reading=[x0, y0, x1, y1], box_reading_with_margin=[X0, Y0, X1, Y1],
               box_raw=dict(col0=i0, row0=j0, col1=i1, row1=j1),
               box_raw_with_margin=dict(col0=c0, row0=r0, col1=c1, row1=r1),
               box_in_crop_raw=dict(col0=i0 - c0, row0=j0 - r0, col1=i1 - c0, row1=j1 - r0),
               note='maps of a crop are in the raw render frame of the crop; boxes x1/y1 exclusive')
    return cname, (r0, r1, c0, c1), rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scroll', required=True, choices=list(orient.SCROLLS))
    ap.add_argument('--mesh', required=True)
    ap.add_argument('--models', required=True, nargs='+', choices=list(ru.CKPT))
    ap.add_argument('--depths', required=True, nargs='+', type=int)
    ap.add_argument('--directions', nargs='+', choices=ru.DIRECTIONS, default=['reverse'])
    ap.add_argument('--cache-dir', help='render66.DiskChunkCache directory (default: no disk cache)')
    ap.add_argument('--cache-readonly', action='store_true', help='only read --cache-dir, never write it')
    ap.add_argument('--crop', help='X0,Y0,X1,Y1 in reading coordinates of the d = +1 image')
    ap.add_argument('--crop-margin', type=int, default=256)
    ap.add_argument('--keep-render', action='store_true', help='do not delete the render (verification only)')
    ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cap-gb', type=float, default=8.0)
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    label0 = f'{a.scroll}_{a.mesh}' + ('__crop' if a.crop else '')
    sys.stdout = Tee(sys.__stdout__, open(LOGS / f'{label0}.log', 'a'))
    me = Path(__file__).resolve()
    print(f'\n=== screen_unit pid {os.getpid()} sha256 {hashlib.sha256(me.read_bytes()).hexdigest()} mtime '
          f'{time.ctime(me.stat().st_mtime)}; run_unit sha256 {hashlib.sha256(Path(ru.__file__).read_bytes()).hexdigest()}'
          f'; {time.strftime("%Y-%m-%dT%H:%M:%S")}; args {vars(a)}', flush=True)

    # refusals first
    free = shutil.disk_usage(ROOT).free
    if free < MIN_FREE:
        raise SystemExit(f'REFUSED: free disk {free / 1e9:.1f} GB < {MIN_FREE / 1e9:.0f} GB')
    pys = {ru.PY_VILLA} | ({ru.PY_HEC} if 'hecate' in a.models else set())
    for py in pys:
        ok, r = mps_ok(py)
        if not ok:
            raise SystemExit(f'REFUSED: MPS not available to {py}: {r.stdout!r} {r.stderr[-300:]!r}')
    for m in a.models:
        if not Path(ru.CKPT[m]).exists():
            raise SystemExit(f'REFUSED: checkpoint missing {ru.CKPT[m]}')
    sc = orient.SCROLLS[a.scroll]
    url = sc['volume']
    if 'hecate' in a.models and sc['um'] != 8.64:
        raise SystemExit('REFUSED: run_unit.py prepares Hecate with --native-um 8.64 only')
    for d in a.depths:
        assert -15 <= d <= 15, d
    if a.cache_readonly and not (a.cache_dir and Path(a.cache_dir).is_dir()):
        raise SystemExit('REFUSED: --cache-readonly needs an existing --cache-dir')
    row = [m for m in orient.mesh_table(a.scroll) if m['mesh'] == a.mesh]
    if len(row) != 1:
        raise SystemExit(f'REFUSED: {a.scroll} {a.mesh} not (uniquely) a mesh of orient.mesh_table')
    row = row[0]
    area, mesh_dir = row['area_cm2'], row['dir']
    if not Path(sc['axis']).exists():
        raise SystemExit(f'REFUSED: no axis {sc["axis"]}, run orient.py --scroll {a.scroll} first')
    o = orient.mesh_orientation(mesh_dir, orient.scroll_axis(a.scroll))
    sign = int(o['chosen_sign'])
    print(f'orientation: sign {sign:+d}, outward {o["outward_fraction_chosen"]} (sign +1: '
          f'{o["outward_fraction_sign_plus1"]}, own centroid {o["outward_fraction_own_centroid"]}); reading flips '
          f'lr {o["flip_lr"]} ud {o["flip_ud"]}', flush=True)
    if not o['confident']:
        raise SystemExit(f'REFUSED: orientation not confident ({o["outward_fraction_chosen"]} < {orient.MIN_CONFIDENT})')
    inv = orient.OUT / f'inventory_{a.scroll}.csv'
    if inv.exists():
        ir = [r for r in csv.DictReader(open(inv)) if r['mesh'] == a.mesh]
        if ir and int(ir[0]['chosen_sign']) != sign:
            raise SystemExit(f'REFUSED: sign {sign} disagrees with {inv} ({ir[0]["chosen_sign"]})')
    cname = crop = None
    if a.crop:
        cname, crop, crec = crop_box(a, mesh_dir, o)
        crop_area = (crop[1] - crop[0]) * (crop[3] - crop[2]) * (sc['um'] * 1e-4) ** 2
        print(f'crop {cname}: raw rows {crop[0]}..{crop[1]} cols {crop[2]}..{crop[3]} (with margin '
              f'{a.crop_margin}), {crop_area:.3f} cm2 of canvas', flush=True)
        area = crop_area                                  # timings per cm2 of what is actually rendered

    # point the production runner at this mesh or crop
    name, work, maps = names(a.scroll, a.mesh, cname)
    out = work / f'{name}_66.zarr'
    ru.N, ru.DATA, ru.RENDER, ru.MAPS = name, work, {'plain': out}, {'plain': maps}
    ru.TIMINGS = LOGS / f'{name}.unit_timings.tsv'
    maps.mkdir(parents=True, exist_ok=True)
    if a.crop:
        write_json(maps / 'crop.json', crec)
    lock = ROOT / 'data/fl/tmp' / f'{name}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            os.kill(int(lock.read_text().strip()), 0)
            raise SystemExit(f'REFUSED: another screen_unit holds {lock}')
        except (ProcessLookupError, ValueError):
            pass
    lock.write_text(str(os.getpid()))
    tim = LOGS / 'screen_timings.tsv'
    stage = 'crop_' if a.crop else ''
    summary = dict(name=name, area_cm2=area, render_s=0.0, maps_s=0.0, made=[], failed=[], pid=os.getpid(),
                   start=time.strftime('%Y-%m-%dT%H:%M:%S'))

    def timing(st, model, d, secs, rc, t0):
        new = not tim.exists()
        with open(tim, 'a') as f:
            if new:
                f.write('scroll\tmesh\tarea_cm2\tstage\tmodel\td\tseconds\ts_per_cm2\trc\tstart\n')
            f.write(f'{a.scroll}\t{a.mesh}{"/" + cname if cname else ""}\t{area:.4g}\t{stage}{st}\t{model}\t{d}\t'
                    f'{secs:.1f}\t{secs / area:.1f}\t{rc}\t{time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))}\n')

    try:
        need = [(dr, d, m) for dr in a.directions for d in a.depths for m in a.models
                if not (ru.out_path('plain', m, d, dr).exists() and ru.out_path('plain', m, d, dr).stat().st_size > 0)]
        print(f'{len(need)} of {len(a.directions) * len(a.depths) * len(a.models)} maps to make; maps -> {maps}', flush=True)
        if need:
            rj = out.with_name(out.name + '.render.json')
            if (out / '0').exists() and rj.exists() and json.loads(rj.read_text()).get('convention', {}).get('sign') == sign:
                print(f'reusing render {out}', flush=True)
                import zarr
                keep_byproducts(np.asarray(zarr.open_group(str(out), mode='r')['0'][:]), maps, f'{name}_66', a.depths)
            else:
                t0 = time.time()
                stack, attrs = render(mesh_dir, out, sign, o, a, url, name, crop)
                keep_byproducts(stack, maps, f'{name}_66', a.depths)
                secs = time.time() - t0
                rec = dict(attrs['render66'], out=str(out), seconds=round(secs, 1), shape=list(stack.shape),
                           area_cm2=area)
                del stack
                for p in (rj, LOGS / f'{name}.render.json'):
                    write_json(p, rec)
                timing('render', '', '', secs, 0, t0)
                summary['render_s'] = round(secs, 1)
                print(f'render {secs:.1f} s ({secs / area:.1f} s/cm2), {rec["info"]}', flush=True)
            bad = 0
            for dr, d, m in need:
                t0 = time.time()
                rc = ru.one('plain', m, d, dr)
                secs = time.time() - t0
                ul = work / 'logs' / ru.log_name('plain', m, d, dr)
                if ul.exists():
                    with open(LOGS / f'{name}__{m}_d{d:+d}{ru.tag(dr)}.log', 'a') as f:
                        f.write(ul.read_text())
                    ul.unlink()
                fw = '' if dr == 'reverse' else f' {dr}'
                timing('model', m + ('' if dr == 'reverse' else f':{dr}'), d, secs, rc, t0)
                summary['maps_s'] = round(summary['maps_s'] + secs, 1)
                summary['made' if rc == 0 else 'failed'].append(f'{m} d{d:+d}{fw}')
                print(f'model {m} d{d:+d}{fw}: rc {rc}, {secs:.1f} s ({secs / area:.1f} s/cm2)', flush=True)
                bad |= rc != 0
            if bad:
                raise SystemExit(f'FAILED units; render kept at {out} for the resume')
        if a.keep_render:
            print(f'render kept (--keep-render): {out}', flush=True)
        elif work.exists():
            shutil.rmtree(work)
            print(f'render deleted: {work}', flush=True)
    finally:
        lock.unlink(missing_ok=True)
        summary['end'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        write_json(LOGS / f'{name}.last_run.json', summary)
    print('done', flush=True)


if __name__ == '__main__':
    main()
