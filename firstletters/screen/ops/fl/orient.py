#!/usr/bin/env python3
"""orient.py: per-mesh render normal sign so that +d points AWAY from the scroll axis, as in Nieuwlaar's PHerc0343 run
(his sidecar outputs/concat_w047-w048_R5B2_z9500-11000.json: orientation.normal_outward_fraction 0.9973, no flip).

Normal: the renderer's own, render_tifxyz.vertex_normals (VC grid_normal_int, n = dP/dcolumn x dP/drow at the stored
vertex, which render66 uses in 'nearest' mode), on the renderer's validity (x != -1 and finite). Outward fraction = share
of the vertices that have a normal with n . r > 0, r = the horizontal unit vector from the axis (interpolated at the
vertex z) to the vertex. Chosen sign = +1 when the sign +1 fraction is above 0.5, else -1; a render with that sign has
+d outward. His measure_orientation (make_submission_image.py) is run on the same mesh and axis as a cross-check, and
gives the reading flips (flip_lr, flip_ud) that ensemble.py uses.

Axis. PHerc0800 has no published umbilicus: the meshes' meta.json fit_config names none, the organisers' lasagna json
(PHerc0800/representations/predictions/lasagna/20250521135224-lasagna-20260419180421/PHerc0800.lasagna.json) has
umbilicus_json "", and the bucket's PHerc0800/ prefix holds no umbilicus file (photos, representations, segments,
volumes; checked 2026-10-12). So the axis is ESTIMATED from the meshes themselves: every pscamillo mesh is one full
winding whose node rows are closed loops around the fitter's umbilicus, so the umbilicus lies inside the INNERMOST
winding of each window. The centroid of every row of that innermost mesh that sweeps at least 300 degrees around it is
one axis sample at the row's median z, and the axis at z is the median of the samples within +-64 voxels, on a 64 voxel
z grid. Outer windings are NOT used: in this crushed scroll their loop centroids sit up to hundreds of voxels from the
inner ones, and a first version that pooled all wraps put the axis outside 7 of the 19 w020 loops (rows sweeping 5 to
39 degrees around it). Check: every row of every mesh should sweep about 360 degrees around the axis. Written in his
umbilicus json format (control_points x, y, z), so his load_umbilicus reads it.

  orient.py --control              Nieuwlaar's concat mesh with his umbilicus: must give about 0.997 for sign +1, and
                                   about 0.003 for the same mesh with its columns reversed (the guard watched failing)
  orient.py --scroll PHerc0800     results/fl/axis_PHerc0800.json and results/fl/inventory_PHerc0800.csv
"""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path('/Users/andreashad04/money/vesuv')
PSC = ROOT / 'data/fl/pscamillo'
OUT = ROOT / 'results/fl'
NLREPO = ROOT / 'results/nieuwlaar/repo'
sys.path.insert(0, str(ROOT / 'public/firstletters'))
sys.path.insert(0, str(NLREPO / 'ink/scripts'))
import render_tifxyz as rt            # noqa: E402  the renderer's normals
import make_submission_image as msi   # noqa: E402  his load_mesh, load_umbilicus, measure_orientation

CONTROL_MESH = NLREPO / 'outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz'
CONTROL_UMB = NLREPO / 'inputs/umbilicus_PHerc0343.json'
CONTROL_SIDECAR = NLREPO / 'outputs/concat_w047-w048_R5B2_z9500-11000.json'
MIN_CONFIDENT = 0.9                   # the chosen sign must reach this outward fraction, else the mesh is flagged
AXIS_STEP, AXIS_HALF, MIN_SWEEP, MIN_ROW_NODES, MIN_SAMPLES = 64, 64, 300.0, 16, 5
BUCKET = 'https://vesuvius-challenge-open-data.s3.amazonaws.com'
PATCHES = ROOT / 'data/fl/patches'   # local sheet patches, <scroll>/<name>.tifxyz (research/uncontested/patch/patches.py)
SCROLLS = {   # volume (uint8 OME-Zarr level 0, 128^3 raw chunks), voxel size (um), axis json, mesh source
    'PHerc0800': dict(volume=f'{BUCKET}/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr/0', um=8.64,
                      axis=OUT / 'axis_PHerc0800.json', meshes='pscamillo'),
    # the positive control: his own concat mesh and umbilicus, through the identical code
    'PHerc0343': dict(volume=f'{BUCKET}/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr/0', um=8.64,
                      axis=CONTROL_UMB, meshes=[CONTROL_MESH]),
    # uncontested First Letters scroll: no umbilicus exists; the axis is the COARSE, unvalidated cross-section centroid
    # and the meshes are the validated local patches (research/uncontested/patch/validation.json)
    'PHerc0483B': dict(volume=f'{BUCKET}/PHerc0483B/volumes/20251124083638-8.640um-1.2m-116keV-masked.zarr/0', um=8.64,
                       axis=OUT / 'axis_PHerc0483B.json', meshes='patches'),
    # the second uncontested First Letters scroll, treated the same way: COARSE axis, validated local patches from the
    # unchanged public sheet_patches.py (research/uncontested/patch_0175B/validation.json)
    'PHerc0175B': dict(volume=f'{BUCKET}/PHerc0175B/volumes/20250521125822-8.640um-1.2m-116keV-masked.zarr/0', um=8.64,
                       axis=OUT / 'axis_PHerc0175B.json', meshes='patches'),
    # the third uncontested First Letters scroll, treated the same way: COARSE axis, validated local patches from the
    # unchanged public sheet_patches.py (research/uncontested/patch_0490A/validation.json)
    'PHerc0490A': dict(volume=f'{BUCKET}/PHerc0490A/volumes/20250521151210-8.640um-1.2m-116keV-masked.zarr/0', um=8.64,
                       axis=OUT / 'axis_PHerc0490A.json', meshes='patches'),
}


def mesh_table(scroll):
    """[dict(scroll, mesh, window_z, wrap, area_cm2, gate_verdict, dir)] for every screenable mesh of a scroll; area as
    pscamillo computes it (meta.json area_vx2 times the voxel size squared)."""
    s = SCROLLS[scroll]
    if s['meshes'] == 'pscamillo':
        return [dict(scroll=scroll, mesh=Path(r['path']).name, window_z=int(r['window_z']), wrap=r['wrap'],
                     area_cm2=float(r['area_cm2']), gate_verdict=r['gate_verdict'], dir=PSC / r['path'])
                for r in csv.DictReader(open(PSC / 'data/index.csv')) if r['scroll'] == scroll]
    out = []
    for d in (sorted((PATCHES / scroll).glob('*.tifxyz')) if s['meshes'] == 'patches' else s['meshes']):
        meta = json.loads((d / 'meta.json').read_text())
        Z = tifffile.imread(d / 'z.tif')
        out.append(dict(scroll=scroll, mesh=d.name[:-7] if d.name.endswith('.tifxyz') else d.name,
                        window_z=int(np.floor(Z[Z > 0].min())), wrap='', gate_verdict='', dir=d,
                        area_cm2=round(float(meta['area_vx2']) * (s['um'] * 1e-4) ** 2, 2)))
    return out


def scroll_axis(scroll):
    return msi.load_umbilicus(str(SCROLLS[scroll]['axis']))


def load(mesh_dir):
    X, Y, Z = (tifffile.imread(Path(mesh_dir) / f'{c}.tif').astype(np.float64) for c in 'xyz')
    valid = (X != -1.0) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)      # rt.Surface's validity
    return X, Y, Z, valid


def outward(X, Y, Z, valid, axis, sign=+1.0):
    """outward fraction of the render normal (times sign) around axis = (uz, ux, uy)."""
    P = np.stack([X, Y, Z], -1)
    n, ok = rt.vertex_normals(P, valid)
    n, p = n[ok] * float(sign), P[ok]
    cx, cy = np.interp(p[:, 2], axis[0], axis[1]), np.interp(p[:, 2], axis[0], axis[2])
    r = np.stack([p[:, 0] - cx, p[:, 1] - cy], -1)
    r /= np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
    dot = n[:, 0] * r[:, 0] + n[:, 1] * r[:, 1]
    return dict(outward_fraction=float((dot > 0).mean()), dot_median=float(np.median(dot)), n_vertices=int(dot.size))


def row_samples(X, Y, Z, valid, centre=None):
    """per node row with enough nodes: (median z, centroid x, centroid y, sweep in degrees around the centroid, or
    around centre(z) -> (cx, cy) when given)."""
    out = []
    for i in range(X.shape[0]):
        v = valid[i]
        if v.sum() < MIN_ROW_NODES:
            continue
        x, y, z = X[i, v], Y[i, v], Z[i, v]
        zm = float(np.median(z))
        cx, cy = (float(x.mean()), float(y.mean())) if centre is None else centre(zm)
        t = np.unwrap(np.arctan2(y - cy, x - cx))
        out.append((zm, float(x.mean()), float(y.mean()), float(abs(np.degrees(t[-1] - t[0])))))
    return out


def estimate_axis(mesh_dirs):
    S = []
    for d in mesh_dirs:
        S += [s for s in row_samples(*load(d)) if s[3] >= MIN_SWEEP]
    S = np.array(S)
    pts = []
    for z in np.arange(np.floor(S[:, 0].min() / AXIS_STEP) * AXIS_STEP, S[:, 0].max() + AXIS_STEP, AXIS_STEP):
        m = np.abs(S[:, 0] - z) <= AXIS_HALF
        if m.sum() < MIN_SAMPLES:
            continue
        mx, my = float(np.median(S[m, 1])), float(np.median(S[m, 2]))
        pts.append(dict(x=round(mx, 1), y=round(my, 1), z=int(z), n=int(m.sum()),
                        mad_x=round(float(np.median(np.abs(S[m, 1] - mx))), 1),
                        mad_y=round(float(np.median(np.abs(S[m, 2] - my))), 1)))
    return pts, len(S)


def axis_path(scroll):
    return OUT / f'axis_{scroll}.json'


def load_axis(scroll):
    return msi.load_umbilicus(str(axis_path(scroll)))


def mesh_orientation(mesh_dir, axis):
    """everything screen_unit.py and the inventory need for one mesh."""
    X, Y, Z, valid = load(mesh_dir)
    plus, minus = outward(X, Y, Z, valid, axis, +1.0), outward(X, Y, Z, valid, axis, -1.0)
    sign = +1 if plus['outward_fraction'] > 0.5 else -1
    chosen = plus if sign > 0 else minus
    own = np.array([s for s in row_samples(X, Y, Z, valid) if s[3] >= MIN_SWEEP])
    own_frac = None
    if len(own) >= 2:
        o = np.argsort(own[:, 0])
        own_frac = outward(X, Y, Z, valid, (own[o, 0], own[o, 1], own[o, 2]))['outward_fraction']
    sweeps = [s[3] for s in row_samples(X, Y, Z, valid, centre=lambda z: (float(np.interp(z, axis[0], axis[1])),
                                                                       float(np.interp(z, axis[0], axis[2]))))]
    Xh, Yh, Zh, okh, _ = msi.load_mesh(str(mesh_dir))
    _, his = msi.measure_orientation(Xh, Yh, Zh, okh, axis)
    return dict(grid=f'{X.shape[0]}x{X.shape[1]}', valid_nodes=int(valid.sum()),
                outward_fraction_sign_plus1=round(plus['outward_fraction'], 4),
                dot_median_sign_plus1=round(plus['dot_median'], 4),
                outward_fraction_own_centroid=None if own_frac is None else round(own_frac, 4),
                his_outward_fraction=his['normal_outward_fraction'],
                chosen_sign=sign, outward_fraction_chosen=round(chosen['outward_fraction'], 4),
                confident=bool(chosen['outward_fraction'] >= MIN_CONFIDENT),
                axis_sweep_deg_median=round(float(np.median(sweeps)), 1) if sweeps else None,
                flip_lr=his['flip_lr'], flip_ud=his['flip_ud'])


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(text)
    os.replace(tmp, path)


def control():
    X, Y, Z, valid = load(CONTROL_MESH)
    umb = msi.load_umbilicus(str(CONTROL_UMB))
    mine = outward(X, Y, Z, valid, umb, +1.0)['outward_fraction']
    rev = outward(X[:, ::-1], Y[:, ::-1], Z[:, ::-1], valid[:, ::-1], umb, +1.0)['outward_fraction']
    minus = outward(X, Y, Z, valid, umb, -1.0)['outward_fraction']
    Xh, Yh, Zh, okh, _ = msi.load_mesh(str(CONTROL_MESH))
    _, his = msi.measure_orientation(Xh, Yh, Zh, okh, umb)
    published = json.loads(CONTROL_SIDECAR.read_text())['orientation']['normal_outward_fraction']
    checks = {'mine_sign_plus1_about_0.997': abs(mine - 0.997) < 0.005,
              'his_function_equals_his_sidecar': his['normal_outward_fraction'] == published,
              'columns_reversed_flips_it': abs(rev - (1 - mine)) < 0.002 and rev < 0.01,
              'sign_minus1_flips_it': abs(minus - (1 - mine)) < 0.002}
    res = dict(mesh=str(CONTROL_MESH), umbilicus=str(CONTROL_UMB),
               umbilicus_sha256=hashlib.sha256(CONTROL_UMB.read_bytes()).hexdigest(),
               mine_sign_plus1=round(mine, 4), mine_sign_minus1=round(minus, 4), mine_columns_reversed=round(rev, 4),
               his_measure_orientation=his['normal_outward_fraction'], his_sidecar=published,
               checks=checks, all_pass=all(checks.values()))
    write_atomic(OUT / 'orient_control.json', json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res['all_pass']


def inventory(scroll):
    rows = [r for r in csv.DictReader(open(PSC / 'data/index.csv')) if r['scroll'] == scroll]
    dirs = [PSC / r['path'] for r in rows]
    inner = {}
    for r in rows:                              # innermost winding (smallest wrap number) of every z window
        if r['window_z'] not in inner or int(r['wrap'][1:]) < int(inner[r['window_z']]['wrap'][1:]):
            inner[r['window_z']] = r
    pts, n_samples = estimate_axis([PSC / r['path'] for r in inner.values()])
    ax = dict(control_points=pts, scroll=scroll, units='voxels of the 8.64 um volume (x = column, y = row, z = slice)',
              source='ESTIMATED from the pscamillo meshes (no published umbilicus exists for this scroll)',
              method=f'median of node-row loop centroids (rows sweeping >= {MIN_SWEEP} deg) of the innermost winding '
                     f'of each window ({sorted(set(r["wrap"] for r in inner.values()))}) within +-{AXIS_HALF} voxels '
                     f'of each {AXIS_STEP}-voxel z step; {n_samples} row samples from {len(inner)} meshes',
              orient_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_atomic(axis_path(scroll), json.dumps(ax, indent=1))
    axis = load_axis(scroll)
    out = []
    for r, d in zip(rows, dirs):
        o = mesh_orientation(d, axis)
        out.append(dict(scroll=scroll, mesh=Path(r['path']).name, window_z=r['window_z'], wrap=r['wrap'],
                        area_cm2=r['area_cm2'], gate_verdict=r['gate_verdict'], **o, path=r['path']))
    out.sort(key=lambda r: (int(r['window_z']), r['wrap']))
    tmp = OUT / f'inventory_{scroll}.csv.partial'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    os.replace(tmp, OUT / f'inventory_{scroll}.csv')
    f1 = np.array([r['outward_fraction_sign_plus1'] for r in out])
    neg = sum(r['chosen_sign'] < 0 for r in out)
    print(f'{scroll}: {len(out)} meshes; axis {len(pts)} points z {pts[0]["z"]}..{pts[-1]["z"]} from {n_samples} rows, '
          f'median MAD x {np.median([p["mad_x"] for p in pts]):.1f} y {np.median([p["mad_y"] for p in pts]):.1f} vx')
    print(f'  sign +1 outward fraction: min {f1.min():.4f} median {np.median(f1):.4f} max {f1.max():.4f}')
    print(f'  chosen sign -1: {neg} of {len(out)}; not confident (< {MIN_CONFIDENT}): '
          f'{sum(not r["confident"] for r in out)}; own-centroid sign disagrees: '
          f'{sum((r["outward_fraction_own_centroid"] is not None) and ((r["outward_fraction_own_centroid"] > 0.5) != (r["chosen_sign"] > 0)) for r in out)}; '
          f'his sign disagrees: {sum((r["his_outward_fraction"] > 0.5) != (r["chosen_sign"] > 0) for r in out)}')
    sw = [r['axis_sweep_deg_median'] for r in out]
    print(f'  row sweep around the axis (deg): min {min(sw)} median {np.median(sw):.1f}; flips (lr, ud): '
          f'{sorted(set((r["flip_lr"], r["flip_ud"]) for r in out))}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--control', action='store_true')
    ap.add_argument('--scroll')
    a = ap.parse_args()
    me = Path(__file__).resolve()
    print(f'orient pid {os.getpid()} sha256 {hashlib.sha256(me.read_bytes()).hexdigest()} '
          f'mtime {time.ctime(me.stat().st_mtime)}', flush=True)
    ok = True
    if a.control:
        ok = control()
    if a.scroll:
        inventory(a.scroll)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
