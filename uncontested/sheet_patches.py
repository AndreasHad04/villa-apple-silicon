#!/usr/bin/env python3
"""sheet_patches.py: local sheet patches of a scroll that has no umbilicus, traced through the m7 surface prediction.

Every default reproduces the PHerc0483B run (z band 4992 to 6512, h keys every 16 slices, seed key 5760, 8.64 um).
Inputs are open data: the m7 surface prediction at level 0 (OME-Zarr, nonzero = predicted sheet) and the surface tracks
that villa's extract_surface_tracks.py makes from it, exported to one npz by export_tracks.py next to this file.
Patches are tifxyz directories (x.tif, y.tif, z.tif float32, -1 where invalid, plus meta.json), one vertex every 20
voxels, which villa's tools and firstletters/render66.py read.

Usage, from the repository root (every path is an argument; the file names below are the defaults):

  python uncontested/export_tracks.py PHerc0483B_tracks.dbm tracks_export.npz
  python uncontested/sheet_patches.py axis --tracks tracks_export.npz --axis axis.json --sections level5_sections.npz
  python uncontested/sheet_patches.py build --tracks tracks_export.npz --m7 sp_m7.zarr/0 --axis axis.json \\
      --sections level5_sections.npz --out patches --record build.json
  python uncontested/sheet_patches.py validate --tracks tracks_export.npz --m7 sp_m7.zarr/0 --axis axis.json \\
      --out patches --record build.json --validation validation.json
  python firstletters/render66.py --mesh patches/<name>.tifxyz --volume <volume>/0 --out renders/<name>_66.zarr
  python uncontested/sheet_patches.py renders --out patches --renders renders --validation validation.json
  python uncontested/sheet_patches.py compare patches/<name>.tifxyz other/<name>.tifxyz

  build --key K seeds at another h key; build --key K --candidate I --name N writes candidate I of key K only.
  axis reads level 5 of --volume over HTTPS (about 50 MB for PHerc0483B). CPU only. build reads the h tracks only
  (about 2 GB resident on PHerc0483B); validate also reads the vertical tracks.

Method. The axis is COARSE: the centroid of the filled masked-CT cross-section at level 5, used only to choose the
outward sign of each patch, never to unroll. A seed is a long h track at the seed key, extended only through clean
end-to-end breaks (exactly one other track end nearby, never at a junction) up to about 2 cm, and kept when its
geometry is sheet-like (radius fraction, normal sign consistency, alignment with the radial direction, spacing to the
neighbouring sheet). The walk moves every sample of the curve along its in-plane normal only, every STEP_Z slices: a ray
of +-RAY voxels through the m7 band finds the run of predicted voxels nearest to the sample. A sample is accepted when
the run centre is within min(DMAX, FRAC_SP x the distance to the neighbouring run), the run is at most WMAX wide and the
neighbouring run is at least SP_MIN away; displacement outliers are dropped, gaps up to GAP samples are interpolated,
and a step is refused when fewer than MIN_MATCH of the valid samples are accepted. Where the second difference between
h keys 16 slices apart exceeds K2K_MAX (a switch to a neighbouring sheet), those columns are cut from that key outward.
"""
import argparse, hashlib, json, os, shutil, sys, time, urllib.error, urllib.request
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'firstletters'))
import render_tifxyz as rt  # noqa: E402  the renderer's vertex normals (villa's grid_normal_int)

VOLUME = ('https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0483B/volumes/'
          '20251124083638-8.640um-1.2m-116keV-masked.zarr')
SCROLL, BAND, Z0, UM, L5 = 'PHerc0483B', (4992, 6512, 16), 5760, 8.64, 32
KSTEP = BAND[2]
KEYS = list(range(BAND[0], BAND[1] + 1, KSTEP))
DS, COL_EVERY, ROW_EVERY, SCALE = 2.0, 10, 20, 0.05     # output vertex every 20 voxels
DMAX, FRAC_SP, SP_MIN, SP_R = 4.0, 0.45, 6.0, 80.0    # DMAX per walk step
MIN_MATCH, GAP, GAP_END, JUMP, MIN_KEEP = 0.65, 50, 25, 3.0, 0.5
K2K_MAX, JUMP_PAD = 8.0, 10         # bar on the second difference between h keys, about half the sheet spacing
RAY, WMAX = 24.0, 7.0                # normal ray half-length and the widest m7 run taken as one sheet (voxels)
OFFS = np.arange(-RAY, RAY + 0.25, 0.5)
MED_WIN, STEP_Z = 51, 4             # outlier median window (samples); walk step in slices
TARGET, MAX_SEED, MIN_SEED, END_GAP = 2300.0, 2600.0, 800.0, 15.0
RFRAC, MIN_SGN, MIN_DOT, MIN_SPOK = (0.3, 0.9), 0.95, 0.5, 0.9
SEP_DEG, MAX_PATCHES, MAX_WALK, MIN_OUT, MIN_AREA = 30.0, 6, 64, 0.9, 0.25
MIN_CONFIDENT = 0.9                  # the chosen outward sign must reach this fraction of vertices
COLS = {'h': (1, 2), 'vx': (0, 1), 'vy': (0, 2)}        # 2D coordinates per kind: (y, x), (z, y), (z, x)
CRIT = dict(on_sheet_vertices_min=0.9, key_second_difference_max_max=K2K_MAX, vertical_on_track_vs_control=2.0,
            switch_events='reported, not gated: the detector is uncalibrated and fires the same way when the '
                          'vertical skeleton crosses a bridge between touching sheets',
            outward_fraction_min=MIN_OUT)
ON_SHEET_RULE = ('a vertex counts as on-sheet when sp_m7 level 0 has a nonzero voxel within Chebyshev distance 2 of '
                 'its rounded position (a 5 x 5 x 5 box)')


def configure(a):
    global SCROLL, BAND, KSTEP, KEYS, Z0, UM
    SCROLL, BAND, Z0, UM = a.scroll, tuple(a.band), a.key, a.voxel_um
    KSTEP = BAND[2]
    KEYS = list(range(BAND[0], BAND[1] + 1, KSTEP))
    if Z0 not in KEYS or KSTEP % STEP_Z or KSTEP % 2:
        raise SystemExit(f'--key {Z0} must be an h key of --band {BAND}, whose step must be even and a multiple of '
                         f'{STEP_Z}')


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(text)
    os.replace(tmp, path)


def savez_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.partial.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def get(url, tries=4):
    err = None
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:      # network error: retry
            err = e
        time.sleep(2 + 3 * t)
    raise SystemExit(f'fetch failed {url}: {err}')


def densify(pts, trk):
    """polylines (contiguous per track id) resampled to at most 1 voxel between points; returns points, track ids."""
    same = trk[1:] == trk[:-1]
    a, b, tid = pts[:-1][same], pts[1:][same], trk[:-1][same]
    n = np.maximum(np.ceil(np.linalg.norm(b - a, axis=1)).astype(np.int64), 1)
    rep = np.repeat(np.arange(len(a)), n)
    t = ((np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n))[:, None]
    last = np.r_[~same, True]
    return np.vstack([a[rep] + t * (b - a)[rep], pts[last]]), np.r_[tid[rep], trk[last]]


class Tracks:
    def __init__(self, path, kinds=tuple(COLS)):
        d = np.load(path)
        self.k = {}
        for kind in kinds:
            P, T, K = d[f'{kind}_pts'], d[f'{kind}_trk'], d[f'{kind}_key']
            assert np.all(np.diff(T) >= 0), kind
            self.k[kind] = dict(P=P, T=T, K=K, start=np.r_[0, np.cumsum(np.bincount(T, minlength=len(K)))])
        self.cache = {}

    def ids(self, kind, val):
        return np.nonzero(self.k[kind]['K'] == val)[0]

    def track(self, t, kind='h'):
        s = self.k[kind]
        return s['P'][s['start'][t]:s['start'][t + 1]][:, COLS[kind]].astype(np.float64)

    def dense(self, val, kind='h'):
        key = (kind, int(val))
        if key not in self.cache:
            s, ids = self.k[kind], self.ids(kind, val)
            if not len(ids):
                self.cache[key] = None
            else:
                sel = np.concatenate([np.arange(s['start'][t], s['start'][t + 1]) for t in ids])
                pts, dt = densify(s['P'][sel][:, COLS[kind]].astype(np.float64), s['T'][sel])
                self.cache[key] = (pts, dt, cKDTree(pts))
        return self.cache[key]


def arc(poly):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]


def resample(poly, own, ds=DS):
    s = arc(poly)
    u = np.arange(0.0, s[-1] + 1e-9, ds)
    P = np.stack([np.interp(u, s, poly[:, 0]), np.interp(u, s, poly[:, 1])], -1)
    return P, own[np.clip(np.searchsorted(s, u, side='right') - 1, 0, len(poly) - 1)]


def frames(P):
    g = ndimage.uniform_filter1d(np.gradient(P, axis=0), 7, axis=0, mode='nearest')
    T = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    return T, np.stack([-T[:, 1], T[:, 0]], -1)


def spacing(S, T, N, own, keyd, every=4):
    """distance along the in-plane normal to the nearest point of another track (the neighbouring sheet), capped at SP_R."""
    pts, dt, tree = keyd
    m = len(S)
    idx = np.unique(np.r_[np.arange(0, m, every), m - 1])
    val = np.full(len(idx), SP_R)
    for j, (i, L) in enumerate(zip(idx, tree.query_ball_point(S[idx], r=SP_R))):
        if not L:
            continue
        L = np.asarray(L)
        L = L[dt[L] != own[i]]
        if not len(L):
            continue
        q = pts[L] - S[i]
        t, w = q @ N[i], np.abs(q @ T[i])
        k = (w < 3.0) & (np.abs(t) > 2.0)
        if k.any():
            val[j] = np.abs(t[k]).min()
    return np.interp(np.arange(m), idx, val)


def snap(m7, z, pred, N):
    """runs of predicted m7 voxels along the in-plane normal through each predicted position: offset of the centre of
    the run nearest to it, that run's width, and the distance to the nearest other run centre (the neighbouring sheet)."""
    vals = m7.plane_values(z, pred[:, None, :] + OFFS[None, :, None] * N[:, None, :])
    m = len(pred)
    tc, width, sp = np.full(m, np.nan), np.full(m, np.nan), np.full(m, np.nan)
    for i in range(m):
        dv = np.diff(np.r_[0, vals[i].astype(np.int8), 0])
        st, en = np.nonzero(dv == 1)[0], np.nonzero(dv == -1)[0] - 1
        wd = OFFS[en] - OFFS[st] + 0.5
        keep = wd >= 1.5                                   # single-voxel specks are not sheets
        st, en, wd = st[keep], en[keep], wd[keep]
        if not len(st):
            continue
        cen = (OFFS[st] + OFFS[en]) / 2
        k = int(np.argmin(np.abs(cen)))
        if st[k] == 0 or en[k] == len(OFFS) - 1:          # the run reaches the end of the ray: no edge there
            continue
        others = np.abs(np.delete(cen, k) - cen[k])
        tc[i], width[i], sp[i] = cen[k], wd[k], (others.min() if len(others) else 2 * RAY)
    return tc, width, sp


def step(cur, valid, z, m7, keyd):
    """cur: positions of every sample (interior holes carry interpolated positions); valid: on the sheet at the previous
    step. Samples between the first and last valid one are snapped, so a hole is re-acquired when the band returns."""
    idx = np.nonzero(valid)[0]
    span = np.arange(idx[0], idx[-1] + 1)
    T, N = frames(cur)
    tc, width, sp = snap(m7, z, cur[span], N[span])
    tc0, sp0, w0 = np.nan_to_num(tc, nan=1e9), np.nan_to_num(sp), np.nan_to_num(width, nan=1e9)
    acc = (np.abs(tc0) <= np.minimum(DMAX, FRAC_SP * sp0)) & (sp0 >= SP_MIN) & (w0 <= WMAX)
    disp = np.where(acc, tc0, 0.0)[:, None] * N[span]
    a_idx = np.nonzero(acc)[0]
    n_out = 0
    if len(a_idx) >= 5:          # displacement outliers among accepted samples
        med = np.stack([ndimage.median_filter(disp[a_idx, c], size=MED_WIN, mode='nearest') for c in (0, 1)], -1)
        out = np.linalg.norm(disp[a_idx] - med, axis=1) > JUMP
        acc[a_idx[out]] = False
        n_out = int(out.sum())
    accf = np.zeros(len(cur), bool)
    accf[span[acc]] = True
    frac = float(accf[valid].mean())
    st = dict(valid=int(valid.sum()), accepted=int(accf.sum()), frac=round(frac, 4), outliers=n_out,
              reacquired=int((accf & ~valid).sum()), no_run=int(np.isnan(tc).sum()))
    if acc.any():
        st.update(offset_median=round(float(np.median(np.abs(tc[acc]))), 2), width_median=round(float(np.median(width[acc])), 1),
                  sp_median=round(float(np.median(sp[acc])), 1))
        if keyd is not None:
            dtr, _ = keyd[2].query(cur[span[acc]] + disp[acc])
            st.update(track_within3=round(float((dtr <= 3).mean()), 4), track_d_median=round(float(np.median(dtr)), 2))
    if frac < MIN_MATCH:
        st['refused'] = f'accepted {frac:.3f} < {MIN_MATCH}'
        return None, None, None, st
    full = np.full((len(cur), 2), np.nan)
    full[span[acc]] = disp[acc]
    nv = accf.copy()
    insp = np.zeros(len(cur), bool)
    insp[span] = True
    lab, n = ndimage.label(insp & ~accf)
    filled = 0
    for r in range(1, n + 1):
        ii = np.nonzero(lab == r)[0]
        lo, hi = ii[0] - 1, ii[-1] + 1
        left, right = lo >= 0 and accf[lo], hi < len(cur) and accf[hi]
        if len(ii) > (GAP if (left and right) else GAP_END) or not (left or right):
            continue
        if left and right:
            w = ((ii - lo) / (hi - lo))[:, None]
            full[ii] = (1 - w) * full[lo] + w * full[hi]
        else:
            full[ii] = full[lo] if left else full[hi]
        nv[ii] = True
        filled += len(ii)
    st['filled'] = filled
    new = cur.copy()
    new[nv] = cur[nv] + full[nv]
    vidx = np.nonzero(nv)[0]
    if len(vidx) >= 2:           # interior holes follow the interpolated displacement, invalid until re-acquired
        hole = np.arange(vidx[0], vidx[-1] + 1)
        hole = hole[~nv[hole]]
        if len(hole):
            dv = new[vidx] - cur[vidx]
            new[hole] = cur[hole] + np.stack([np.interp(hole, vidx, dv[:, 0]), np.interp(hole, vidx, dv[:, 1])], -1)
    return new, nv, accf, st


def walk(trk, seed, m7):
    n0 = len(seed)
    curves = {Z0: (seed.copy(), np.ones(n0, bool), np.ones(n0, bool))}
    steps, stops = {}, {}
    for sgn in (+1, -1):
        cur, valid, z = seed.copy(), np.ones(n0, bool), Z0
        while True:
            zn = z + STEP_Z * sgn
            if zn < KEYS[0] or zn > KEYS[-1]:
                stops['up' if sgn > 0 else 'down'] = f'band end at {z}'
                break
            new, nv, acc, st = step(cur, valid, zn, m7, trk.dense(zn) if (zn - KEYS[0]) % KSTEP == 0 else None)
            steps[f'{z}>{zn}'] = st
            if new is None:
                stops['up' if sgn > 0 else 'down'] = f'refused {z}>{zn}: {st["refused"]}'
                break
            if nv.sum() < MIN_KEEP * n0:
                stops['up' if sgn > 0 else 'down'] = f'stopped {z}>{zn}: {int(nv.sum())} of {n0} samples left'
                break
            curves[zn] = (new, nv, acc)
            cur, valid, z = new, nv, zn
    return curves, steps, stops


def second_diff(curves, S, full=False):
    """{k: |S(k+16) - 2 S(k) + S(k-16)| per sample valid at all three h keys}: about 0 for a steadily inclined sheet,
    about one sheet spacing where the surface switched to a neighbouring sheet between keys."""
    kh = [k for k in sorted(curves) if (k - KEYS[0]) % KSTEP == 0]
    out = {}
    for k in kh[1:-1]:
        if k - KSTEP in curves and k + KSTEP in curves:
            v = curves[k - KSTEP][1] & curves[k][1] & curves[k + KSTEP][1]
            J = np.linalg.norm(S[k + KSTEP] - 2 * S[k] + S[k - KSTEP], axis=1)
            out[k] = np.where(v, J, np.nan) if full else J[v]
    return out


def trim_jumps(curves, S):
    """per column, where the second difference exceeds K2K_MAX, those samples (padded by JUMP_PAD) are dropped from that
    key outward (away from Z0); then each direction stops at the first key with fewer than MIN_KEEP of the seed samples
    left. The first difference is not used: a steadily inclined sheet exceeds the bar over long runs of keys."""
    ks = sorted(curves)
    n0 = len(curves[Z0][0])
    out = {k: (P, v.copy(), a) for k, (P, v, a) in curves.items()}
    events = []
    for k, J in sorted(second_diff(curves, S, full=True).items(), key=lambda kv: abs(kv[0] - Z0)):
        m = np.nan_to_num(J, nan=0.0) > K2K_MAX
        if not m.any():
            continue
        m = ndimage.binary_dilation(m, iterations=JUMP_PAD)
        for z in ks:
            if (k == Z0) or (k > Z0 and z >= k - KSTEP) or (k < Z0 and z <= k + KSTEP):
                out[z][1][m] = False
        events.append((k, int(m.sum())))
    keep = [Z0]
    for sg in (+1, -1):
        z = Z0 + sg * STEP_Z
        while z in out and out[z][1].sum() >= MIN_KEEP * n0:
            keep.append(z)
            z += sg * STEP_Z
    return {k: out[k] for k in sorted(keep)}, events


def smooth(curves):
    """5-sample moving average along u within valid runs, then [1, 2, 1] / 4 across neighbouring walk steps."""
    ks = sorted(curves)
    S = {}
    for z in ks:
        P, v, _ = curves[z]
        Q = P.copy()
        lab, n = ndimage.label(v)
        for r in range(1, n + 1):
            ii = np.nonzero(lab == r)[0]
            if len(ii) >= 5:
                Q[ii] = ndimage.uniform_filter1d(P[ii], 5, axis=0, mode='nearest')
        S[z] = Q
    out = {}
    for z in ks:
        Q = S[z].copy()
        if z - STEP_Z in S and z + STEP_Z in S:
            both = curves[z][1] & curves[z - STEP_Z][1] & curves[z + STEP_Z][1]
            Q[both] = (S[z - STEP_Z][both] + 2 * S[z][both] + S[z + STEP_Z][both]) / 4
        out[z] = Q
    return out


def at_z(curves, S, z):
    ks = sorted(curves)
    k0 = ks[0] + ((z - ks[0]) // STEP_Z) * STEP_Z
    k1 = min(k0 + STEP_Z, ks[-1])
    w = 0.0 if k1 == k0 else (z - k0) / float(STEP_Z)
    return (1 - w) * S[k0] + w * S[k1], curves[k0][1] & curves[k1][1]


def grid(curves, S):
    ks = sorted(curves)
    rows = np.arange(ks[0], ks[-1] + 1, ROW_EVERY)
    cols = np.arange(0, len(S[ks[0]]), COL_EVERY)
    X = np.full((len(rows), len(cols)), -1.0, np.float32)
    Y, Zg = X.copy(), X.copy()
    for r, z in enumerate(rows):
        P, v = at_z(curves, S, z)
        P, v = P[cols], v[cols]
        X[r, v], Y[r, v], Zg[r, v] = P[v, 1], P[v, 0], z
    return X, Y, Zg


def area_vx2(X, Y, Zg):
    P = np.stack([X, Y, Zg], -1).astype(np.float64)
    v = X != -1
    q = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
    a = 0.5 * np.linalg.norm(np.cross(P[1:, 1:] - P[:-1, :-1], P[1:, :-1] - P[:-1, 1:]), axis=-1)
    return float(a[q].sum())


def endpoints(trk, ids):
    pt, dr, tid, which = [], [], [], []
    for t in ids:
        p = trk.track(t)
        if len(p) < 2:
            continue
        k = min(4, len(p) - 1)
        for w, (e, f) in enumerate(((p[0], p[k]), (p[-1], p[-1 - k]))):
            v = f - e
            pt.append(e)
            dr.append(v / max(np.linalg.norm(v), 1e-9))
            tid.append(int(t))
            which.append(w)
    pt = np.array(pt)
    return dict(pt=pt, dir=np.array(dr), tid=np.array(tid), which=np.array(which), tree=cKDTree(pt))


def chain(trk, t0, E, used):
    poly, members, joins = trk.track(t0), [int(t0)], []
    own = np.full(len(poly), t0)
    for _ in (0, 1):
        while arc(poly)[-1] < TARGET and len(poly) >= 2:
            end, back = poly[-1], poly[max(0, len(poly) - 5)]
            o = end - back
            if np.linalg.norm(o) < 1e-6:
                break
            o = o / np.linalg.norm(o)
            near = [k for k in E['tree'].query_ball_point(end, r=END_GAP) if int(E['tid'][k]) not in members]
            if len(near) != 1:            # no end, or a junction: stop
                break
            k = near[0]
            t = int(E['tid'][k])
            g = E['pt'][k] - end
            ng = float(np.linalg.norm(g))
            if t in used or E['dir'][k] @ o < np.cos(np.radians(35)) or (ng > 2 and (g / ng) @ o < np.cos(np.radians(50))):
                break
            p = trk.track(t)
            p = p[::-1] if E['which'][k] == 1 else p
            poly, own = np.vstack([poly, p]), np.r_[own, np.full(len(p), t)]
            members.append(t)
            joins.append(round(ng, 1))
        poly, own = poly[::-1], own[::-1]
    return poly, own, members, joins


def load_axis(path):
    """control points {x, y, z} of an axis json -> (z, x, y) arrays sorted by z, as villa's umbilicus.json readers do."""
    cp = json.loads(Path(path).read_text())['control_points']
    uz = np.array([p['z'] for p in cp], float)
    ux = np.array([p['x'] for p in cp], float)
    uy = np.array([p['y'] for p in cp], float)
    o = np.argsort(uz)
    return uz[o], ux[o], uy[o]


def load_tifxyz(d):
    X, Y, Z = (tifffile.imread(Path(d) / f'{c}.tif').astype(np.float64) for c in 'xyz')
    return X, Y, Z, (X != -1.0) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)


def outward(X, Y, Z, valid, axis, sign=+1.0):
    """fraction of render normals (times sign) that point away from the axis (z, x, y)."""
    P = np.stack([X, Y, Z], -1)
    n, ok = rt.vertex_normals(P, valid)
    n, p = n[ok] * float(sign), P[ok]
    cx, cy = np.interp(p[:, 2], axis[0], axis[1]), np.interp(p[:, 2], axis[0], axis[2])
    r = np.stack([p[:, 0] - cx, p[:, 1] - cy], -1)
    r /= np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
    dot = n[:, 0] * r[:, 0] + n[:, 1] * r[:, 1]
    return dict(outward_fraction=float((dot > 0).mean()), dot_median=float(np.median(dot)), n_vertices=int(dot.size))


def orientation(d, axis):
    X, Y, Z, valid = load_tifxyz(d)
    plus, minus = outward(X, Y, Z, valid, axis, +1.0), outward(X, Y, Z, valid, axis, -1.0)
    sign = +1 if plus['outward_fraction'] > 0.5 else -1
    chosen = plus if sign > 0 else minus
    return dict(grid=f'{X.shape[0]}x{X.shape[1]}', valid_nodes=int(valid.sum()),
                outward_fraction_sign_plus1=round(plus['outward_fraction'], 4),
                dot_median_sign_plus1=round(plus['dot_median'], 4), chosen_sign=sign,
                outward_fraction_chosen=round(chosen['outward_fraction'], 4),
                confident=bool(chosen['outward_fraction'] >= MIN_CONFIDENT))


def level5_sections(volume):
    meta = json.loads(get(f'{volume}/5/.zarray'))
    (Zs, Ys, Xs), c = meta['shape'], meta['chunks'][0]
    assert meta['compressor'] is None and meta['dtype'] == '|u1', meta
    za, zb = KEYS[0] // L5, (KEYS[-1] + KSTEP) // L5 + 1
    ny, nx = -(-Ys // c), -(-Xs // c)
    planes = {}
    for zc in range(za // c, (zb - 1) // c + 1):
        blk = np.zeros((c, ny * c, nx * c), np.uint8)
        for yc in range(ny):
            for xc in range(nx):
                b = get(f'{volume}/5/{zc}/{yc}/{xc}')
                if b is not None:
                    blk[:, yc * c:(yc + 1) * c, xc * c:(xc + 1) * c] = np.frombuffer(b, np.uint8).reshape(c, c, c)
        for z5 in range(max(za, zc * c), min(zb, (zc + 1) * c)):
            planes[z5] = blk[z5 - zc * c, :Ys, :Xs].copy()
    out = {}
    for z5, p in planes.items():
        lab, n = ndimage.label(p > 0)
        if n:
            sz = np.bincount(lab.ravel())
            sz[0] = 0
            out[z5] = ndimage.binary_fill_holes(lab == int(np.argmax(sz)))
    return out


def cmd_axis(a):
    t0 = time.time()
    secs = level5_sections(a.volume)
    z5s = np.array(sorted(secs))
    zz, cx, cy = [], [], []
    for z5 in z5s:
        yy, xx = np.nonzero(secs[z5])
        zz.append(z5 * L5 + (L5 - 1) / 2)
        cx.append(xx.mean() * L5 + (L5 - 1) / 2)
        cy.append(yy.mean() * L5 + (L5 - 1) / 2)
    zz, cx, cy = map(np.array, (zz, cx, cy))
    savez_atomic(a.sections, z5=z5s, masks=np.stack([secs[k] for k in z5s]))
    trk = Tracks(a.tracks, kinds=('h',))
    pts = []
    for z in KEYS:
        P = np.concatenate([trk.track(t) for t in trk.ids('h', z)])
        V = P[ConvexHull(P[:, ::-1]).vertices][:, ::-1]        # x, y, counterclockwise
        x, y = V[:, 0], V[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        A = cr.sum() / 2
        hx, hy = ((x + x1) * cr).sum() / (6 * A), ((y + y1) * cr).sum() / (6 * A)
        ax, ay = float(np.interp(z, zz, cx)), float(np.interp(z, zz, cy))
        pts.append(dict(x=round(ax, 1), y=round(ay, 1), z=int(z), m7_hull_x=round(float(hx), 1),
                        m7_hull_y=round(float(hy), 1), hull_minus_ct=round(float(np.hypot(hx - ax, hy - ay)), 1)))
    dd = np.array([p['hull_minus_ct'] for p in pts])
    area = [float(secs[k].sum() * (L5 * UM * 1e-4) ** 2) for k in z5s]
    ax = dict(control_points=pts, scroll=SCROLL, status='COARSE AND UNVALIDATED. NOT AN UMBILICUS.',
              units='voxels of the level 0 volume (x = column, y = row, z = slice)',
              source='centroid of the cross-section: largest connected nonzero region of the masked CT at level 5 '
                     f'({L5}-voxel pixels), holes filled, one centroid per level-5 plane, linearly interpolated to every '
                     f'h key of the tracks band (z {KEYS[0]} to {KEYS[-1]}, step {KSTEP})',
              caution='Never tested against a reference axis; it can sit hundreds of voxels from the true one. Use it '
                      'only to choose the outward sign of local patches far from it.',
              check=dict(what='convex-hull centroid of all m7 h-track points at the same key',
                         distance_median=round(float(np.median(dd)), 1), distance_max=round(float(dd.max()), 1)),
              section_area_cm2=dict(min=round(min(area), 2), max=round(max(area), 2)),
              written_by='uncontested/sheet_patches.py axis', seconds=round(time.time() - t0, 1))
    write_atomic(a.axis, json.dumps(ax, indent=1))
    print(f'axis: {len(pts)} points, level-5 planes {len(z5s)}, x {min(cx):.0f}..{max(cx):.0f} y {min(cy):.0f}..'
          f'{max(cy):.0f}; hull-centroid distance median {np.median(dd):.1f} max {dd.max():.1f}; section area '
          f'{min(area):.2f}..{max(area):.2f} cm2; {time.time() - t0:.0f} s')


def seed_candidates(trk, axis, sections, z0):
    """seed chains at key z0 and their geometry; returns the chains, the candidates and the axis point at z0."""
    L5d = np.load(sections)
    m = L5d['masks'][int(np.argmin(np.abs(L5d['z5'] * L5 + (L5 - 1) / 2 - z0)))]
    ax0, ay0 = float(np.interp(z0, axis[0], axis[1])), float(np.interp(z0, axis[0], axis[2]))
    ey, ex = np.nonzero(m & ~ndimage.binary_erosion(m))
    ex, ey = ex * L5 + (L5 - 1) / 2, ey * L5 + (L5 - 1) / 2
    eth, er = np.arctan2(ey - ay0, ex - ax0), np.hypot(ex - ax0, ey - ay0)

    def outline(th):
        k = np.abs(np.angle(np.exp(1j * (eth - th)))) < np.radians(3)
        return float(er[k].max()) if k.any() else float('nan')

    ids = trk.ids('h', z0)
    L = {int(t): arc(trk.track(t))[-1] for t in ids}
    E = endpoints(trk, ids)
    used, chains = set(), []
    for t in sorted(L, key=lambda t: -L[t]):
        if t in used or L[t] < 400:
            continue
        poly, own, members, joins = chain(trk, t, E, used)
        used.update(members)
        if arc(poly)[-1] >= MIN_SEED:
            chains.append((poly, own, members, joins))
    keyd = trk.dense(z0)
    cands = []
    for ci, (poly, own, members, joins) in enumerate(chains):
        P, ow = resample(poly, own)
        n = int(MAX_SEED / DS)
        if len(P) > n:
            s0 = (len(P) - n) // 2
            P, ow = P[s0:s0 + n], ow[s0:s0 + n]
        T, N = frames(P)
        rel = P - np.array([ay0, ax0])
        r = np.linalg.norm(rel, axis=1)
        dot = (N * rel / r[:, None]).sum(1)
        mid = P[len(P) // 2]
        th = float(np.arctan2(mid[0] - ay0, mid[1] - ax0))
        R = outline(th)
        sp = spacing(P, T, N, ow, keyd)
        c = dict(i=ci, tracks=members, joins=joins, arc_vx=round((len(P) - 1) * DS, 1),
                 angle_deg=round(float(np.degrees(th)) % 360, 1), theta=th, r_median=round(float(np.median(r)), 1),
                 outline_r=round(R, 1), rfrac=round(float(np.median(r)) / R, 3),
                 sign_consistency=round(float(max((dot > 0).mean(), (dot < 0).mean())), 4),
                 abs_dot_median=round(float(np.median(np.abs(dot))), 3), spacing_median=round(float(np.median(sp)), 1),
                 spacing_ok_frac=round(float((sp >= SP_MIN).mean()), 3), seed=P)
        c['geom_ok'] = bool(RFRAC[0] <= c['rfrac'] <= RFRAC[1] and c['sign_consistency'] >= MIN_SGN and
                            c['abs_dot_median'] >= MIN_DOT and c['spacing_ok_frac'] >= MIN_SPOK)
        cands.append(c)
    return chains, cands, (ax0, ay0)


def trace(trk, c, m7, axis):
    """walk one candidate up and down the band, trim switches, grid it and orient its columns for outward sign +1."""
    curves, steps, stops = walk(trk, c['seed'], m7)
    S, cuts = smooth(curves), []
    for _ in range(8):
        kept, ev = trim_jumps(curves, S)
        cuts += [f'{k}:{n}' for k, n in ev]
        if not ev and len(kept) == len(curves):
            break
        curves = kept
        S = smooth(curves)
    c['jump_pairs_16'] = cuts
    X, Y, Zg = grid(curves, S)
    v = X != -1
    of = outward(X.astype(float), Y.astype(float), Zg.astype(float), v, axis, +1.0)['outward_fraction']
    c['columns_reversed'] = bool(of < 0.5)
    if c['columns_reversed']:
        X, Y, Zg = X[:, ::-1].copy(), Y[:, ::-1].copy(), Zg[:, ::-1].copy()
        of = outward(X.astype(float), Y.astype(float), Zg.astype(float), X != -1, axis, +1.0)['outward_fraction']
    ks = sorted(curves)
    c.update(keys=[ks[0], ks[-1]], n_keys=len(ks), stops=stops, outward_sign_plus1=round(of, 4),
             area_cm2=round(area_vx2(X, Y, Zg) * (UM * 1e-4) ** 2, 3), grid_shape=list(X.shape),
             valid_vertices=int((X != -1).sum()), _curves=curves, _S=S, _grid=(X, Y, Zg), _steps=steps)


def write_tifxyz(d, X, Y, Zg, name, info):
    """x, y, z float32 tifs plus meta.json, written into a .partial directory that then replaces d."""
    d = Path(d)
    tmp = d.with_name(d.name + '.partial')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for ch, A in zip('xyz', (X, Y, Zg)):
        tifffile.imwrite(tmp / f'{ch}.tif', A.astype(np.float32))
    v = X != -1
    meta = dict(uuid=f'{SCROLL}_{name}', name=f'{SCROLL}_{name}', type='seg', format='tifxyz', scale=[SCALE, SCALE],
                bbox=[[float(X[v].min()), float(Y[v].min()), float(Zg[v].min())],
                      [float(X[v].max()), float(Y[v].max()), float(Zg[v].max())]],
                area_vx2=round(area_vx2(X, Y, Zg), 1), patch=info)
    (tmp / 'meta.json').write_text(json.dumps(meta, indent=2))
    if d.exists():
        shutil.rmtree(d)
    os.replace(tmp, d)


def cmd_build(a):
    t0 = time.time()
    axis = load_axis(a.axis)
    trk = Tracks(a.tracks, kinds=('h',))       # the vertical tracks serve validation only
    chains, cands, (ax0, ay0) = seed_candidates(trk, axis, a.sections, Z0)
    m7 = M7(a.m7)
    print(f'{len(chains)} seed chains >= {MIN_SEED} vx at z {Z0}; geometry ok {sum(c["geom_ok"] for c in cands)}; '
          f'{time.time() - t0:.0f} s', flush=True)
    used_sep = None
    if a.candidate is not None:
        sel = [c for c in cands if c['i'] == a.candidate]
        if not sel:
            raise SystemExit(f'no candidate {a.candidate} at key {Z0} ({len(cands)} candidates)')
        trace(trk, sel[0], m7, axis)
        c = sel[0]
        c['name'] = a.name or f'c{c["i"]}_z{Z0}_a{int(round(c["angle_deg"])) % 360:03d}'
        if not (c['geom_ok'] and c['outward_sign_plus1'] >= MIN_OUT and c['area_cm2'] >= MIN_AREA):
            print(f'WARNING: candidate {c["i"]} fails a build bar: geometry {c["geom_ok"]}, outward '
                  f'{c["outward_sign_plus1"]} (bar {MIN_OUT}), area {c["area_cm2"]} cm2 (bar {MIN_AREA})', flush=True)
    else:
        walked = sorted([c for c in cands if c['geom_ok']], key=lambda c: -c['arc_vx'])[:MAX_WALK]
        for c in walked:
            trace(trk, c, m7, axis)
        ok = sorted([c for c in walked if c['outward_sign_plus1'] >= MIN_OUT and c['area_cm2'] >= MIN_AREA],
                    key=lambda c: -c['area_cm2'])
        for sep in (SEP_DEG, 20.0):
            sel = []
            for c in ok:
                if all(abs(np.degrees(np.angle(np.exp(1j * (c['theta'] - s['theta']))))) >= sep for s in sel):
                    sel.append(c)
                if len(sel) == MAX_PATCHES:
                    break
            used_sep = sep
            if len(sel) >= 3:
                break
        sel.sort(key=lambda c: c['angle_deg'])
        for k, c in enumerate(sel, a.first_index):
            c['name'] = f'pt{k}_z{Z0}_a{int(round(c["angle_deg"])) % 360:03d}'
    out, curves_dir = Path(a.out), Path(a.curves or Path(a.out) / 'curves')
    for c in sel:
        name, (X, Y, Zg), ks = c['name'], c['_grid'], sorted(c['_curves'])
        savez_atomic(curves_dir / f'{name}.npz', keys=np.array(ks),
                     pos=np.stack([c['_curves'][z][0] for z in ks]), valid=np.stack([c['_curves'][z][1] for z in ks]),
                     acc=np.stack([c['_curves'][z][2] for z in ks]), smooth=np.stack([c['_S'][z] for z in ks]),
                     columns_reversed=np.array(c['columns_reversed']))
        info = dict(method='local sheet patch from the m7 surface tracks, no umbilicus (uncontested/sheet_patches.py '
                           'build)', seed_key=Z0, candidate=c['i'], seed_tracks=c['tracks'], seed_joins_vx=c['joins'],
                    seed_arc_vx=c['arc_vx'], keys=c['keys'], walk_stops=c['stops'],
                    columns_reversed_for_outward_plus1=c['columns_reversed'],
                    axis=Path(a.axis).name + ' (COARSE, unvalidated)')
        write_tifxyz(out / f'{name}.tifxyz', X, Y, Zg, name, info)
        c['orientation'] = orientation(out / f'{name}.tifxyz', axis)
    keep = ('i', 'name', 'tracks', 'joins', 'arc_vx', 'angle_deg', 'r_median', 'outline_r', 'rfrac', 'sign_consistency',
            'abs_dot_median', 'spacing_median', 'spacing_ok_frac', 'geom_ok', 'keys', 'n_keys', 'stops',
            'outward_sign_plus1', 'columns_reversed', 'jump_pairs_16', 'area_cm2', 'grid_shape', 'valid_vertices',
            'orientation')
    rec = dict(scroll=SCROLL, params=dict(Z0=Z0, DS=DS, COL_EVERY=COL_EVERY, ROW_EVERY=ROW_EVERY, SCALE=SCALE, DMAX=DMAX,
                                          FRAC_SP=FRAC_SP, SP_MIN=SP_MIN, MIN_MATCH=MIN_MATCH, GAP=GAP, GAP_END=GAP_END,
                                          JUMP=JUMP, MIN_KEEP=MIN_KEEP, MED_WIN=MED_WIN, STEP_Z=STEP_Z, RAY=RAY,
                                          WMAX=WMAX, K2K_MAX=K2K_MAX, JUMP_PAD=JUMP_PAD, TARGET=TARGET,
                                          MAX_SEED=MAX_SEED, MIN_SEED=MIN_SEED, END_GAP=END_GAP, RFRAC=RFRAC,
                                          MIN_SGN=MIN_SGN, MIN_DOT=MIN_DOT, MIN_SPOK=MIN_SPOK, SEP_DEG_USED=used_sep,
                                          MIN_OUT=MIN_OUT, MIN_AREA=MIN_AREA),
               axis_at_Z0=[round(ax0, 1), round(ay0, 1)], n_chains=len(chains),
               candidates=[{k: c[k] for k in keep if k in c} for c in cands],
               steps={c['name']: c['_steps'] for c in sel}, selected=[c['name'] for c in sel],
               seconds=round(time.time() - t0, 1))
    write_atomic(a.record, json.dumps(rec, indent=1))
    for c in sel:
        o = c['orientation']
        print(f'{c["name"]}: area {c["area_cm2"]} cm2 keys {c["keys"]} grid {c["grid_shape"]} outward chosen '
              f'{o["outward_fraction_chosen"]} sign {o["chosen_sign"]} confident {o["confident"]}')
    print(f'build: {len(sel)} patches at key {Z0}' + (f' (separation {used_sep} deg)' if used_sep else '') +
          f', {time.time() - t0:.0f} s')


class M7:
    def __init__(self, path):
        import zarr
        self.a, self.cache, self.zrow = zarr.open_array(str(path), mode='r'), {}, None
        self.c = int(self.a.chunks[0])
        assert tuple(self.a.chunks) == (self.c,) * 3, self.a.chunks

    def plane_values(self, z, Q):
        """nearest-voxel m7 values (bool) at integer slice z for points Q (..., 2) y, x; keeps one chunk row cached."""
        c, zi = self.c, int(z)
        zc = zi // c
        if zc != self.zrow:
            self.cache.clear()
            self.zrow = zc
        Yi, Xi = np.rint(Q[..., 0]).astype(np.int64), np.rint(Q[..., 1]).astype(np.int64)
        out = np.zeros(Yi.shape, bool)
        ins = (Yi >= 0) & (Xi >= 0) & (Yi < self.a.shape[1]) & (Xi < self.a.shape[2])
        cy, cx = Yi // c, Xi // c
        for a, b in np.unique(np.stack([cy[ins], cx[ins]], -1), axis=0):
            msk = ins & (cy == a) & (cx == b)
            out[msk] = self.chunk((zc, int(a), int(b)))[zi - zc * c, Yi[msk] - a * c, Xi[msk] - b * c] > 0
        return out

    def chunk(self, k):
        if k not in self.cache:
            c = self.c
            self.cache[k] = np.asarray(self.a[k[0] * c:(k[0] + 1) * c, k[1] * c:(k[1] + 1) * c, k[2] * c:(k[2] + 1) * c])
        return self.cache[k]

    def near(self, P, r=2):
        """P (n, 3) z, y, x: True where a predicted voxel lies within Chebyshev distance r of the rounded point."""
        c, out = self.c, np.zeros(len(P), bool)
        for n, (z, y, x) in enumerate(np.rint(P).astype(np.int64)):
            lo, hi = np.array([z - r, y - r, x - r]), np.array([z + r + 1, y + r + 1, x + r + 1])
            hit = False
            for zc in range(lo[0] // c, (hi[0] - 1) // c + 1):
                for yc in range(lo[1] // c, (hi[1] - 1) // c + 1):
                    for xc in range(lo[2] // c, (hi[2] - 1) // c + 1):
                        o = np.array([zc, yc, xc]) * c
                        a, b = np.maximum(lo - o, 0), np.minimum(hi - o, c)
                        if self.chunk((zc, yc, xc))[a[0]:b[0], a[1]:b[1], a[2]:b[2]].any():
                            hit = True
            out[n] = hit
        return out


def dense_curve(P, v, up=4):
    parts = []
    lab, n = ndimage.label(v)
    for r in range(1, n + 1):
        ii = np.nonzero(lab == r)[0]
        t = np.linspace(0, len(ii) - 1, (len(ii) - 1) * up + 1)
        parts.append(np.stack([np.interp(t, np.arange(len(ii)), P[ii, 0]), np.interp(t, np.arange(len(ii)), P[ii, 1])], -1))
    return np.vstack(parts)


def key_to_key(curves, S):
    ks = sorted(curves)
    step_max = max(float(np.linalg.norm(S[b][curves[a][1] & curves[b][1]] - S[a][curves[a][1] & curves[b][1]], axis=1).max())
                   for a, b in zip(ks[:-1], ks[1:]))
    ks = [k for k in ks if (k - KEYS[0]) % KSTEP == 0]
    mx, md, same = [], [], []
    for a, b in zip(ks[:-1], ks[1:]):
        both = curves[a][1] & curves[b][1]
        d, _ = cKDTree(dense_curve(S[a], curves[a][1])).query(S[b][both])
        mx.append(float(d.max()))
        md.append(float(np.median(d)))
        same.append(float(np.linalg.norm(S[b][both] - S[a][both], axis=1).max()))
    w = int(np.argmax(mx))
    allc = np.concatenate([cKDTree(dense_curve(S[a], curves[a][1])).query(S[b][curves[a][1] & curves[b][1]])[0]
                           for a, b in zip(ks[:-1], ks[1:])])
    over = [int(m > K2K_MAX) for m in mx]
    run, longest = 0, 0
    for o in over:
        run = run + 1 if o else 0
        longest = max(longest, run)
    sd = second_diff(curves, S)
    sdv = np.concatenate([d for d in sd.values() if len(d)]) if sd else np.zeros(1)
    return dict(pairs=len(mx), closest_max=round(max(mx), 2), closest_p99=round(float(np.percentile(allc, 99)), 2),
                pairs_over_bar=int(sum(over)), longest_run_of_consecutive_pairs_over_bar=longest,
                second_diff_max=round(float(sdv.max()), 2), second_diff_p99=round(float(np.percentile(sdv, 99)), 2),
                frac_over_bar=round(float((allc > K2K_MAX).mean()), 5), closest_median=round(float(np.median(md)), 2),
                worst_pair=f'{ks[w]}>{ks[w + 1]}', same_u_displacement_max=round(max(same), 2),
                per_walk_step_max=round(step_max, 2))


def vertical_check(trk, curves, S, sp_med, shift=0.0):
    """crossings of the surface with the vertical (vx, vy) tracks; shift moves the surface along its in-plane normal by
    this many voxels (the off-sheet negative control)."""
    ks = sorted(curves)
    if shift:
        S = {k: S[k] + shift * frames(S[k])[1] for k in ks}
    rows = [(z,) + at_z(curves, S, z) for z in range(ks[0], ks[-1] + 1, 2)]
    out = {}
    for kind, coord in (('vx', 1), ('vy', 0)):
        lo = min(P[v, coord].min() for _, P, v in rows if v.any())
        hi = max(P[v, coord].max() for _, P, v in rows if v.any())
        vals = [x for x in np.unique(trk.k[kind]['K']) if lo <= x <= hi]
        D, share, n_lines, switches = [], [], 0, 0
        for X0 in vals:
            vk = trk.dense(X0, kind)
            cross = []
            for z, P, v in rows:
                s = P[:, coord] - X0
                tg = np.diff(P, axis=0)
                steep = np.abs(tg[:, coord]) >= 0.7 * np.maximum(np.linalg.norm(tg, axis=1), 1e-9)
                okc = v[:-1] & v[1:] & (s[:-1] * s[1:] <= 0) & (s[:-1] != s[1:]) & steep
                for i in np.nonzero(okc)[0]:
                    w = s[i] / (s[i] - s[i + 1])
                    cross.append((z, (1 - w) * P[i, 1 - coord] + w * P[i + 1, 1 - coord], i + w))
            if not cross or vk is None:
                continue
            C = np.array(cross)
            d, j = vk[2].query(C[:, :2])
            D.append(d)
            line = np.round(C[:, 2] / 25).astype(int)
            tids = vk[1][j]
            for l in np.unique(line):
                o = np.nonzero(line == l)[0]
                o = o[np.argsort(C[o, 0])]
                mo = o[d[o] <= 3.0]
                for e in range(1, len(mo)):            # a change of matched track while the old one goes on beside us
                    A, zc = tids[mo[e - 1]], C[mo[e], 0]
                    if tids[mo[e]] == A:
                        continue
                    Ap = vk[0][vk[1] == A]
                    if Ap[:, 0].max() >= zc + 16:
                        za = Ap[np.argmin(np.abs(Ap[:, 0] - (zc + 16)))]
                        pz = o[np.argmin(np.abs(C[o, 0] - (zc + 16)))]
                        if abs(C[pz, 0] - (zc + 16)) <= 4 and abs(za[1] - C[pz, 1]) >= 0.5 * sp_med:
                            switches += 1
                m = (line == l) & (d <= 3.0)
                if m.sum() >= 10:
                    cnt = np.bincount(np.unique(vk[1][j[m]], return_inverse=True)[1])
                    share.append((cnt.max() / cnt.sum(), int(m.sum())))
                    n_lines += 1
        if not D:
            out[kind] = dict(planes=len(vals), crossings=0)
            continue
        D = np.concatenate(D)
        cov = D <= sp_med
        out[kind] = dict(planes=len(vals), crossings=int(len(D)), median_distance=round(float(np.median(D)), 2),
                         within_3=round(float((D <= 3).mean()), 4), covered_within_spacing=round(float(cov.mean()), 4),
                         on_track_of_covered=round(float((D[cov] <= 3).mean()), 4) if cov.any() else None,
                         lines=n_lines, switch_events=switches, dominant_track_share=round(float(np.average([s for s, _ in share],
                                                                                     weights=[n for _, n in share])), 3)
                         if share else None)
    allD = [out[k] for k in out if out[k].get('crossings')]
    out['switch_events'] = sum(o.get('switch_events', 0) for o in allD)
    out['lines'] = sum(o.get('lines', 0) for o in allD)
    n = sum(o['crossings'] for o in allD)
    out['on_track_of_covered_all'] = round(sum((o['on_track_of_covered'] or 0) * o['crossings'] * o['covered_within_spacing']
                                               for o in allD) / max(sum(o['crossings'] * o['covered_within_spacing']
                                                                        for o in allD), 1e-9), 4) if n else None
    return out


def validate_one(d, curves, S, steps, cand, axis, trk, m7):
    t1 = time.time()
    ks = sorted(curves)
    kh = [k for k in ks if (k - KEYS[0]) % KSTEP == 0]
    X, Y, Z, v = load_tifxyz(d)
    on_v = m7.near(np.stack([Z[v], Y[v], X[v]], -1))
    kp = np.vstack([np.c_[np.full(int(curves[k][1].sum()), k), S[k][curves[k][1]]][::2] for k in kh])
    mids = []
    for a in kh:
        if a + KSTEP // 2 in curves:
            vv = curves[a + KSTEP // 2][1]
            mids.append(np.c_[np.full(int(vv.sum()), a + KSTEP // 2), S[a + KSTEP // 2][vv]][::2])
    mp = np.vstack(mids)
    on_k, on_m = m7.near(kp), m7.near(mp)
    m7.cache.clear()
    acc = np.concatenate([curves[k][2][curves[k][1]] for k in ks])
    sp_med = float(np.median([steps[s]['sp_median'] for s in steps if 'sp_median' in steps[s]]))
    vert = vertical_check(trk, curves, S, sp_med)
    ctrl = {f'{sg:+.1f}_spacing': vertical_check(trk, curves, S, sp_med, sg * sp_med)['on_track_of_covered_all']
            for sg in (-0.5, 0.5)}
    vert['control_half_spacing_shift_on_track'] = ctrl
    o = orientation(d, axis)
    meta = json.loads((Path(d) / 'meta.json').read_text())
    r = dict(area_cm2=round(meta['area_vx2'] * (UM * 1e-4) ** 2, 3), grid=list(X.shape), valid_vertices=int(v.sum()),
             keys=[ks[0], ks[-1]], height_vx=ks[-1] - ks[0], angle_deg=cand['angle_deg'], rfrac=cand['rfrac'],
             m7=dict(on_sheet_vertices=round(float(on_v.mean()), 4), n_vertices=int(len(on_v)),
                     on_sheet_key_rows=round(float(on_k.mean()), 4), n_key_samples=int(len(on_k)),
                     on_sheet_mid_key_rows=round(float(on_m.mean()), 4), n_mid_samples=int(len(on_m))),
             walk=dict(track_matched_fraction=round(float(acc.mean()), 4), local_spacing_median=round(sp_med, 1),
                       stops=cand['stops']),
             keys_check=key_to_key(curves, S), keys_check_raw=key_to_key(curves, {k: curves[k][0] for k in ks}),
             vertical=vert, orientation=o, seconds=round(time.time() - t1, 1))
    r['pass'] = bool(r['m7']['on_sheet_vertices'] >= CRIT['on_sheet_vertices_min'] and
                     r['keys_check']['second_diff_max'] <= CRIT['key_second_difference_max_max'] and
                     (vert['on_track_of_covered_all'] or 0) >= CRIT['vertical_on_track_vs_control'] *
                     max(max([x or 0 for x in ctrl.values()]), 1e-9) and o['confident'])
    return r


def cmd_validate(a):
    t0 = time.time()
    axis = load_axis(a.axis)
    b = json.loads(Path(a.record).read_text())
    trk, m7 = Tracks(a.tracks), M7(a.m7)
    cand = {c.get('name'): c for c in b['candidates'] if c.get('name')}
    curves_dir = Path(a.curves or Path(a.out) / 'curves')
    res = dict(criteria=CRIT, on_sheet_rule=ON_SHEET_RULE, axis=Path(a.axis).name + ' (COARSE, unvalidated)', patches={})
    for name in b['selected']:
        cz = np.load(curves_dir / f'{name}.npz')
        ks = [int(k) for k in cz['keys']]
        curves = {k: (cz['pos'][i], cz['valid'][i], cz['acc'][i]) for i, k in enumerate(ks)}
        S = {k: cz['smooth'][i] for i, k in enumerate(ks)}
        r = validate_one(Path(a.out) / f'{name}.tifxyz', curves, S, b['steps'][name], cand[name], axis, trk, m7)
        res['patches'][name] = r
        v, o = r['vertical'], r['orientation']
        print(f'{name}: area {r["area_cm2"]} on-sheet {r["m7"]["on_sheet_vertices"]} 2nd-diff max '
              f'{r["keys_check"]["second_diff_max"]} vertical {v["on_track_of_covered_all"]} control '
              f'{v["control_half_spacing_shift_on_track"]} switches {v["switch_events"]}/{v["lines"]} outward '
              f'{o["outward_fraction_chosen"]} pass {r["pass"]} ({r["seconds"]} s)', flush=True)
    vp = Path(a.validation)
    if vp.exists():
        for n, p in json.loads(vp.read_text()).get('patches', {}).items():
            if n in res['patches'] and 'render' in p:
                res['patches'][n]['render'] = p['render']
    res['n_pass'] = sum(p['pass'] for p in res['patches'].values())
    res['seconds'] = round(time.time() - t0, 1)
    write_atomic(vp, json.dumps(res, indent=1))
    print(f'validate: {res["n_pass"]} of {len(res["patches"])} pass, {time.time() - t0:.0f} s')


def cmd_renders(a):
    import cv2
    import zarr
    vp = Path(a.validation)
    res = json.loads(vp.read_text())
    for name, r in res['patches'].items():
        out = Path(a.renders) / f'{name}_66.zarr'
        js = out.with_name(out.name + '.render.json')
        if not js.exists():
            r['render'] = dict(missing=str(out))
            continue
        rec = json.loads(js.read_text())
        st = np.asarray(zarr.open_group(str(out), mode='r')['0'][:])
        n, H, W = st.shape
        vm = st.any(0)
        prof = [round(float(st[k][vm].mean()), 2) for k in range(n)]
        g = tifffile.imread(Path(a.out) / f'{name}.tifxyz' / 'x.tif') != -1
        expect = int((g[:-1, :-1] & g[1:, :-1] & g[:-1, 1:] & g[1:, 1:]).sum()) * int(round(1 / SCALE)) ** 2
        L = st[33].astype(np.float32)
        lo, hi = np.percentile(L[vm], [1, 99.5])
        img = np.clip((L - lo) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8)
        img[~vm] = 0
        w = min(800, W)
        h = max(1, int(round(H * w / W)))
        p = Path(a.renders) / 'previews' / f'{name}_L33.png'
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + '.partial.png')
        cv2.imwrite(str(tmp), cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
        os.replace(tmp, p)
        edges = float(np.mean(prof[:6] + prof[-6:]))
        r['render'] = dict(out=str(out), shape=[n, H, W], valid_px=rec['valid_px'],
                           valid_fraction_canvas=round(rec['valid_px'] / (H * W), 4),
                           valid_fraction_of_mesh_quads=round(rec['valid_px'] / expect, 4) if expect else None,
                           seconds=rec['seconds'], layer_mean=prof, peak_layer=int(np.argmax(prof)), peak_mean=max(prof),
                           edge_layers_mean=round(edges, 2), preview=str(p), preview_hw=[h, w])
        print(f'{name}: render {n}x{H}x{W}, valid canvas {r["render"]["valid_fraction_canvas"]}, peak layer '
              f'{r["render"]["peak_layer"]} ({max(prof)} vs edges {edges:.1f})')
    write_atomic(vp, json.dumps(res, indent=1))


def cmd_compare(paths):
    """exit 0 only when x.tif, y.tif and z.tif of two tifxyz directories are byte-identical."""
    if len(paths) != 2:
        raise SystemExit('compare takes two tifxyz directories')
    ok = True
    for ch in 'xyz':
        fa, fb = Path(paths[0]) / f'{ch}.tif', Path(paths[1]) / f'{ch}.tif'
        ba, bb = fa.read_bytes(), fb.read_bytes()
        same = ba == bb
        arr = np.array_equal(tifffile.imread(fa), tifffile.imread(fb))
        print(f'{ch}.tif bytes identical {same}, arrays identical {arr}, sha256 {hashlib.sha256(ba).hexdigest()[:16]} '
              f'{hashlib.sha256(bb).hexdigest()[:16]}')
        ok &= same
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], epilog=__doc__.split('\n\n', 2)[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('axis', 'build', 'validate', 'renders', 'compare'))
    ap.add_argument('paths', nargs='*', help='compare: two tifxyz directories')
    ap.add_argument('--tracks', default='tracks_export.npz', help='tracks npz from export_tracks.py')
    ap.add_argument('--m7', default='sp_m7.zarr/0', help='m7 surface prediction, level 0 zarr array')
    ap.add_argument('--axis', default='axis.json', help='coarse axis json (written by axis, read by build/validate)')
    ap.add_argument('--sections', default='level5_sections.npz', help='level 5 cross-sections (written by axis)')
    ap.add_argument('--out', default='patches', help='directory of the tifxyz patches')
    ap.add_argument('--curves', default=None, help='walked curves per patch (default <out>/curves)')
    ap.add_argument('--record', default='build.json', help='build record (written by build, read by validate)')
    ap.add_argument('--validation', default='validation.json', help='validation record')
    ap.add_argument('--renders', default='renders', help='render66 outputs <name>_66.zarr')
    ap.add_argument('--volume', default=VOLUME, help='masked CT OME-Zarr (axis reads level 5)')
    ap.add_argument('--scroll', default=SCROLL)
    ap.add_argument('--band', type=int, nargs=3, default=list(BAND), metavar=('FIRST', 'LAST', 'STEP'),
                    help='h keys of the tracks export')
    ap.add_argument('--key', type=int, default=Z0, help='seed key')
    ap.add_argument('--voxel-um', type=float, default=UM)
    ap.add_argument('--first-index', type=int, default=1, help='build: number of the first patch name (pt<N>_...)')
    ap.add_argument('--candidate', type=int, default=None, help='build: write only this candidate of --key')
    ap.add_argument('--name', default=None, help='build --candidate: patch name')
    a = ap.parse_args()
    configure(a)
    me = Path(__file__).resolve()
    print(f'sheet_patches.py {a.cmd} pid {os.getpid()} sha256 {hashlib.sha256(me.read_bytes()).hexdigest()[:16]} '
          f'mtime {time.ctime(me.stat().st_mtime)}', flush=True)
    if a.cmd == 'compare':
        sys.exit(0 if cmd_compare(a.paths) else 1)
    dict(axis=cmd_axis, build=cmd_build, validate=cmd_validate, renders=cmd_renders)[a.cmd](a)


if __name__ == '__main__':
    main()
