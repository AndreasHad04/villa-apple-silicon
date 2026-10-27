#!/usr/bin/env python3
"""export_tracks.py: the surface-tracks DBM of villa's extract_surface_tracks.py (legacy format: a pickled list of (n, 3) integer z, y, x arrays per key)
to one npz that the CPU patch builder reads without the spiral venv.

  export_tracks.py DBM OUT_NPZ

Per kind (h, vx, vy): <kind>_pts int16 (N, 3) z, y, x; <kind>_trk int32 (N,) track id; <kind>_key int32 (n_tracks,)
the key value of each track (z for h, x for vx, y for vy); track ids are unique across the kind.
"""
import dbm, os, pickle, sys
import numpy as np

src, out = sys.argv[1], sys.argv[2]
acc = {}
with dbm.open(src, 'r') as db:
    for k in db.keys():
        ks = k.decode() if isinstance(k, bytes) else k
        kind, val = ks.split(':')[0], ks.split(':', 1)[1]
        if kind not in ('h', 'vx', 'vy'):
            continue
        a = acc.setdefault(kind, dict(pts=[], trk=[], key=[]))
        for t in pickle.loads(db[k]):
            t = np.asarray(t)
            if t.ndim != 2 or t.shape[1] != 3 or len(t) == 0:
                continue
            a['trk'].append(np.full(len(t), len(a['key']), np.int32))
            a['pts'].append(t.astype(np.int16))
            a['key'].append(int(val))
res = {}
for kind, a in acc.items():
    res[f'{kind}_pts'] = np.concatenate(a['pts'])
    res[f'{kind}_trk'] = np.concatenate(a['trk'])
    res[f'{kind}_key'] = np.array(a['key'], np.int32)
tmp = out + '.partial.npz'
np.savez(tmp, **res)
os.replace(tmp, out)
for kind in sorted(acc):
    p, k = res[f'{kind}_pts'], res[f'{kind}_key']
    print(kind, 'points', len(p), 'tracks', len(k), 'keys', len(set(k.tolist())), 'key values', sorted(set(k.tolist()))[:4], '...', sorted(set(k.tolist()))[-2:])
