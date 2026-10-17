#!/usr/bin/env python3
"""render66.py: render a tifxyz mesh into a 66-layer surface volume the way Nieuwlaar's PHerc. 343 First Letters
pipeline does with VC3D 41701aa (his mesh/render_66_layers.sh), using a CPU-only Python renderer instead of VC3D.
No VC3D build, no GPU, any OS with Python 3.10+, numpy, tifffile, urllib3, zarr (3.x) and numcodecs.

Convention (his render_66_layers.sh and image sidecar "orientation"):
  layer k (0..65) samples p + (k - 32.5) * n, n = normalise(dP/dcolumn x dP/drow) at the nearest stored vertex
  (VC's default normal), no flipped normals; depth d = k - 33; in-plane orientation = the VC canvas (identity).
  Positions are bilinear over the 4 surrounding stored vertices (all must be valid), the CT is sampled trilinearly
  with voxel centres on integers and 0 outside the volume, and uint8 is the truncation of the clamped value.
  The sampling itself is render_tifxyz.py (next to this file), read from villa's vc_render_tifxyz sources.

Volume: an uncompressed uint8 OME-Zarr level with 128^3 chunks on the open-data bucket, read anonymously over HTTPS
(only the chunks the mesh needs). With --cache-dir each chunk is also kept on disk (atomic rename), so an
interrupted render resumes without refetching; without it nothing touches the disk but the output.

Output: <out> (zarr v2 group, array "0", 66 x H x W uint8, chunks 66 x 128 x 128, blosc lz4), written to
<out>.partial, read back and compared, then renamed; plus <out>.render.json and <out>.render.log.

PHerc. 343, the First Letters segment (the mesh comes from his mesh/build_concat.py, or his outputs/ folder):
  python render66.py --mesh outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz \\
      --volume https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr/0 \\
      --out concat_w047-w048_R5B2_z9500-11000_66.zarr --cache-dir chunks_0343
That fetches 1,053 chunks (2.21 GB) once; the sampling itself took about 10 s on an M1 Max.

Other grids, with vc_render_tifxyz's own option names: --num-slices N renders N layers centred at (N - 1) / 2
(VC's buildOffsetList, so 66 gives the half-integer offsets above and 31 gives integer offsets -15..15), and
--flip-normals samples along -n. VC3D's default 31-slice render with --flip-normals is therefore
  python render66.py --mesh <seg>.tifxyz --volume <volume level url> --out <seg>_31.zarr --num-slices 31 --flip-normals
The defaults (66 layers, VC's normal) are unchanged.
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_tifxyz as rt  # noqa: E402

N_LAYERS, CENTRE = 66, 32.5
rt.NUM_LAYERS = N_LAYERS        # module globals read at call time by Surface.tile_points and render_surface
rt.CENTER = CENTRE


class DiskChunkCache(rt.ChunkCache):
    """rt.ChunkCache plus a resumable disk store: <dir>/cz/cy/cx (raw 2 MiB) or <dir>/cz/cy/cx.404."""

    def __init__(self, volume_url, shape, threads, cache_dir: Path):
        super().__init__(volume_url, shape, threads)
        self.dir = cache_dir

    def _one(self, k):
        cz, cy, cx = rt.unkey(k)
        p = self.dir / str(cz) / str(cy) / str(cx)
        if p.with_name(p.name + '.404').exists():
            return k, None, 0
        if p.exists() and p.stat().st_size == rt.CHUNK_BYTES:
            return k, np.fromfile(p, np.uint8).reshape(rt.CHUNK, rt.CHUNK, rt.CHUNK), 0
        st, body = rt.http_get(f'{self.url}/{cz}/{cy}/{cx}', expect_len=rt.CHUNK_BYTES)
        p.parent.mkdir(parents=True, exist_ok=True)
        if st == 404:
            m = p.with_name(p.name + '.404')
            tmp = m.with_name(m.name + '.partial')
            tmp.write_bytes(b'')
            os.replace(tmp, m)
            return k, None, 0
        tmp = p.with_name(p.name + '.partial')
        tmp.write_bytes(body)
        os.replace(tmp, p)
        return k, np.frombuffer(body, np.uint8).reshape(rt.CHUNK, rt.CHUNK, rt.CHUNK), len(body)


def write_zarr(stack, out: Path, attrs):
    import zarr, numcodecs
    tmp = out.with_name(out.name + '.partial')
    if tmp.exists():
        shutil.rmtree(tmp)
    g = zarr.open_group(str(tmp), mode='w', zarr_format=2)
    n, H, W = stack.shape
    arr = g.create_array('0', shape=(n, H, W), chunks=(n, 128, 128), dtype='uint8', fill_value=0,
                         compressors=numcodecs.Blosc(cname='lz4', clevel=3, shuffle=1),
                         chunk_key_encoding={'name': 'v2', 'separator': '/'})
    for r0 in range(0, H, 128):
        arr[:, r0:r0 + 128, :] = stack[:, r0:r0 + 128, :]
    g.attrs.update(attrs)
    back = np.asarray(zarr.open_group(str(tmp), mode='r')['0'][:])   # read-back check before the rename
    if not np.array_equal(back, stack):
        raise SystemExit('read-back mismatch, output not published')
    if out.exists():
        shutil.rmtree(out)
    os.rename(tmp, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mesh', required=True, help='tifxyz folder (x.tif, y.tif, z.tif, meta.json)')
    ap.add_argument('--volume', required=True, help='URL of one OME-Zarr level, e.g. .../<volume>.zarr/0')
    ap.add_argument('--out', required=True, help='output .zarr path')
    ap.add_argument('--cache-dir', help='keep fetched chunks here (resumable); default: memory only')
    ap.add_argument('--threads', type=int, default=32, help='parallel chunk downloads')
    ap.add_argument('--workers', type=int, default=8, help='CPU threads for geometry and sampling')
    ap.add_argument('--cap-gb', type=float, default=8.0, help='in-memory chunk cache cap (GB)')
    ap.add_argument('--plan-only', action='store_true', help='print the chunk count and stop')
    ap.add_argument('--num-slices', type=int, default=N_LAYERS,
                    help='layers, centred at (N - 1) / 2 as vc_render_tifxyz does (default 66)')
    ap.add_argument('--flip-normals', action='store_true', help='sample along -n, as vc_render_tifxyz --flip-normals')
    a = ap.parse_args()
    if a.num_slices < 1:
        ap.error('--num-slices must be at least 1')
    n_layers, centre, sign = a.num_slices, 0.5 * (a.num_slices - 1), (-1.0 if a.flip_normals else +1.0)
    rt.NUM_LAYERS, rt.CENTER = n_layers, centre
    default_grid = (n_layers, centre, sign) == (N_LAYERS, CENTRE, +1.0)
    mesh, out = Path(a.mesh), Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rt.RESULTS = out.parent
    rt.LOG = out.with_name(out.name + '.render.log')
    t0 = time.time()
    me = Path(__file__).resolve()
    rt.log(f'=== render66 pid {os.getpid()} src sha256 {rt.sha256_file(me)}; '
           f'renderer sha256 {rt.sha256_file(Path(rt.__file__))}')
    meta = json.loads((mesh / 'meta.json').read_text())
    X, Y, Z = (tifffile.imread(mesh / f'{c}.tif') for c in 'xyz')
    sc = meta['scale']
    sc = (float(sc[0]), float(sc[1])) if isinstance(sc, (list, tuple)) else (float(sc), float(sc))
    shas = {f'{c}.tif': rt.sha256_file(mesh / f'{c}.tif') for c in 'xyz'}
    shape, _ = rt.volume_shape(a.volume)
    surf = rt.Surface(X, Y, Z, sc, normal_mode='nearest', px_shift=0.0)
    rt.log(f'  mesh {mesh.name} grid {X.shape} scale {sc} -> canvas {surf.H} x {surf.W}; volume {shape}')
    if a.plan_only:
        keys = set()
        for t in rt.tiles_of(surf.H, surf.W):
            _, _, q = surf.tile_points(*t, sign, 0.0)
            keys.update(int(k) for k in rt.chunk_keys(q, shape))
        rt.log(f'  plan: {len(keys)} chunks ({len(keys) * rt.CHUNK_BYTES / 1e9:.2f} GB raw)')
        return
    if a.cache_dir:
        cache = DiskChunkCache(a.volume, shape, a.threads, Path(a.cache_dir))
    else:
        cache = rt.ChunkCache(a.volume, shape, a.threads)
    stack, valid, info, _ = rt.render_surface(surf, cache, sign, 0.0, a.workers, a.cap_gb * 1e9, mesh.name)
    assert stack.shape[0] == n_layers, stack.shape
    for k in range(n_layers):          # outside the valid mask every layer must be 0 (VC writes nothing there)
        assert not stack[k][~valid].any()
    rule = ('layer k samples p + (k - 32.5) * n; d = k - 33' if default_grid else
            f'layer k samples p + (k - {centre:g}) * {"-" if sign < 0 else ""}n')
    attrs = dict(render66=dict(convention=dict(layers=n_layers, centre=centre, sign=sign, normal_mode='nearest',
                                               normal='normalise(dP/dcolumn x dP/drow)', transform='identity',
                                               rule=rule),
                               volume=a.volume, mesh=str(mesh), mesh_sha256=shas, scale=sc,
                               renderer_sha256=rt.sha256_file(Path(rt.__file__)), script_sha256=rt.sha256_file(me),
                               info=info, valid_px=int(valid.sum())))
    write_zarr(stack, out, attrs)
    rec = dict(attrs['render66'], out=str(out), seconds=round(time.time() - t0, 1), shape=list(stack.shape))
    tmp = out.with_name(out.name + '.render.json.partial')
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, out.with_name(out.name + '.render.json'))
    rt.log(f'  written {out} {stack.shape}, valid {int(valid.sum())} px, {time.time() - t0:.0f} s; info {json.dumps(info)}')


if __name__ == '__main__':
    main()
