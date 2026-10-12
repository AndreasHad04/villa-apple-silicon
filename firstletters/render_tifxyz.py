#!/usr/bin/env python3
"""Render tifxyz segments into 31-layer surface volumes laid out like the official
PHerc0800 renders (``layers/00.tif`` .. ``layers/30.tif``, uint8). CPU only, numpy.

Conventions, read from villa's vc_render_tifxyz / QuadSurface / Slicing sources:

* canvas = lround(grid_shape / scale); output pixel (r, c) samples the stored grid at
  gx = sx * (c - (W - 1) / 2) + Ws / 2 (same for rows), i.e. (c + 0.5) / 20 for scale 0.05;
  positions are bilinear over the 4 surrounding stored vertices, all of which must be valid
  (a vertex is invalid when its x is -1 or any coordinate is not finite);
* normal = normalise(cross(P[r, c+1] - P[r, c-1], P[r+1, c] - P[r-1, c])), xyz order, taken
  at the nearest stored vertex ("nearest", VC's linear-mode default) or at the continuous
  grid position with +-1 bilinear tangents ("bilinear", villa's python self-render);
* layer k samples p + (k - 15 + layer_shift) * sign * n, trilinear in zyx with voxel centres
  on integers, 0 outside [0, shape); uint8 = truncation of the clamped float value.

The sign, layer shift, normal mode and in-plane orientation are NOT assumed. ``control``
measures them against the official render of a PHerc0800 segment and writes them to
results/render/control_0800.json; ``pherc1203`` refuses to run unless that file says PASS.
Volume chunks are fetched anonymously over HTTPS into a per-segment in-memory cache that is
dropped when the segment is written (no chunk ever touches the disk).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile
import urllib3

ROOT = Path(__file__).resolve().parents[2]
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
RESULTS = ROOT / "results" / "render"
LOG = RESULTS / "render.log"
NUM_LAYERS = 31
CENTER = 0.5 * (NUM_LAYERS - 1)  # vc_render_tifxyz buildOffsetList: 15.0
CHUNK = 128
CHUNK_BYTES = CHUNK ** 3  # 2,097,152: uint8, compressor null, so every stored chunk is full size
KB = 1 << 10  # chunk-index radix for int64 keys; every volume here has < 1024 chunks per axis
TILE = 256
CENTRAL = int(CENTER)
BAR_CENTRAL, BAR_MEDIAN = 0.90, 0.85

CONTROL_SEG = "20251029010146-auto_grown_20251029010146642"
CONTROL = dict(
    seg=CONTROL_SEG,
    tifxyz=f"{BUCKET}/PHerc0800/segments/{CONTROL_SEG}/mesh/20251029010146-on-20250521135224-8.64um.tifxyz",
    original_meta=f"{BUCKET}/PHerc0800/segments/{CONTROL_SEG}/mesh/intermediate/tifxyz_original/meta.json",
    volume=f"{BUCKET}/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr/0",
    render_zattrs=f"{BUCKET}/PHerc0800/segments/{CONTROL_SEG}/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/.zattrs",
    official=ROOT / "data" / "v8in_0800" / CONTROL_SEG / "layers",
    out=ROOT / "data" / "render_check_0800" / CONTROL_SEG,
)
CONTROL_JSON = RESULTS / "control_0800.json"
P1203_VOLUME = f"{BUCKET}/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr/0"
P1203_PREFIX = "PHerc1203/segments/raw/"
OUT1203 = ROOT / "data" / "v8in_1203"

HTTP = urllib3.PoolManager(num_pools=8, maxsize=64, retries=False,
                           timeout=urllib3.Timeout(connect=20, read=180))
_log_lock = threading.Lock()


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    with _log_lock:
        print(line, flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- HTTP

def http_get(url: str, expect_len: int | None = None, attempts: int = 10):
    """GET with retries. Returns (200, body) or (404, b""). Anything else is retried, then raised."""
    err = None
    for attempt in range(attempts):
        try:
            r = HTTP.request("GET", url, preload_content=True)
            if r.status == 404:
                return 404, b""
            if r.status == 200:
                if expect_len is None or len(r.data) == expect_len:
                    return 200, r.data
                err = f"body {len(r.data)} bytes, expected {expect_len}"
            else:
                err = f"HTTP {r.status}"
        except Exception as e:  # connection reset, timeout, DNS, ...
            err = repr(e)
        time.sleep(min(60.0, 0.5 * 2 ** attempt))
    raise RuntimeError(f"GET {url} failed after {attempts} attempts: {err}")


def s3_list(prefix: str):
    import re
    import urllib.parse
    prefixes, keys, token = [], [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
        if token:
            q["continuation-token"] = token
        st, body = http_get(BUCKET + "/?" + urllib.parse.urlencode(q))
        if st != 200:
            raise RuntimeError(f"list {prefix}: {st}")
        t = body.decode()
        prefixes += re.findall(r"<CommonPrefixes><Prefix>([^<]*)</Prefix></CommonPrefixes>", t)
        for c in re.findall(r"<Contents>(.*?)</Contents>", t, re.S):
            keys.append(re.search(r"<Key>([^<]*)</Key>", c).group(1))
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", t)
        if not m:
            return prefixes, keys
        token = m.group(1)


def fetch_small(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    st, body = http_get(url)
    if st != 200:
        raise RuntimeError(f"{url}: HTTP {st}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    tmp.write_bytes(body)
    os.replace(tmp, dst)
    return dst


# ----------------------------------------------------------------------------- volume

def key_of(cz, cy, cx):
    return (np.asarray(cz, np.int64) * KB + np.asarray(cy, np.int64)) * KB + np.asarray(cx, np.int64)


def unkey(k: int):
    k = int(k)
    return k // (KB * KB), (k // KB) % KB, k % KB


class ChunkCache:
    """Per-segment in-memory chunk store. None marks an absent (404, all-zero) chunk."""

    def __init__(self, volume_url: str, shape, threads: int):
        self.url = volume_url
        self.shape = tuple(int(s) for s in shape)
        self.threads = threads
        self.data: dict[int, np.ndarray | None] = {}
        self.n200 = self.n404 = self.bytes = self.refetch = 0
        self.ever: set[int] = set()
        self.fetch_seconds = 0.0

    def _one(self, k: int):
        cz, cy, cx = unkey(k)
        st, body = http_get(f"{self.url}/{cz}/{cy}/{cx}", expect_len=CHUNK_BYTES)
        if st == 404:
            return k, None, 0
        return k, np.frombuffer(body, np.uint8).reshape(CHUNK, CHUNK, CHUNK), len(body)

    def ensure(self, keys, label: str = "") -> None:
        missing = [int(k) for k in keys if int(k) not in self.data]
        if not missing:
            return
        t0 = time.time()
        last = t0
        done = 0
        with ThreadPoolExecutor(self.threads) as ex:
            futs = [ex.submit(self._one, k) for k in missing]
            for fut in as_completed(futs):
                k, arr, nb = fut.result()
                self.data[k] = arr
                if k in self.ever:
                    self.refetch += 1
                self.ever.add(k)
                if arr is None:
                    self.n404 += 1
                else:
                    self.n200 += 1
                    self.bytes += nb
                done += 1
                now = time.time()
                if now - last > 60 or done == len(missing):
                    rate = self.bytes / 1e6 / max(1e-9, self.fetch_seconds + now - t0)
                    log(f"  fetch {label} {done}/{len(missing)} chunks, total {self.bytes/1e9:.2f} GB, "
                        f"{rate:.1f} MB/s avg, 404s {self.n404}")
                    last = now
        self.fetch_seconds += time.time() - t0

    def drop(self, keep) -> None:
        keep = set(int(k) for k in keep)
        for k in [k for k in self.data if k not in keep]:
            del self.data[k]

    def resident_bytes(self) -> int:
        return sum(CHUNK_BYTES for v in self.data.values() if v is not None)


# ----------------------------------------------------------------------------- surface

def bilerp(field, valid, gy, gx):
    """Bilinear sample of an (Hs, Ws, 3) field at the outer product of gy and gx.
    Valid only where all 4 corners are valid (villa core.bilinear_field_tile)."""
    Hs, Ws = valid.shape
    y0 = np.clip(np.floor(gy).astype(np.int64), 0, Hs - 1)
    x0 = np.clip(np.floor(gx).astype(np.int64), 0, Ws - 1)
    y1 = np.minimum(y0 + 1, Hs - 1)
    x1 = np.minimum(x0 + 1, Ws - 1)
    wy = (gy - y0)[:, None, None]
    wx = (gx - x0)[None, :, None]
    ok = (valid[np.ix_(y0, x0)] & valid[np.ix_(y0, x1)] & valid[np.ix_(y1, x0)] & valid[np.ix_(y1, x1)])
    f00, f01 = field[np.ix_(y0, x0)], field[np.ix_(y0, x1)]
    f10, f11 = field[np.ix_(y1, x0)], field[np.ix_(y1, x1)]
    out = (f00 * (1 - wx) + f01 * wx) * (1 - wy) + (f10 * (1 - wx) + f11 * wx) * wy
    ok &= np.isfinite(out).all(axis=-1)
    return out, ok


def vertex_normals(P, valid):
    """VC grid_normal_int at every interior vertex; border vertices and any vertex with an
    invalid self or +-1 neighbour get no normal."""
    Hs, Ws = valid.shape
    n = np.full((Hs, Ws, 3), np.nan)
    ok = np.zeros((Hs, Ws), bool)
    if Hs < 3 or Ws < 3:
        return n, ok
    xv = P[1:-1, 2:] - P[1:-1, :-2]
    yv = P[2:, 1:-1] - P[:-2, 1:-1]
    c = np.cross(xv, yv)
    norm = np.linalg.norm(c, axis=-1)
    good = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1]
            & valid[:-2, 1:-1] & (norm > 1e-9))
    n[1:-1, 1:-1] = c / np.maximum(norm, 1e-12)[..., None]
    ok[1:-1, 1:-1] = good
    n[~ok] = np.nan
    return n, ok


class Surface:
    def __init__(self, X, Y, Z, scale_xy, normal_mode="nearest", px_shift=0.0):
        self.P = np.stack([X, Y, Z], axis=-1).astype(np.float64)
        self.Hs, self.Ws = X.shape
        self.valid = (X != -1.0) & np.isfinite(self.P).all(axis=-1)
        self.sx, self.sy = float(scale_xy[0]), float(scale_xy[1])
        # vc_render_tifxyz: full_size = lround(grid_size * render_scale / _scale), render_scale 1
        self.W = max(1, int(np.floor(self.Ws / self.sx + 0.5)))
        self.H = max(1, int(np.floor(self.Hs / self.sy + 0.5)))
        self.normal_mode = normal_mode
        self.px_shift = float(px_shift)
        if normal_mode == "nearest":
            self.vn, self.vn_ok = vertex_normals(self.P, self.valid)
        elif normal_mode != "bilinear":
            raise ValueError(normal_mode)

    def grid_coords(self, r0, r1, c0, c1):
        r = np.arange(r0, r1, dtype=np.float64) + self.px_shift
        c = np.arange(c0, c1, dtype=np.float64) + self.px_shift
        gy = self.sy * (r - 0.5 * (self.H - 1)) + 0.5 * self.Hs
        gx = self.sx * (c - 0.5 * (self.W - 1)) + 0.5 * self.Ws
        return gy, gx

    def normals(self, gy, gx):
        if self.normal_mode == "nearest":
            iy = np.clip(np.floor(gy + 0.5).astype(np.int64), 0, self.Hs - 1)
            ix = np.clip(np.floor(gx + 0.5).astype(np.int64), 0, self.Ws - 1)
            return self.vn[np.ix_(iy, ix)], self.vn_ok[np.ix_(iy, ix)]
        ny = np.clip(gy, 1.0, max(self.Hs - 3.0, 1.0))
        nx = np.clip(gx, 1.0, max(self.Ws - 3.0, 1.0))
        px, a = bilerp(self.P, self.valid, ny, nx + 1.0)
        mx, b = bilerp(self.P, self.valid, ny, nx - 1.0)
        py, c = bilerp(self.P, self.valid, ny + 1.0, nx)
        my, d = bilerp(self.P, self.valid, ny - 1.0, nx)
        xv, yv = px - mx, py - my
        xn, yn = np.linalg.norm(xv, axis=-1), np.linalg.norm(yv, axis=-1)
        ok = a & b & c & d & (xn > 1e-9) & (yn > 1e-9)
        n = np.cross(xv / np.maximum(xn, 1e-12)[..., None], yv / np.maximum(yn, 1e-12)[..., None])
        nn = np.linalg.norm(n, axis=-1)
        ok &= nn > 1e-9
        return n / np.maximum(nn, 1e-12)[..., None], ok

    def tile_points(self, r0, r1, c0, c1, sign, layer_shift):
        """Sample points of one canvas tile: (sel, q) with q (31, M, 3) in xyz voxel units."""
        gy, gx = self.grid_coords(r0, r1, c0, c1)
        xyz, ok = bilerp(self.P, self.valid, gy, gx)
        n, nok = self.normals(gy, gx)
        ok &= nok
        sel = np.flatnonzero(ok.ravel())
        p = xyz.reshape(-1, 3)[sel]
        d = n.reshape(-1, 3)[sel] * float(sign)
        offs = np.arange(NUM_LAYERS, dtype=np.float64) - CENTER + float(layer_shift)
        q = p[None, :, :] + offs[:, None, None] * d[None, :, :]
        return ok, sel, q


def tiles_of(H, W, tile=TILE):
    return [(r0, min(r0 + tile, H), c0, min(c0 + tile, W))
            for r0 in range(0, H, tile) for c0 in range(0, W, tile)]


def chunk_keys(q, shape):
    """Exact set of chunks holding any of the 8 trilinear corners of the in-bounds samples."""
    flat = q.reshape(-1, 3)
    x, y, z = flat[:, 0], flat[:, 1], flat[:, 2]
    Z, Y, X = shape
    inb = (z >= 0) & (y >= 0) & (x >= 0) & (z < Z) & (y < Y) & (x < X)
    if not inb.any():
        return np.zeros(0, np.int64)
    fz = np.floor(z[inb]).astype(np.int64)
    fy = np.floor(y[inb]).astype(np.int64)
    fx = np.floor(x[inb]).astype(np.int64)
    keys = np.unique(key_of(fz >> 7, fy >> 7, fx >> 7))
    edge = ((fz & 127) == 127) | ((fy & 127) == 127) | ((fx & 127) == 127)
    if edge.any():
        ez, ey, ex = fz[edge], fy[edge], fx[edge]
        extra = []
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    zz, yy, xx = ez + dz, ey + dy, ex + dx
                    ib = (zz < Z) & (yy < Y) & (xx < X)  # VC: sampleInt returns 0 past the edge
                    extra.append(key_of(zz[ib] >> 7, yy[ib] >> 7, xx[ib] >> 7))
        keys = np.union1d(keys, np.unique(np.concatenate(extra)))
    return keys


def sample_trilinear(q, cache: ChunkCache, keys):
    """VC sampleTrilinear on the cached chunks: float32 lerps x then y then z, 0 outside [0, shape)."""
    flat = q.reshape(-1, 3)
    x, y, z = flat[:, 0], flat[:, 1], flat[:, 2]
    Z, Y, X = cache.shape
    out = np.zeros(flat.shape[0], np.float32)
    inb = (z >= 0) & (y >= 0) & (x >= 0) & (z < Z) & (y < Y) & (x < X)
    if not inb.any():
        return out.reshape(q.shape[:-1])
    zi, yi, xi = z[inb], y[inb], x[inb]
    fz = np.floor(zi).astype(np.int64)
    fy = np.floor(yi).astype(np.int64)
    fx = np.floor(xi).astype(np.int64)
    z0, y0, x0 = int(fz.min()), int(fy.min()), int(fx.min())
    z1, y1, x1 = int(fz.max()) + 1, int(fy.max()) + 1, int(fx.max()) + 1  # inclusive, may equal shape
    D, Hb, Wb = z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1
    block = np.zeros((D, Hb, Wb), np.uint8)  # voxels past the volume edge stay 0, as VC's sampleInt
    for k in keys:
        arr = cache.data[int(k)]  # KeyError here means the chunk plan missed a chunk: fail loudly
        if arr is None:
            continue
        cz, cy, cx = unkey(k)
        zs, ze = max(cz * CHUNK, z0), min(cz * CHUNK + CHUNK, z1 + 1)
        ys, ye = max(cy * CHUNK, y0), min(cy * CHUNK + CHUNK, y1 + 1)
        xs, xe = max(cx * CHUNK, x0), min(cx * CHUNK + CHUNK, x1 + 1)
        if zs >= ze or ys >= ye or xs >= xe:
            continue
        block[zs - z0:ze - z0, ys - y0:ye - y0, xs - x0:xe - x0] = \
            arr[zs - cz * CHUNK:ze - cz * CHUNK, ys - cy * CHUNK:ye - cy * CHUNK, xs - cx * CHUNK:xe - cx * CHUNK]
    bf = block.ravel()
    lz, ly, lx = fz - z0, fy - y0, fx - x0
    wz = (zi - fz).astype(np.float32)
    wy = (yi - fy).astype(np.float32)
    wx = (xi - fx).astype(np.float32)
    i000 = (lz * Hb + ly) * Wb + lx
    sz, sy = Hb * Wb, Wb
    c000 = bf[i000].astype(np.float32)
    c001 = bf[i000 + 1].astype(np.float32)
    c010 = bf[i000 + sy].astype(np.float32)
    c011 = bf[i000 + sy + 1].astype(np.float32)
    c100 = bf[i000 + sz].astype(np.float32)
    c101 = bf[i000 + sz + 1].astype(np.float32)
    c110 = bf[i000 + sz + sy].astype(np.float32)
    c111 = bf[i000 + sz + sy + 1].astype(np.float32)
    c00 = c000 + wx * (c001 - c000)
    c01 = c010 + wx * (c011 - c010)
    c10 = c100 + wx * (c101 - c100)
    c11 = c110 + wx * (c111 - c110)
    c0 = c00 + wy * (c01 - c00)
    c1 = c10 + wy * (c11 - c10)
    out[inb] = c0 + wz * (c1 - c0)
    return out.reshape(q.shape[:-1])


def to_uint8(v):
    return np.clip(v, 0.0, 255.0).astype(np.uint8)  # VC: clamp then T(v), truncation for uint8


# ----------------------------------------------------------------------------- render

def render_surface(surf: Surface, cache: ChunkCache, sign: float, layer_shift: float,
                   workers: int, cap_bytes: float, label: str, keep_float: bool = False):
    """Render all 31 layers. Returns (stack uint8 (31,H,W), valid (H,W), info, float stack or None)."""
    H, W = surf.H, surf.W
    tiles = tiles_of(H, W)
    t0 = time.time()

    def plan(t):
        r0, r1, c0, c1 = t
        _, _, q = surf.tile_points(r0, r1, c0, c1, sign, layer_shift)
        return chunk_keys(q, cache.shape)

    with ThreadPoolExecutor(workers) as ex:
        tile_keys = list(ex.map(plan, tiles))
    union = np.unique(np.concatenate(tile_keys)) if tile_keys else np.zeros(0, np.int64)
    t_plan = time.time() - t0
    log(f"  plan {label}: canvas {H}x{W}, {len(tiles)} tiles, {len(union)} chunks needed "
        f"({len(union) * CHUNK_BYTES / 1e9:.2f} GB if all present), {t_plan:.1f} s")

    # Batches of consecutive tiles whose chunk union fits half the cap (the rest is prefetch room).
    cap_chunks = max(64, int(cap_bytes // CHUNK_BYTES))
    batches, cur, cur_keys = [], [], np.zeros(0, np.int64)
    for i, ks in enumerate(tile_keys):
        nxt = np.union1d(cur_keys, ks)
        if cur and len(nxt) > cap_chunks // 2:
            batches.append((cur, cur_keys))
            cur, nxt = [], ks
        cur.append(i)
        cur_keys = nxt
    if cur:
        batches.append((cur, cur_keys))

    stack = np.zeros((NUM_LAYERS, H, W), np.uint8)
    fstack = np.zeros((NUM_LAYERS, H, W), np.float32) if keep_float else None
    valid = np.zeros((H, W), bool)
    t_compute = 0.0

    def work(i):
        r0, r1, c0, c1 = tiles[i]
        ok, sel, q = surf.tile_points(r0, r1, c0, c1, sign, layer_shift)
        h, w = r1 - r0, c1 - c0
        valid[r0:r1, c0:c1] = ok
        if sel.size == 0:
            return
        v = sample_trilinear(q, cache, tile_keys[i])
        buf = np.zeros((NUM_LAYERS, h * w), np.uint8)
        buf[:, sel] = to_uint8(v)
        stack[:, r0:r1, c0:c1] = buf.reshape(NUM_LAYERS, h, w)
        if fstack is not None:
            fb = np.zeros((NUM_LAYERS, h * w), np.float32)
            fb[:, sel] = v
            fstack[:, r0:r1, c0:c1] = fb.reshape(NUM_LAYERS, h, w)

    for b, (idx, keys) in enumerate(batches):
        nxt_keys = batches[b + 1][1] if b + 1 < len(batches) else np.zeros(0, np.int64)
        cache.drop(np.union1d(keys, nxt_keys))
        cache.ensure(keys, label=f"{label} batch {b + 1}/{len(batches)}")
        tc = time.time()
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, idx))
        t_compute += time.time() - tc
    info = dict(tiles=len(tiles), batches=len(batches), chunks_needed=int(len(union)),
                chunks_fetched_200=cache.n200, chunks_absent_404=cache.n404, refetches=cache.refetch,
                bytes_fetched=cache.bytes, seconds_plan=round(t_plan, 2),
                seconds_fetch=round(cache.fetch_seconds, 2), seconds_compute=round(t_compute, 2),
                peak_cache_bytes_bound=int(min(len(union), cap_chunks) * CHUNK_BYTES))
    return stack, valid, info, fstack


# ----------------------------------------------------------------------------- transforms

TRANSFORMS = {
    "identity": lambda a: a,
    "flipud": lambda a: a[::-1, :],
    "fliplr": lambda a: a[:, ::-1],
    "rot180": lambda a: a[::-1, ::-1],
    "transpose": lambda a: a.T,                    # b[i, j] = a[j, i]
    "rot90ccw": lambda a: a.T[::-1, :],             # b[i, j] = a[j, n-1-i]  (np.rot90(a, 1))
    "rot90cw": lambda a: a.T[:, ::-1],              # b[i, j] = a[n-1-j, i]  (np.rot90(a, -1))
    "antitranspose": lambda a: a[::-1, ::-1].T,     # b[i, j] = a[n-1-j, n-1-i]
}


def pearson(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


# ----------------------------------------------------------------------------- io

def load_tifxyz(url_dir: str, dst: Path):
    for name in ("meta.json", "x.tif", "y.tif", "z.tif"):
        fetch_small(f"{url_dir}/{name}", dst / name)
    meta = json.loads((dst / "meta.json").read_text())
    X, Y, Z = (tifffile.imread(dst / f"{a}.tif") for a in "xyz")
    if not (X.shape == Y.shape == Z.shape and X.ndim == 2):
        raise ValueError(f"tifxyz shapes {X.shape} {Y.shape} {Z.shape}")
    shas = {f"{a}.tif": sha256_file(dst / f"{a}.tif") for a in "xyz"}
    return meta, X, Y, Z, shas


def volume_shape(volume_url: str):
    st, body = http_get(volume_url + "/.zarray")
    za = json.loads(body)
    assert za["chunks"] == [CHUNK] * 3 and za["dtype"] == "|u1" and za["compressor"] is None, za
    assert za.get("dimension_separator") == "/" and za["order"] == "C" and za["fill_value"] == 0, za
    return tuple(za["shape"]), za


def write_layers(stack, seg_dir: Path, transform: str):
    """Atomic: write layers.partial/00.tif..30.tif, then rename to layers/."""
    tmp = seg_dir / "layers.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shas, files = {}, []
    T = TRANSFORMS[transform]
    for i in range(NUM_LAYERS):
        p = tmp / f"{i:02d}.tif"
        tifffile.imwrite(p, np.ascontiguousarray(T(stack[i])), compression="zlib",
                         compressionargs={"level": 6})
        shas[p.name] = sha256_file(p)
        files.append(p.name)
    dst = seg_dir / "layers"
    if dst.exists():  # left by a run killed between rename and render.json; not a finished unit
        shutil.rmtree(dst)
    os.replace(tmp, dst)
    return dst, shas


def check_readback(layers_dir: Path, stack, transform: str):
    """Read back through the production v8-in loader and compare to what was rendered."""
    sys.path.insert(0, str(ROOT / "v8in"))
    from ink8um.inference import list_layer_files, read_stack  # production loader
    files = list_layer_files(layers_dir)
    assert [f.name for f in files] == [f"{i:02d}.tif" for i in range(NUM_LAYERS)], files
    got = read_stack(layers_dir)  # centred 24 of 31 -> layers 3..26, clipped to [0, 200]
    T = TRANSFORMS[transform]
    start = (NUM_LAYERS - 24) // 2
    want = np.stack([np.clip(T(stack[i]), 0, 200) for i in range(start, start + 24)], axis=2)
    assert got.shape == want.shape and np.array_equal(got, want), "read_stack read-back mismatch"
    return start


def disk_free_bytes() -> int:
    st = os.statvfs(str(ROOT))
    return st.f_bavail * st.f_frsize


def df_g_line() -> str:
    r = subprocess.run(["df", "-g", "/Users/andreashad04"], capture_output=True, text=True)
    return r.stdout.strip().splitlines()[-1] if r.returncode == 0 else f"df failed rc={r.returncode}"


def layer_stats(stack, valid):
    n = int(valid.sum())
    means = [float(stack[i][valid].mean()) if n else float("nan") for i in range(NUM_LAYERS)]
    zero_c = float((stack[CENTRAL][valid] == 0).mean()) if n else float("nan")
    return means, zero_c


# ----------------------------------------------------------------------------- control

def compare_to_official(ours, valid, official, label):
    """Full alignment table: transform x direction x layer offset, per-layer Pearson."""
    cover = official.max(axis=0) > 0
    table = []
    for tname, T in TRANSFORMS.items():
        tv = T(valid)
        if tv.shape != cover.shape:
            continue
        mask = cover & tv
        to = [np.ascontiguousarray(T(ours[j]))[mask] for j in range(NUM_LAYERS)]
        of = [official[i][mask] for i in range(NUM_LAYERS)]
        for direction in ("forward", "reversed"):
            for off in range(-3, 4):
                rs = []
                for i in range(NUM_LAYERS):
                    j = (i if direction == "forward" else NUM_LAYERS - 1 - i) + off
                    rs.append(pearson(of[i], to[j]) if 0 <= j < NUM_LAYERS else None)
                vals = [r for r in rs if r is not None]
                table.append(dict(transform=tname, direction=direction, layer_offset=off,
                                  central=rs[CENTRAL], median=float(np.median(vals)),
                                  n_layers=len(vals), per_layer=rs, mask_px=int(mask.sum())))
    best = max(table, key=lambda e: (e["median"], e["central"] if e["central"] is not None else -9))
    log(f"  {label}: best {best['transform']} {best['direction']} offset {best['layer_offset']}: "
        f"central {best['central']:.4f} median {best['median']:.4f} on {best['mask_px']} px")
    return table, best


def run_control(args):
    log("=== CONTROL: PHerc0800 " + CONTROL_SEG + f" (pid {os.getpid()}, renderer sha256 "
        f"{sha256_file(Path(__file__))[:16]})")
    seg_dir = CONTROL["out"]
    meta, X, Y, Z, tshas = load_tifxyz(CONTROL["tifxyz"], seg_dir / "tifxyz")
    orig_meta = json.loads(fetch_small(CONTROL["original_meta"], seg_dir / "tifxyz_original_meta.json").read_text())
    zattrs = json.loads(fetch_small(CONTROL["render_zattrs"], seg_dir / "official_render.zattrs.json").read_text())
    shape, _ = volume_shape(CONTROL["volume"])
    official = np.stack([tifffile.imread(p) for p in sorted(CONTROL["official"].glob("*.tif"),
                                                           key=lambda p: int(p.stem))])
    sys.path.insert(0, str(ROOT / "v8in"))
    from ink8um.inference import list_layer_files, _read_layer
    prod = np.stack([_read_layer(p) for p in list_layer_files(CONTROL["official"])])
    assert np.array_equal(prod, official), "official layers differ between loaders"
    assert official.shape[0] == NUM_LAYERS
    log(f"  tifxyz grid {X.shape}, scale {meta['scale']}, volume {shape}, official {official.shape}")

    cache = ChunkCache(CONTROL["volume"], shape, args.threads)
    variants = []
    for mode in ("nearest", "bilinear"):
        surf = Surface(X, Y, Z, meta["scale"], normal_mode=mode)
        stack, valid, info, fstack = render_surface(surf, cache, +1.0, 0.0, args.workers,
                                                    args.cap_gb * 1e9, f"control/{mode}", keep_float=True)
        table, best = compare_to_official(stack, valid, official, f"normals={mode}")
        variants.append(dict(mode=mode, surf=surf, stack=stack, valid=valid, info=info, fstack=fstack,
                             table=table, best=best))
    win = max(variants, key=lambda v: (v["best"]["median"], v["best"]["central"]))
    best = win["best"]
    T = TRANSFORMS[best["transform"]]
    sign = +1.0 if best["direction"] == "forward" else -1.0
    # in the official frame: official[i] == ours_aligned[i] with ours rendered at the found convention
    j_of = lambda i: (i if best["direction"] == "forward" else NUM_LAYERS - 1 - i) + best["layer_offset"]
    cover = official.max(axis=0) > 0
    tv = T(win["valid"])
    mask = cover & tv
    iou = float((cover & tv).sum() / max(1, (cover | tv).sum()))
    # exactness at the central layer and over all aligned layers
    eq, md, md_nz = [None] * NUM_LAYERS, [None] * NUM_LAYERS, [None] * NUM_LAYERS
    for i in range(NUM_LAYERS):
        j = j_of(i)
        if 0 <= j < NUM_LAYERS:
            a = official[i][mask].astype(np.int64)
            b = np.ascontiguousarray(T(win["stack"][j]))[mask].astype(np.int64)
            eq[i] = float((a == b).mean())
            md[i] = float((a - b).mean())
            nz = (a > 0) & (b > 0)
            md_nz[i] = pearson(a[nz], b[nz]) if nz.sum() > 100 else None
    # negative control: same alignment, ours shifted 40 px in-plane
    shifted = []
    for i in range(NUM_LAYERS):
        j = j_of(i)
        if 0 <= j < NUM_LAYERS:
            b = np.roll(np.ascontiguousarray(T(win["stack"][j])), (40, 40), axis=(0, 1))
            vb = np.roll(tv, (40, 40), axis=(0, 1))
            m2 = cover & vb & tv
            shifted.append(pearson(official[i][m2], b[m2]))
    # residual sub-pixel in-plane shift at the central layer (phase correlation on the masked image)
    a = np.where(mask, official[CENTRAL].astype(np.float64), 0.0)
    b = np.where(mask, np.ascontiguousarray(T(win["stack"][j_of(CENTRAL)])).astype(np.float64), 0.0)
    a -= a[mask].mean() * mask
    b -= b[mask].mean() * mask
    Fa, Fb = np.fft.fft2(a), np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    R /= np.maximum(np.abs(R), 1e-12)
    corr = np.fft.ifft2(R).real
    pk = np.unravel_index(np.argmax(corr), corr.shape)
    shift_px = [int(pk[0]) if pk[0] < corr.shape[0] // 2 else int(pk[0]) - corr.shape[0],
                int(pk[1]) if pk[1] < corr.shape[1] // 2 else int(pk[1]) - corr.shape[1]]
    # sampler self-test: trilinear vs scipy map_coordinates on the same cached voxels
    selftest = sampler_selftest(win["surf"], cache, sign)
    passed = bool(best["central"] is not None and best["central"] >= BAR_CENTRAL
                  and best["median"] >= BAR_MEDIAN)
    shape_checks = dict(canvas=[win["surf"].H, win["surf"].W], official=list(official.shape[1:]),
                        zattrs_canvas_size=zattrs.get("canvas_size"),
                        canvas_matches_official=[win["surf"].H, win["surf"].W] == list(official.shape[1:]),
                        valid_px=int(win["valid"].sum()), area_vx2_original=orig_meta.get("area_vx2"),
                        valid_px_over_area_vx2=float(win["valid"].sum() / orig_meta["area_vx2"]),
                        official_cover_px=int(cover.sum()), mask_iou_official_vs_ours=iou)
    stats_means, zero_c = layer_stats(win["stack"], win["valid"])
    out = dict(
        verdict="PASS" if passed else "FAIL", passed=passed,
        bars=dict(central_min=BAR_CENTRAL, median_min=BAR_MEDIAN),
        segment=CONTROL_SEG, volume=CONTROL["volume"], tifxyz=CONTROL["tifxyz"],
        official_layers=str(CONTROL["official"].relative_to(ROOT)),
        official_render_attrs={k: zattrs.get(k) for k in ("canvas_size", "num_slices", "slice_step",
                                                          "note_axes_order", "source_zarr")},
        best=dict(normal_mode=win["mode"], transform=best["transform"], direction=best["direction"],
                  layer_offset=best["layer_offset"], central=best["central"], median=best["median"],
                  mask_px=best["mask_px"], per_layer=best["per_layer"]),
        convention=dict(normal_mode=win["mode"], sign=sign,
                        layer_shift=float(best["layer_offset"] if sign > 0 else -best["layer_offset"]),
                        transform=best["transform"], px_shift=0.0,
                        meaning=("official layer i == our sample at p + (i - 15 + layer_shift) * sign * n, "
                                 "n = normalise(cross(P[r,c+1]-P[r,c-1], P[r+1,c]-P[r-1,c])) in xyz, "
                                 "then the in-plane transform applied to each layer")),
        exactness=dict(fraction_identical_per_layer=eq, mean_official_minus_ours_per_layer=md,
                       pearson_both_nonzero_per_layer=md_nz,
                       fraction_identical_central=eq[CENTRAL]),
        negative_control_shift40px=dict(per_layer=shifted, median=float(np.nanmedian(shifted))),
        residual_inplane_shift_px_central=shift_px,
        sampler_selftest=selftest,
        shape_checks=shape_checks,
        our_layer_mean_in_valid=stats_means, our_central_zero_fraction=zero_c,
        variants=[dict(normal_mode=v["mode"], best=dict((k, v["best"][k]) for k in
                                                        ("transform", "direction", "layer_offset",
                                                         "central", "median", "mask_px")),
                       table=v["table"]) for v in variants],
        fetch=win["info"] | dict(chunks_fetched_200=cache.n200, chunks_absent_404=cache.n404,
                                 bytes_fetched=cache.bytes),
        renderer_sha256=sha256_file(Path(__file__)), tifxyz_sha256=tshas,
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    # our control layers in the official frame, for inspection (not used by anything)
    if passed:
        aligned = np.stack([win["stack"][j_of(i)] if 0 <= j_of(i) < NUM_LAYERS
                            else np.zeros_like(win["stack"][0]) for i in range(NUM_LAYERS)])
        write_layers(aligned, seg_dir, best["transform"])
    write_json_atomic(CONTROL_JSON, out)
    log(f"  CONTROL {out['verdict']}: normals={win['mode']} transform={best['transform']} "
        f"direction={best['direction']} offset={best['layer_offset']} central r={best['central']:.4f} "
        f"median r={best['median']:.4f}; identical px central {out['exactness']['fraction_identical_central']}; "
        f"shift-40px negative median r={out['negative_control_shift40px']['median']:.4f}; "
        f"mask IoU {iou:.4f}; residual shift {shift_px}; selftest {selftest['max_abs_diff']:.2e}")
    return out


def sampler_selftest(surf: Surface, cache: ChunkCache, sign: float):
    """Independent path: rebuild a dense crop from the cached chunks and sample with scipy."""
    from scipy.ndimage import map_coordinates
    r0, c0 = surf.H // 2 - 32, surf.W // 2 - 32
    ok, sel, q = surf.tile_points(r0, r0 + 64, c0, c0 + 64, sign, 0.0)
    keys = chunk_keys(q, cache.shape)
    cache.ensure(keys, label="selftest")
    ours = sample_trilinear(q, cache, keys).ravel()
    flat = q.reshape(-1, 3)
    zyx = flat[:, ::-1]
    lo = np.floor(zyx.min(axis=0)).astype(int)
    hi = np.floor(zyx.max(axis=0)).astype(int) + 2
    crop = np.zeros(tuple(hi - lo), np.uint8)
    for cz in range(lo[0] // CHUNK, (hi[0] - 1) // CHUNK + 1):
        for cy in range(lo[1] // CHUNK, (hi[1] - 1) // CHUNK + 1):
            for cx in range(lo[2] // CHUNK, (hi[2] - 1) // CHUNK + 1):
                k = int(key_of(cz, cy, cx))
                if k not in cache.data:
                    cache.ensure([k], label="selftest")
                arr = cache.data[k]
                if arr is None:
                    continue
                a0 = np.array([cz, cy, cx]) * CHUNK
                s = np.maximum(a0, lo)
                e = np.minimum(a0 + CHUNK, hi)
                if (s >= e).any():
                    continue
                crop[tuple(slice(s[d] - lo[d], e[d] - lo[d]) for d in range(3))] = \
                    arr[tuple(slice(s[d] - a0[d], e[d] - a0[d]) for d in range(3))]
    ref = map_coordinates(crop.astype(np.float64), (zyx - lo).T, order=1, mode="constant", cval=0.0,
                          prefilter=False)
    diff = np.abs(ref - ours.astype(np.float64))
    # a deliberately broken path (x and z swapped) must disagree, or the test proves nothing
    bad = map_coordinates(crop.astype(np.float64), np.clip((flat - lo[::-1]), 0, None).T,
                          order=1, mode="constant", cval=0.0, prefilter=False)
    bad_diff = float(np.abs(bad - ref).mean())
    return dict(samples=int(diff.size), max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                broken_path_mean_abs_diff=bad_diff, passed=bool(diff.max() < 1e-3 and bad_diff > 1.0))


# ----------------------------------------------------------------------------- PHerc1203

def list_1203():
    prefixes, _ = s3_list(P1203_PREFIX)
    segs = []
    for pre in prefixes:
        seg = pre.rstrip("/").split("/")[-1]
        meta_path = fetch_small(f"{BUCKET}/{pre}meta.json", OUT1203 / seg / "tifxyz" / "meta.json")
        meta = json.loads(meta_path.read_text())
        segs.append(dict(seg=seg, url=f"{BUCKET}/{pre}".rstrip("/"), area_cm2=float(meta["area_cm2"]),
                         area_vx2=float(meta["area_vx2"])))
    segs.sort(key=lambda s: (s["area_cm2"], s["seg"]))
    return segs


def render_one_1203(s, conv, shape, args):
    seg_dir = OUT1203 / s["seg"]
    t0 = time.time()
    meta, X, Y, Z, tshas = load_tifxyz(s["url"], seg_dir / "tifxyz")
    surf = Surface(X, Y, Z, meta["scale"], normal_mode=conv["normal_mode"], px_shift=conv.get("px_shift", 0.0))
    H, W = surf.H, surf.W
    projected = NUM_LAYERS * H * W  # uncompressed upper bound of the layer files
    free = disk_free_bytes()
    log(f"--- {s['seg']} area {s['area_cm2']:.3f} cm2, grid {X.shape} -> canvas {H}x{W}; "
        f"df -g: {df_g_line()}; free {free/1e9:.1f} GB, layers <= {projected/1e9:.2f} GB")
    if free - projected < args.min_free_gb * 1e9:
        raise SystemExit(f"STOP: free {free/1e9:.1f} GB minus {projected/1e9:.2f} GB would fall under "
                         f"{args.min_free_gb} GB")
    cache = ChunkCache(P1203_VOLUME, shape, args.threads)
    stack, valid, info, _ = render_surface(surf, cache, conv["sign"], conv["layer_shift"],
                                           args.workers, args.cap_gb * 1e9, s["seg"][-9:])
    cache.data.clear()
    del cache
    gc.collect()
    means, zero_c = layer_stats(stack, valid)
    layers_dir, shas = write_layers(stack, seg_dir, conv["transform"])
    start = check_readback(layers_dir, stack, conv["transform"])
    out_shape = list(TRANSFORMS[conv["transform"]](stack[0]).shape)
    vf = float(valid.mean())
    rec = dict(
        segment=s["seg"], scroll="PHerc1203", volume=P1203_VOLUME, tifxyz=s["url"],
        grid_shape=list(X.shape), scale=meta["scale"], canvas_shape=out_shape, layers=NUM_LAYERS,
        valid_px=int(valid.sum()), valid_fraction=vf, area_cm2=s["area_cm2"], area_vx2=s["area_vx2"],
        valid_px_over_area_vx2=float(valid.sum() / s["area_vx2"]),
        layer_mean_in_valid=means, central_layer=CENTRAL, central_mean=means[CENTRAL],
        central_zero_fraction_in_valid=zero_c, flag_central_mostly_zero=bool(zero_c > 0.5),
        brightest_layer=int(np.nanargmax(means)), read_stack_layers=[start, start + 23],
        convention=conv, layer_sha256=shas, layers_bytes=int(sum(p.stat().st_size for p in layers_dir.iterdir())),
        seconds=round(time.time() - t0, 1), **info,
        renderer_sha256=sha256_file(Path(__file__)), tifxyz_sha256=tshas,
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    write_json_atomic(seg_dir / "render.json", rec)
    log(f"    done {s['seg']}: canvas {out_shape}, valid {vf:.3f}, central mean {means[CENTRAL]:.1f} "
        f"(brightest layer {rec['brightest_layer']}), central zero {zero_c:.3f}, "
        f"{info['chunks_fetched_200']} chunks + {info['chunks_absent_404']} absent, "
        f"{info['bytes_fetched']/1e9:.2f} GB fetched, layers {rec['layers_bytes']/1e9:.2f} GB, {rec['seconds']} s")
    return rec


def run_1203(args):
    if not CONTROL_JSON.exists():
        raise SystemExit("no results/render/control_0800.json: run the control first")
    ctl = json.loads(CONTROL_JSON.read_text())
    if not ctl.get("passed"):
        raise SystemExit("control did not PASS; refusing to render PHerc1203")
    conv = ctl["convention"]
    log(f"=== PHerc1203 batch (pid {os.getpid()}, renderer sha256 {sha256_file(Path(__file__))[:16]}), "
        f"convention {json.dumps({k: conv[k] for k in ('normal_mode', 'sign', 'layer_shift', 'transform')})}")
    shape, _ = volume_shape(P1203_VOLUME)
    segs = list_1203()
    if args.only:
        segs = [s for s in segs if s["seg"] in set(args.only)]
    if args.limit:
        segs = segs[:args.limit]
    log(f"  {len(segs)} segments, volume {shape}, order smallest area first")
    for s in segs:
        if (OUT1203 / s["seg"] / "render.json").exists():
            log(f"  skip {s['seg']}: render.json exists")
            continue
        render_one_1203(s, conv, shape, args)
    summarize()


def summarize():
    rows = []
    for p in sorted(OUT1203.glob("*/render.json")):
        r = json.loads(p.read_text())
        rows.append({k: r[k] for k in ("segment", "area_cm2", "canvas_shape", "valid_fraction", "valid_px_over_area_vx2",
                                       "central_mean", "brightest_layer", "central_zero_fraction_in_valid",
                                       "flag_central_mostly_zero", "seconds", "bytes_fetched", "layers_bytes",
                                       "chunks_fetched_200", "chunks_absent_404", "refetches")})
    rows.sort(key=lambda r: (r["area_cm2"], r["segment"]))
    tot = dict(segments=len(rows), bytes_fetched=sum(r["bytes_fetched"] for r in rows),
               layers_bytes=sum(r["layers_bytes"] for r in rows), seconds=sum(r["seconds"] for r in rows),
               flagged=[r["segment"] for r in rows if r["flag_central_mostly_zero"]])
    write_json_atomic(RESULTS / "pherc1203_summary.json", dict(totals=tot, rows=rows))
    print(f"{'segment':30s} {'cm2':>6s} {'canvas':>12s} {'valid':>6s} {'v/area':>6s} {'c.mean':>6s} "
          f"{'peak':>4s} {'c.zero':>6s} {'s':>6s} {'GBfetch':>7s} {'GBlay':>6s}")
    for r in rows:
        print(f"{r['segment']:30s} {r['area_cm2']:6.3f} {str(r['canvas_shape'][0])+'x'+str(r['canvas_shape'][1]):>12s} "
              f"{r['valid_fraction']:6.3f} {r['valid_px_over_area_vx2']:6.3f} {r['central_mean']:6.1f} "
              f"{r['brightest_layer']:4d} {r['central_zero_fraction_in_valid']:6.3f} {r['seconds']:6.0f} "
              f"{r['bytes_fetched']/1e9:7.2f} {r['layers_bytes']/1e9:6.2f}")
    print(json.dumps(tot))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["control", "pherc1203", "summary"])
    ap.add_argument("--threads", type=int, default=24, help="parallel chunk downloads")
    ap.add_argument("--workers", type=int, default=6, help="CPU threads for geometry and sampling")
    ap.add_argument("--cap-gb", type=float, default=6.0, help="in-memory chunk cache cap (GB)")
    ap.add_argument("--min-free-gb", type=float, default=205.0,
                    help="stop before a segment whose layers could push free disk under this (1e9-byte GB)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.command == "control":
        run_control(args)
    elif args.command == "pherc1203":
        run_1203(args)
    else:
        summarize()


if __name__ == "__main__":
    main()
