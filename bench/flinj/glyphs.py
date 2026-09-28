"""ARM FLINJ phase 1, Injection 2: Greek capitals rasterised from a system font, skeletonised, re-drawn at a
fixed physical stroke width and cap height, and the geometry masks the metric uses.

Registered (PREREGISTRATION_FLINJ.md): cap height 2.5 mm, stroke 0.35 mm, rows of 7 letters drawn at random
from the 24, letter pitch 1.2 cap heights, row pitch 1.8 cap heights, rotation uniform in +/- 5 degrees,
64 px clear of crop borders, soft edges Gaussian sigma 0.7 px; strokes = glyph pixels, counters = convex hull
minus the glyph dilated by half a stroke width, far background = farther than 2 cap heights from every glyph.

NOT registered, fixed here before any injected input existed (each is listed in the FLINJ report):
  D1 font: Arial (sans serif, all 24 capitals present); a serif face skeletonises into serif spurs.
  D2 cap height: the OUTER height of the redrawn Eta (skeleton height x scale + stroke width), for every width.
  D3 rotation: per letter, about the letter's own centre (skeleton bounding-box centre).
  D4 layout: superseded by amendment 1, A1 (see plan_layout).
  D5 counters with several glyphs: union of the glyph hulls minus the union of glyphs dilated by half a stroke.
  D6 every mask is restricted to pixels where the surface volume holds data.
"""
import functools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from skimage.morphology import convex_hull_image, skeletonize
from common import REG

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
LETTERS = [chr(c) for c in range(0x391, 0x3AA) if c != 0x3A2]  # the 24 Greek capitals
assert len(LETTERS) == 24
RENDER_CAP = 600  # px: cap height of the source rendering the skeletons are taken from


def _raster(letter, size):
    f = ImageFont.truetype(FONT, size); n = int(size * 1.6)
    im = Image.new("L", (n, n), 0); ImageDraw.Draw(im).text((n // 2, n // 2), letter, font=f, fill=255, anchor="mm")
    return np.asarray(im) > 127


@functools.lru_cache(maxsize=None)
def _font_size():
    m = _raster("Η", 1000); ys = np.nonzero(m.any(1))[0]
    return int(round(1000 * RENDER_CAP / (ys.max() - ys.min() + 1)))


@functools.lru_cache(maxsize=None)
def skeleton(letter):
    """Skeleton points (N, 2) as (y, x) in source px, centred on the skeleton's bounding-box centre."""
    pts = np.argwhere(skeletonize(_raster(letter, _font_size()))).astype(np.float64)
    return pts - (pts.min(0) + pts.max(0)) / 2.0


@functools.lru_cache(maxsize=None)
def ref_height():
    p = skeleton("Η"); return float(p[:, 0].max() - p[:, 0].min())


def _placed(letter, cy, cx, ang, scale):
    p = skeleton(letter) * scale; t = np.deg2rad(ang); c, s = np.cos(t), np.sin(t)
    return np.stack([p[:, 0] * c - p[:, 1] * s + cy, p[:, 0] * s + p[:, 1] * c + cx], 1)


def draw(shape, glyphs, cap, stroke):
    """Label image (int16): 0 background, k+1 on glyph k. glyphs = [(letter, cy, cx, angle_deg)].
    A pixel is on the glyph when its centre is within stroke/2 of the scaled, rotated skeleton (exact to the
    skeleton point spacing, which is under 0.5 px at every registered size)."""
    scale, r = (cap - stroke) / ref_height(), stroke / 2.0
    lab = np.zeros(shape, np.int16); overlap = 0
    for k, (L, cy, cx, ang) in enumerate(glyphs):
        p = _placed(L, cy, cx, ang, scale)
        y0, x0 = np.floor(p.min(0) - r - 1).astype(int); y1, x1 = np.ceil(p.max(0) + r + 2).astype(int)
        y0, x0, y1, x1 = max(y0, 0), max(x0, 0), min(y1, shape[0]), min(x1, shape[1])
        if y1 <= y0 or x1 <= x0: continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d, _ = cKDTree(p).query(np.stack([yy.ravel(), xx.ravel()], 1), distance_upper_bound=r + 1.0)
        on = (d <= r).reshape(yy.shape); sub = lab[y0:y1, x0:x1]
        overlap += int((on & (sub > 0)).sum()); sub[on & (sub == 0)] = k + 1
    return lab, overlap


def soft(binary, sigma):
    return ndi.gaussian_filter(binary.astype(np.float32), sigma, mode="constant", truncate=4.0)


def geometry(lab, cap, stroke, valid):
    """strokes, counters, far background (bool), each restricted to valid pixels (D5, D6)."""
    binary = lab > 0
    dist = ndi.distance_transform_edt(~binary)  # to the nearest glyph pixel
    hull = np.zeros_like(binary)
    for k, sl in enumerate(ndi.find_objects(lab)):
        if sl is None: continue
        hull[sl] |= convex_hull_image(lab[sl] == k + 1)
    counters = hull & (dist > stroke / 2.0)
    far = dist > REG["far_caps"] * cap
    return dict(strokes=binary & valid, counters=counters & valid, far=far & valid)



def _row_glyphs(letters, angles, axis, cap):
    """Glyph centres of one row relative to the first letter, along x (axis 'x') or y (axis 'y')."""
    P = 1.2 * cap
    return [(LETTERS[L], (k * P if axis == "y" else 0.0), (k * P if axis == "x" else 0.0), float(a))
            for k, (L, a) in enumerate(zip(letters, angles))]


def plan_layout(valid, cap, stroke, seed, border=64, *, ink):
    """A1 (amendment 1). One row along image x, letters upright (each keeps its registered +/- 5 degree jitter), as
    many letters as fit at the registered pitch, at most REG letters_max and at least letters_min: the letters and
    angles are the first n of the host's 7 seeded draws. Placed at the first position from the top where every glyph
    pixel is at least `border` px clear of the crop border and of no-data pixels AND (amendment 3, C2) more than 48 px
    from any host-labelled ink pixel, so no 16 to 48 px annulus reaches it; centred along x among the positions
    feasible at that row. Returns dict(axis=None, n=0) when not even letters_min letters fit."""
    rng = np.random.default_rng([seed, 1])
    letters = rng.integers(0, 24, REG["letters_max"]); angles = rng.uniform(-REG["rot_deg"], REG["rot_deg"], REG["letters_max"])
    H, W = valid.shape
    padded = np.pad(valid, 1, constant_values=False)
    allowed = ndi.distance_transform_edt(padded)[1:-1, 1:-1] > border  # >= border clear px to any edge or no-data pixel
    allowed &= ndi.distance_transform_edt(~ink) > REG["ink_clear_px"]  # C2
    bad = (~allowed).astype(np.float32)
    scale, r = (cap - stroke) / ref_height(), stroke / 2.0
    for n in range(REG["letters_max"], REG["letters_min"] - 1, -1):
        rel = _row_glyphs(letters[:n], angles[:n], "x", cap)
        pts = np.concatenate([_placed(L, cy, cx, a, scale) for L, cy, cx, a in rel])
        off = np.floor(pts.min(0) - r - 2)
        size = (np.ceil(pts.max(0) + r + 3) - off).astype(int)
        if size[0] > H or size[1] > W: continue
        local = [(L, cy - off[0], cx - off[1], a) for L, cy, cx, a in rel]
        fp, _ = draw(tuple(size), local, cap, stroke)
        ok = fftconvolve(bad, (fp > 0)[::-1, ::-1].astype(np.float32), mode="valid") < 0.5
        if not ok.any(): continue
        oy = int(np.nonzero(ok.any(1))[0][0]); xs = np.nonzero(ok[oy])[0]
        ox = int(xs[np.argmin(np.abs(xs - (W - size[1]) / 2.0))])
        glyphs = [(L, cy + oy, cx + ox, a) for L, cy, cx, a in local]
        return dict(axis="x", n=n, glyphs=glyphs, origin=[oy, ox], footprint=[int(size[0]), int(size[1])], letters="".join(g[0] for g in glyphs))
    return dict(axis=None, n=0, glyphs=[], letters="".join(LETTERS[L] for L in letters))


def build(valid, cap, stroke, layout, sigma=0.7):
    """Binary glyph mask, soft mask G (Gaussian sigma px), label image and geometry for a planned layout."""
    lab, overlap = draw(valid.shape, layout["glyphs"], cap, stroke)
    return dict(lab=lab, binary=lab > 0, G=soft(lab > 0, sigma), overlap_px=overlap, **geometry(lab, cap, stroke, valid))
