#!/usr/bin/env python3
"""atlas_pack.py: package the PHerc0800 five-model First Letters ink atlas (results/fl/images5/PHerc0800, about 2 GB)
for a GitHub release without ever copying the image tree, because the disk is the constraint.

  atlas_pack.py build             thumbnails (THUMB_W px wide, ensemble and CT), index.html and MANIFEST.sha256
  atlas_pack.py readme            README.md generated from the result files, then MANIFEST.sha256 again
  atlas_pack.py zip [--only images|meta]
                                  image zips of whole z windows (ZIP_STORED, read straight from the source directories,
                                  one at a time, each refused if it would leave less than FREE_FLOOR free), then one
                                  small zip with README.md, index.html, the thumbnails and MANIFEST.sha256
  atlas_pack.py verify            manifest against the zip members and the source files, mesh coverage, per-mesh file
                                  counts, every index.html link against the zip contents; one summary line, rc 1 on
                                  any failure

Release layout, the same folder in every zip so they all extract into one place:
  pherc0800_atlas_v1/README.md, index.html, MANIFEST.sha256, thumbs/<image>_w<THUMB_W>.png
  pherc0800_atlas_v1/images/<mesh>/<image>.png, <name>__stats.json, preview/<image>.png
MANIFEST.sha256 lists every release file except itself, relative to pherc0800_atlas_v1/ (shasum -a 256 -c works there).
Once results/fl/confirm/RESULT.json exists: readme, zip --only meta, verify.
Files written: data/fl/release/pherc0800_atlas_v1/ (small files), data/fl/release/*.zip, data/fl/release/zips.json.
"""
import argparse, ast, collections, csv, hashlib, html, json, math, os, re, shutil, subprocess, sys, tempfile, time
import zipfile
from pathlib import Path

ROOT = Path('/Users/andreashad04/money/vesuv')
SCROLL = 'PHerc0800'
SRC = ROOT / 'results/fl/images5' / SCROLL
REL = ROOT / 'data/fl/release'
NAME = 'pherc0800_atlas_v1'
OUT = REL / NAME
HASHCACHE = REL / '.hashcache.json'
ZIPS_JSON = REL / 'zips.json'
MANIFEST = 'MANIFEST.sha256'
INDEX_ZIP = f'{NAME}_index.zip'
THUMB_W = 400
THUMB_KINDS = ('ensemble', 'ct')
ZIP_CAP = 600 * 10 ** 6          # bytes per image zip, member data plus zip headers
FREE_FLOOR = 42 * 2 ** 30        # a running GPU pipeline refuses under 40e9 free; never leave less than this
INV = ROOT / f'results/fl/inventory_{SCROLL}.csv'
AXIS = ROOT / f'results/fl/axis_{SCROLL}.json'
ORIENT_CONTROL = ROOT / 'results/fl/orient_control.json'
RV3_SUMMARY = ROOT / 'results/readerv3/pc343_s1_summary.json'
RV3_PREREG = ROOT / 'results/readerv3/PREREGISTRATION_RV3.md'
RV3_HF = ROOT / 'models/reader_v3/HF_README.md'
FETCH = ROOT / 'models/nieuwlaar_ckpt/fetch_record.json'
PREREG = ROOT / 'results/fl/PREREGISTRATION_FL_SCREEN.md'
UNSEALED = ROOT / 'results/fl/review/UNSEALED.json'
CROPS = ROOT / 'results/fl/review/neighbour_crops.json'
REVIEW5_FLAGS = ROOT / 'results/fl/review5/flags'
NEIGHBOUR_V1 = ROOT / 'results/fl/neighbour/RESULT.json'
NEIGHBOUR = ROOT / 'results/fl/neighbour/RESULT_v2.json'
CONFIRM = ROOT / 'results/fl/confirm/RESULT.json'
SWEEP = ROOT / 'research/SWEEP_2026-10-24.md'
VILLA = ROOT / 'villa-main'
VILLA_README = VILLA / 'vesuvius/README.md'
VILLA_SCROLLS = VILLA / 'vesuvius/src/vesuvius/install/configs/scrolls.yaml'
NLREPO = ROOT / 'results/nieuwlaar/repo'
PSCREPO = ROOT / 'data/fl/pscamillo'
SCREEN_PY = ROOT / 'ops/fl/screen_unit.py'
ENSEMBLE_PY = ROOT / 'ops/fl/ensemble.py'
RENDER66_PY = ROOT / 'public/firstletters/render66.py'
RT_PY = ROOT / 'public/firstletters/render_tifxyz.py'
DASHES = (chr(0x2013), chr(0x2014))

sys.path.insert(0, str(ROOT / 'ops/nieuwlaar'))
sys.path.insert(0, str(ROOT / 'ops/fl'))


def die(msg):
    raise SystemExit(f'atlas_pack: {msg}')


def write_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data.encode('utf-8') if isinstance(data, str) else data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def sha256_stream(f):
    h = hashlib.sha256()
    for b in iter(lambda: f.read(1 << 23), b''):
        h.update(b)
    return h.hexdigest()


def sha256_file(p):
    with open(p, 'rb') as f:
        return sha256_stream(f)


class HashCache:
    """sha256 keyed by (path, size, mtime_ns) so build and readme do not both re-read 2 GB; verify never uses it"""

    def __init__(self):
        try:
            self.d = json.loads(HASHCACHE.read_text())
        except (OSError, ValueError):
            self.d = {}
        self.dirty = False

    def get(self, p):
        st = os.stat(p)
        key, sig = str(p), [st.st_size, st.st_mtime_ns]
        e = self.d.get(key)
        if e and e[:2] == sig:
            return e[2]
        h = sha256_file(p)
        self.d[key] = sig + [h]
        self.dirty = True
        return h

    def save(self):
        if self.dirty:
            write_atomic(HASHCACHE, json.dumps(self.d, sort_keys=True))


def image_name(base, kind, dtag):
    return f'{base}__{kind}__{dtag}.png'


def thumb_rel(base, kind, dtag):
    return f'thumbs/{base}__{kind}__{dtag}_w{THUMB_W}.png'


def mesh_files(base, kinds, dtag):
    full = [image_name(base, k, dtag) for k in kinds]
    return full + [f'{base}__stats.json'] + [f'preview/{f}' for f in full]


def layout():
    """rows (orient.mesh_table joined with the inventory and each stats.json, sorted by z window then winding), the
    image kinds, the depth tag and the release's image files [(release path, source path)]"""
    import orient
    import run_unit
    table = {r['mesh']: dict(r) for r in orient.mesh_table(SCROLL)}
    with open(INV, newline='') as f:
        inv = {r['mesh']: r for r in csv.DictReader(f)}
    dirs = {p.name for p in SRC.iterdir() if p.is_dir()}
    if not (set(table) == set(inv) == dirs):
        die(f'mesh table ({len(table)}), inventory ({len(inv)}) and image directories ({len(dirs)}) disagree: '
            f'{sorted((set(table) ^ set(inv)) | (set(table) ^ dirs))[:10]}')
    kinds = depths = None
    rows = []
    for m, r in table.items():
        i = inv[m]
        if int(i['window_z']) != r['window_z'] or abs(float(i['area_cm2']) - r['area_cm2']) > 1e-9 \
                or i['wrap'] != r['wrap']:
            die(f'{m}: inventory and mesh table disagree')
        base = f'{SCROLL}_{m}_{run_unit.NL}'
        st = json.loads((SRC / m / f'{base}__stats.json').read_text())
        k = ['ensemble', 'ct'] + list(st['models'])
        d = st['ensemble']['depths']
        if kinds is None:
            kinds, depths = k, d
        elif k != kinds or d != depths:
            die(f'{m}: image kinds {k} or depths {d} differ from {kinds} {depths}')
        r.update(inv=i, stats=st, base=base)
        rows.append(r)
    if len(depths) != 1:
        die(f'expected one image depth, the stats give {depths}')
    dtag = f'd{depths[0]:+03d}'
    rows.sort(key=lambda r: (r['window_z'], r['wrap'], r['mesh']))
    files = []
    for r in rows:
        r['files'] = mesh_files(r['base'], kinds, dtag)
        for rel in r['files']:
            p = SRC / r['mesh'] / rel
            if not p.is_file():
                die(f'missing {p}')
            files.append((f'images/{r["mesh"]}/{rel}', p))
    return rows, kinds, dtag, files


def member_overhead(arc):
    n = len(arc.encode('utf-8'))
    return 30 + n + 46 + n + 64       # local header, central directory entry, allowance for zip64 and extra fields


def plan_zips(rows, files):
    """whole z windows per zip, contiguous in z: the fewest zips that keep each one (data plus headers) under ZIP_CAP,
    and among those the split whose largest zip is smallest"""
    wz = {r['mesh']: r['window_z'] for r in rows}
    per = collections.defaultdict(int)
    for arc, p in files:
        per[wz[arc.split('/')[1]]] += p.stat().st_size + member_overhead(f'{NAME}/{arc}')
    W = sorted(per)
    s = [per[w] for w in W]
    if max(s) + 22 > ZIP_CAP:
        die(f'one z window alone exceeds the zip cap: {max(s)} bytes')
    pre = [0]
    for x in s:
        pre.append(pre[-1] + x)
    n, groups = len(W), None
    for k in range(max(1, math.ceil(pre[-1] / ZIP_CAP)), n + 1):
        best = [[math.inf] * (n + 1) for _ in range(k + 1)]
        cut = [[0] * (n + 1) for _ in range(k + 1)]
        best[0][0] = 0
        for j in range(1, k + 1):
            for i in range(j, n + 1):
                for t in range(j - 1, i):
                    v = max(best[j - 1][t], pre[i] - pre[t])
                    if v < best[j][i]:
                        best[j][i], cut[j][i] = v, t
        if best[k][n] + 22 <= ZIP_CAP:
            groups, i = [], n
            for j in range(k, 0, -1):
                t = cut[j][i]
                groups.append(W[t:i])
                i = t
            groups.reverse()
            break
    plan = []
    for g in groups:
        name = f'{NAME}_z{g[0]}-z{g[-1]}.zip' if len(g) > 1 else f'{NAME}_z{g[0]}.zip'
        plan.append(dict(zip=name, windows=g, meshes=[r['mesh'] for r in rows if r['window_z'] in g],
                         members=[(arc, p) for arc, p in files if wz[arc.split('/')[1]] in g],
                         est_bytes=sum(per[w] for w in g) + 22))
    return plan


def small_files(rows, dtag, need_readme):
    """release files kept under OUT, the manifest excepted"""
    rels = [thumb_rel(r['base'], k, dtag) for r in rows for k in THUMB_KINDS] + ['index.html']
    if (OUT / 'README.md').is_file():
        rels.append('README.md')
    elif need_readme:
        die('README.md is missing: run readme first')
    return rels


def write_manifest(rows, dtag, files):
    cache = HashCache()
    entries = [(arc, cache.get(p)) for arc, p in files]
    cache.save()
    entries += [(rel, sha256_file(OUT / rel)) for rel in small_files(rows, dtag, False)]
    entries.sort()
    write_atomic(OUT / MANIFEST, ''.join(f'{h}  {rel}\n' for rel, h in entries))
    return entries


def read_manifest(text=None):
    text = (OUT / MANIFEST).read_text() if text is None else text
    out = {}
    for line in text.splitlines():
        h, rel = line.split('  ', 1)
        if rel in out:
            die(f'duplicate manifest entry {rel}')
        out[rel] = h
    return out


def make_thumbs(rows, dtag):
    from PIL import Image
    made, sizes = 0, {}
    for r in rows:
        for k in THUMB_KINDS:
            src = SRC / r['mesh'] / image_name(r['base'], k, dtag)
            dst = OUT / thumb_rel(r['base'], k, dtag)
            if not (dst.is_file() and dst.stat().st_mtime_ns >= src.stat().st_mtime_ns):
                with Image.open(src) as im:
                    im.load()
                    w, h = im.size
                    th = im.resize((THUMB_W, max(1, round(h * THUMB_W / w))), Image.Resampling.LANCZOS)
                dst.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=f'.{dst.stem}.', suffix='.png')
                os.close(fd)
                try:
                    th.save(tmp, format='PNG', optimize=True)
                    os.replace(tmp, dst)
                except BaseException:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                    raise
                made += 1
            with Image.open(dst) as im:
                sizes[(r['mesh'], k)] = im.size
    return made, sizes


CSS = """
:root { --bg: #f8f7f3; --fg: #1c1c1a; --muted: #5b5952; --line: #dcd9cf; --card: #ffffff; --link: #1d5ba8; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #141413; --fg: #ebeae4; --muted: #a8a59b; --line: #34332e; --card: #1d1d1b; --link: #8db6ef; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
header, main { max-width: 1320px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.5rem; margin: 8px 0; }
h2 { font-size: 1.15rem; margin: 28px 0 10px; padding-bottom: 4px; border-bottom: 1px solid var(--line); }
h2 small { color: var(--muted); font-weight: normal; font-size: .85rem; }
h3 { font: 600 .95rem ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 0 0 2px; }
a { color: var(--link); }
nav a { margin-right: 10px; white-space: nowrap; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 430px), 1fr)); gap: 14px; }
article { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 10px; }
figure { margin: 6px 0; }
figure img { display: block; width: 100%; max-width: 400px; height: auto; background: #000; }
figcaption, .meta, .links { color: var(--muted); font-size: .82rem; }
.links a { margin-right: 6px; }
"""


def index_html(rows, kinds, dtag, plan, sizes):
    e = html.escape
    zip_of = {m: z['zip'] for z in plan for m in z['meshes']}
    by_z = collections.OrderedDict()
    for r in rows:
        by_z.setdefault(r['window_z'], []).append(r)
    depth = rows[0]['stats']['ensemble']['depths'][0]
    area = sum(r['area_cm2'] for r in rows)
    out = ['<!doctype html>', '<html lang="en">', '<head>', '<meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           f'<title>{SCROLL} ink atlas</title>', f'<style>{CSS}</style>', '</head>', '<body>', '<header>',
           f'<h1>{SCROLL} First Letters ink atlas, v1</h1>',
           f'<p>Five-model ink images at d = {depth:+d} of {len(rows)} community meshes ({area:.2f} cm2 in '
           f'{len(by_z)} z windows), each beside its CT layer. Candidates only: no letters are claimed. README.md has '
           f'the recipe, the checks and the limits.</p>',
           '<p>Click a thumbnail for the full-resolution image. The links work once every zip of the release is '
           'extracted into the same folder.</p>',
           '<nav>z windows: ' + ' '.join(f'<a href="#z{z}">{z}</a>' for z in by_z) + '</nav>', '</header>', '<main>']
    for z, rs in by_z.items():
        zips = sorted({zip_of[r['mesh']] for r in rs})
        out.append(f'<section id="z{z}"><h2>z window {z} <small>({len(rs)} meshes, '
                   f'{sum(r["area_cm2"] for r in rs):.2f} cm2, in {e(", ".join(zips))})</small></h2>')
        out.append('<div class="grid">')
        for r in rs:
            m, b, st = r['mesh'], r['base'], r['stats']
            o = st['orientation']
            img = lambda k: e(f'images/{m}/{image_name(b, k, dtag)}')
            out.append(f'<article id="{e(m)}">')
            out.append(f'<h3>{e(m)}</h3>')
            out.append(f'<p class="meta">area {r["area_cm2"]:.2f} cm2, z window {z}, winding {e(r["wrap"])}, '
                       f'{st["W"]} x {st["H"]} px, flips lr {str(o["flip_lr"]).lower()} ud '
                       f'{str(o["flip_ud"]).lower()}, in {e(zip_of[m])}</p>')
            for k in THUMB_KINDS:
                w, h = sizes[(m, k)]
                label = 'CT layer' if k == 'ct' else f'{k}, d = {depth:+d}'
                out.append(f'<figure><a href="{img(k)}"><img src="{e(thumb_rel(b, k, dtag))}" width="{w}" '
                           f'height="{h}" loading="lazy" alt="{e(m)} {e(label)}"></a>'
                           f'<figcaption>{e(label)}</figcaption></figure>')
            full = ' '.join(f'<a href="{img(k)}">{e(k)}</a>' for k in kinds)
            prev = ' '.join(f'<a href="{e(f"images/{m}/preview/{image_name(b, k, dtag)}")}">{e(k)}</a>'
                            for k in kinds)
            out.append(f'<p class="links">full resolution: {full}<br>previews: {prev}<br>'
                       f'<a href="{e(f"images/{m}/{b}__stats.json")}">stats.json</a></p>')
            out.append('</article>')
        out.append('</div></section>')
    out += ['</main>', '</body>', '</html>', '']
    return '\n'.join(out)


def cmd_build(a):
    t0 = time.time()
    rows, kinds, dtag, files = layout()
    plan = plan_zips(rows, files)
    made, sizes = make_thumbs(rows, dtag)
    write_atomic(OUT / 'index.html', index_html(rows, kinds, dtag, plan, sizes))
    entries = write_manifest(rows, dtag, files)
    shipped = {p for _, p in files}
    skipped = collections.Counter(re.sub(r'z\d+_w\d+', '<mesh>', p.name) for p in SRC.rglob('*')
                                  if p.is_file() and p not in shipped)
    print(f'build: {len(rows)} meshes, {len(files)} image files, {made} thumbnails made '
          f'({len(rows) * len(THUMB_KINDS)} expected), index.html, {MANIFEST} with {len(entries)} entries'
          f'{"" if (OUT / "README.md").is_file() else " (README.md not written yet: readme rewrites the manifest)"}; '
          f'source files left out: {dict(skipped)}; zip plan: '
          + ', '.join(f'{z["zip"]} ({len(z["meshes"])} meshes, about {z["est_bytes"] / 1e6:.1f} MB)' for z in plan)
          + f'; {time.time() - t0:.1f} s')


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def docstring(path):
    return ast.get_docstring(ast.parse(Path(path).read_text()), clean=False) or ''


def paragraph(text, start, stop=None):
    """the paragraph of text that begins at the first occurrence of start, cut before stop, lines joined"""
    i = text.find(start)
    if i < 0:
        die(f'cannot find {start!r} in a source docstring')
    p = text[i:].split('\n\n')[0]
    if stop and stop in p:
        p = p[:p.index(stop)]
    return ' '.join(x.strip() for x in p.splitlines()).strip()


def licence_holder(repo):
    lines = (Path(repo) / 'LICENSE').read_text().splitlines()
    m = re.search(r'Copyright \(c\) \d{4} (.+)', '\n'.join(lines[:5]))
    return lines[0].strip(), (m.group(1).strip() if m else 'unknown')


def villa_data_licence():
    text = VILLA_README.read_text()
    if '## Data licence' not in text:
        die(f'{VILLA_README} has no "Data licence" section')
    sec = text.split('## Data licence', 1)[1]
    m = re.search(r'All Vesuvius Challenge datasets are released under (.+?) \((https?://[^)\s]+)\), except Scrolls '
                  r'(\d+)\W(\d+) and Fragments (\d+)\W(\d+)', sec)
    if not m:
        die('the data licence sentence in villa README changed: read it and update atlas_pack.py')
    commit = git(VILLA, 'log', '-1', '--format=%h', '--grep=(#1913)')
    head = git(VILLA, 'rev-parse', '--short', 'HEAD')
    scrolls = {}
    for num, name in re.findall(r'full-scrolls/Scroll(\d+)/([A-Za-z0-9]+)\.volpkg', VILLA_SCROLLS.read_text()):
        scrolls.setdefault(int(num), name)
    lo, hi = int(m.group(3)), int(m.group(4))
    named = [scrolls.get(i) for i in range(lo, hi + 1)]
    return dict(licence=m.group(1), url=m.group(2), scrolls=(lo, hi), fragments=(int(m.group(5)), int(m.group(6))),
                commit=commit, head=head, named=named,
                ours_excluded=None if None in named else all(SCROLL.replace('0', '') != n.replace('0', '') and
                                                             SCROLL != n for n in named))


def table(header, rows):
    return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + \
           ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]


def readme_text(rows, kinds, dtag, files, plan):
    import orient
    import rv3_unit as rv3
    from PIL import Image
    ru = rv3.ru
    st0 = rows[0]['stats']
    depth = st0['ensemble']['depths'][0]
    dl = f'd = {depth:+d}'
    tags = list(st0['models'])
    keys = [st0['models'][t]['model'] for t in tags]
    n, area = len(rows), sum(r['area_cm2'] for r in rows)
    zs = sorted({r['window_z'] for r in rows})
    wraps = sorted({r['wrap'] for r in rows})
    Hs, Ws = [r['stats']['H'] for r in rows], [r['stats']['W'] for r in rows]
    data_bytes = sum(p.stat().st_size for _, p in files)
    pw = 0
    for r in rows:
        with Image.open(SRC / r['mesh'] / 'preview' / image_name(r['base'], 'ensemble', dtag)) as im:
            pw = max(pw, im.size[0])
    unseal = json.load(open(UNSEALED))
    crops = json.load(open(CROPS))
    nb = json.load(open(NEIGHBOUR))
    confirm = json.load(open(CONFIRM)) if CONFIRM.is_file() else None
    status = 'pending' if confirm is None else ('VALID, see section 8' if confirm.get('valid') else
              'VOID: the control did not pass, so no candidate is confirmed (section 8)')
    L = []
    add = L.extend

    add([f'# {SCROLL} First Letters ink atlas, v1', '',
         f'Five-model ink images of all {n} community meshes of {SCROLL}, the eligible First Letters scroll we '
         f'screened. **Candidates only: no letters are claimed.** Blind-side confirmation of the candidates: '
         f'**{status}**.', ''])

    add(['## 1. What this is', '',
         f'- {n} meshes, {area:.2f} cm2 in total, in {len(zs)} z windows (z {zs[0]} to {zs[-1]}), windings '
         f'{", ".join(wraps)}.',
         f'- Per mesh {len(rows[0]["files"])} files: the ensemble image, the CT layer and each model\'s own image '
         f'({", ".join(tags)}), all at {dl}, at full resolution ({min(Ws)} to {max(Ws)} px wide, {min(Hs)} to '
         f'{max(Hs)} px high) and as previews (at most {pw} px wide), plus `stats.json` (the stretch of every model, '
         f'the orientation flips and the sha256 of every input map).',
         f'- {len(files)} files under `images/`, {data_bytes / 1e9:.2f} GB, in {len(plan)} zips grouped by z window; '
         f'`index.html` shows {THUMB_W} px thumbnails of each mesh\'s ensemble and CT, grouped by z window, linked to '
         f'the full-resolution files.', '',
         'How to use: download every zip, extract them all into one folder (each holds `' + NAME + '/`), open `'
         + NAME + '/index.html`. Check the files with `cd ' + NAME + ' && shasum -a 256 -c ' + MANIFEST + '`.', ''])
    add(table(['zip', 'z windows', 'meshes', 'data (MB)'],
              [[f'`{z["zip"]}`', f'{z["windows"][0]} to {z["windows"][-1]}', len(z['meshes']),
                f'{sum(p.stat().st_size for _, p in z["members"]) / 1e6:.1f}'] for z in plan]
              + [[f'`{INDEX_ZIP}`', 'all', 'README.md, index.html, thumbnails, manifest', '']]))
    L.append('')

    nl_url, nl_head = git(NLREPO, 'remote', 'get-url', 'origin'), git(NLREPO, 'rev-parse', '--short', 'HEAD')
    nl_lic, nl_name = licence_holder(NLREPO)
    ens_doc = docstring(ENSEMBLE_PY)
    add(['## 2. Recipe', '',
         f'{nl_name}\'s PHerc0343 First Letters recipe ({nl_url.removesuffix(".git")}, commit {nl_head}), reproduced '
         f'on an Apple Silicon Mac (Metal, MPS) with our scripts. Screen of one mesh (`ops/fl/screen_unit.py`):', '',
         '> ' + paragraph(docstring(SCREEN_PY), 'First Letters screen').rstrip(), '',
         'Render convention (`firstletters/render66.py` of this repository, a CPU renderer that follows his VC3D render):', '',
         '> ' + paragraph(docstring(RENDER66_PY), 'Convention'), '',
         'Ensemble (`ops/fl/ensemble.py`, his `ensemble_maps.py`):', '',
         '> ' + paragraph(ens_doc, 'Statistics are', 'Differences:'), '',
         f'The images here are at {dl} on the reading side, with {len(keys)} models: the four public 9 um models of '
         f'his recipe plus Reader V3 as the fifth (`ops/fl/rv3_with.py` appends it to the model table at run time, '
         f'last, so the four keep his order). Each layer stack has {ru.NL} layers.', ''])

    fetch = json.load(open(FETCH))
    ck_rows, notes = [], []
    for t, k in zip(tags, keys):
        p = ru.CKPT.get(k, rv3.CKPT if k == rv3.MODEL else None)
        if p is None:
            die(f'no checkpoint for model {k}')
        p = Path(p)
        h = sha256_file(p)
        ck_rows.append([t, f'`{k}`', f'`{p.name}`', f'{p.resolve().stat().st_size:,}', f'`{h}`'])
        if k == rv3.MODEL:
            if h != rv3.CKPT_SHA256:
                die(f'Reader V3 checkpoint {h} is not the pinned {rv3.CKPT_SHA256}')
            hf = RV3_HF.read_text()
            author = re.search(r'Author: ([^.\n]+)', hf)
            gh = re.search(r'https://github\.com/[^\s)]+', hf)
            lic = 'MIT' if re.search(r'^license: mit$', hf, re.M) else 'see its README'
            notes.append(f'- `{k}`: Reader V3 by {author.group(1).strip() if author else "its author"} '
                         f'({gh.group(0) if gh else "see its model card"}), licence {lic}; the file matches the sha256 '
                         f'pinned in `ops/fl/rv3_unit.py`.')
        if k in fetch and isinstance(fetch[k], dict) and fetch[k].get('note'):
            notes.append(f'- `{k}`: {fetch[k]["note"]} (`models/nieuwlaar_ckpt/fetch_record.json`, exact: '
                         f'{str(fetch[k].get("exact")).lower()}).')
    add(['### Models and checkpoints', '',
         'sha256 computed from the files that `ops/nieuwlaar/run_unit.py` CKPT and `ops/fl/rv3_unit.py` CKPT point at:',
         ''])
    add(table(['image', 'model', 'checkpoint', 'bytes', 'sha256'], ck_rows))
    add([''] + notes + [''])

    s = json.load(open(RV3_SUMMARY))
    arms, dec = s['arms'], s['decision']
    four = [k for k in keys if k != rv3.MODEL]
    def arm(models, rotated):
        hit = [v for v in arms.values() if sorted(v['models']) == sorted(models) and v['rotated_rv3'] == rotated]
        if len(hit) != 1:
            die(f'cannot identify the Reader V3 arm {models} rotated={rotated}')
        return hit[0]
    B, C = arm(four, False), arm(four + [rv3.MODEL], False)
    A, N1, N1C = arm([rv3.MODEL], False), arm([rv3.MODEL], True), arm(four + [rv3.MODEL], True)
    nbox = len(B['per_box'])
    ref = dec['reference_single']
    rv3pre = ' '.join(RV3_PREREG.read_text().split())
    lim = re.search(r'Training data includes .*?It is not a held-out test of Reader V3\.', rv3pre)
    endpoint = re.search(r'This endpoint measures agreement with his four-model image:.*?\.', rv3pre)
    add(['## 3. Why Reader V3 joined', '',
         f'Tested on his PHerc0343 segment against his published image and his {nbox} letter boxes '
         f'(`results/readerv3/pc343_s1_summary.json`, binding: {str(dec["binding"]).lower()}, written {s["written"]}). '
         f'r_alpha is the Pearson correlation of our image with his over the compared area; letter boxes agree when '
         f'our ink fraction in the box agrees with his.', ''])
    add(table(['arm', 'models', 'Reader V3 rotated', 'letter boxes agreeing', 'letter-box r_alpha', 'r_alpha'],
              [[k, ', '.join(v['models']), str(v['rotated_rv3']).lower(), f'{v["boxes_agree"]} of {len(v["per_box"])}',
                f'{v["r_alpha_letters"]:.4f}', f'{v["r_alpha"]:.4f}'] for k, v in arms.items()]))
    add(['',
         f'The four public models agree on {B["boxes_agree"]} of {nbox} boxes and the same four plus Reader V3 on '
         f'{C["boxes_agree"]}. Reader V3 alone agrees on {A["boxes_agree"]} (letter-box r_alpha '
         f'{A["r_alpha_letters"]:.4f}, under the best single public model, {dec["reference_single_model"]}, at '
         f'{ref["r_alpha_letters"]:.4f}), so it is not used alone. Rotated it agrees on {N1["boxes_agree"]}, and the '
         f'four plus the rotated copy agree on {N1C["boxes_agree"]}, so the gain comes from Reader V3\'s aligned '
         f'content, not from having a fifth input. Verdict: {dec["verdict"]}.', '',
         f'Limit (from `results/readerv3/PREREGISTRATION_RV3.md`): {lim.group(0) if lim else "see that file"}'
         + (f' {endpoint.group(0)}' if endpoint else ''), ''])

    axis, oc = json.load(open(AXIS)), json.load(open(ORIENT_CONTROL))
    cp = axis['control_points']
    czs = [p['z'] if isinstance(p, dict) else p[2] for p in cp]
    inv = [r['inv'] for r in rows]
    of = sorted(float(i['outward_fraction_chosen']) for i in inv)
    conf = sum(i['confident'] == 'True' for i in inv)
    signs = collections.Counter(i['chosen_sign'] for i in inv)
    add(['## 4. Orientation: the axis and its control', '',
         f'- Axis (`results/fl/axis_{SCROLL}.json`): {axis["source"]}. Method: {axis["method"]}. '
         f'{len(cp)} control points, z {min(czs)} to {max(czs)}. It is our estimate, not an official umbilicus.',
         f'- Per mesh, +d points away from this axis: the sign of the render normal is chosen per mesh '
         f'(`results/fl/inventory_{SCROLL}.csv`): sign {", ".join(f"{int(k):+d} on {v}" for k, v in sorted(signs.items()))} '
         f'meshes; outward fraction of the chosen sign {of[0]:.4f} to {of[-1]:.4f}; {conf} of {n} meshes confident '
         f'(at least {orient.MIN_CONFIDENT}).',
         f'- Control (`results/fl/orient_control.json`): on his PHerc0343 mesh with his umbilicus our measure gives '
         f'{oc["mine_sign_plus1"]} outward for sign +1 (his own measure {oc["his_measure_orientation"]}, his sidecar '
         f'{oc["his_sidecar"]}), {oc["mine_sign_minus1"]} for sign -1 and {oc["mine_columns_reversed"]} with the '
         f'columns reversed; all checks pass: {str(oc["all_pass"]).lower()}.', ''])

    ens_src = ENSEMBLE_PY.read_text().splitlines()
    flips = [x.strip() for x in ens_src if re.search(r'b = b\[::-1\] if flip_ud|b\[:, ::-1\] if flip_lr', x)]
    grid = [x.strip() for x in RT_PY.read_text().splitlines()
            if re.match(r'\s*(self\.W = |self\.H = |gy = self\.sy|gx = self\.sx)', x)]
    if len(flips) != 2 or len(grid) != 4:
        die(f'could not find the flip lines ({len(flips)}) or the grid lines ({len(grid)}) in the sources')
    reading = re.search(r"orientation 'reading' = (.*?), MEASURED", ' '.join(ens_doc.split()))
    fc = collections.Counter((r['stats']['orientation']['flip_lr'], r['stats']['orientation']['flip_ud'])
                             for r in rows)
    modes = collections.Counter(r['stats']['orientation']['mode'] for r in rows)
    add(['## 5. From an image pixel back to the mesh', '',
         f'Every image is in reading orientation (`orientation.mode` in stats.json: '
         f'{", ".join(f"{k} on {v}" for k, v in modes.items())} meshes)'
         f'{": " + reading.group(1) if reading else ""}. ensemble.py turns the render canvas with the mesh\'s own '
         f'flips (`orientation.flip_lr`, `orientation.flip_ud`, measured per mesh around the axis; the rule is in '
         f'`orientation.measured.rule`):', '', '```python'] + flips + ['```', '',
         'So pixel (r, c) of an H x W image comes from canvas pixel (r\', c\'): r\' = H - 1 - r when flip_ud is true, '
         'else r; c\' = W - 1 - c when flip_lr is true, else c. The canvas samples the mesh grid (x.tif, y.tif, z.tif '
         'of the tifxyz, Hs x Ws stored vertices, meta.json `scale` = (sx, sy)) at grid position (gy, gx) '
         '(`render_tifxyz.Surface`):', '', '```python'] + grid + ['```', '',
         'The 3D point and the layer offset at (gy, gx) follow the render convention in section 2. Flips across the '
         'meshes: ' + ', '.join(f'flip_lr {str(lr).lower()} and flip_ud {str(ud).lower()} on {v}'
                                 for (lr, ud), v in sorted(fc.items())) + ' meshes.', ''])

    pre = PREREG.read_text()
    sec = pre.split('## Review', 1)[1].split('\n## ', 1)[0]
    sec = '\n'.join(sec.splitlines()[1:]).strip().replace(' (a subagent)', ' [...]')
    regions = [g for g in crops['regions'] if g['scroll'] == SCROLL]
    reg_of = {}
    for i, g in enumerate(regions, 1):
        for sh, sl in g['flags']:
            reg_of[(g['mesh'], sh, sl)] = i
    flagged = [f for f in unseal['flagged'] if not f['control']]
    fl_rows = [[f'`{f["mesh"]}`', f['window'], f['sheet'], f['slot'], f'{f["valid_frac"]:.4f}',
                reg_of.get((f['mesh'], f['sheet'], f['slot']), '?')] for f in flagged]
    n_reg_flagged = len({reg_of.get((f['mesh'], f['sheet'], f['slot'])) for f in flagged})
    parts5 = sorted(REVIEW5_FLAGS.glob('part*.md')) if REVIEW5_FLAGS.is_dir() else []
    add(['## 6. Blinded screen review and its sensitivity', '',
         f'The review ran on the four-model screen images; the five-model images in this atlas have their own blinded '
         + (lambda u5: f'review with the same brief, complete ({len(parts5)} of {len(unseal["parts"])} reviewer parts): valid '
                        f'{str(u5["valid"]).lower()}, {u5["n_flagged_control_on_letters"]} of {u5["n_control_windows_with_letters"]} '
                        f'control windows holding his letters flagged, {u5["n_flagged_screen"]} PHerc0800 windows flagged '
                        f'(`results/fl/review5/UNSEALED.json`; Reader V3 trained on six PHerc0343P segments, so the control is '
                        f'not held out for it); those windows were not taken further because the blind-side confirmation '
                        f'design (section 8) was VOID. ')(json.load(open(ROOT / 'results/fl/review5/UNSEALED.json')))
         + f'Registered rule (`results/fl/PREREGISTRATION_FL_SCREEN.md`, section Review):', '']
        + ['> ' + x if x.strip() else '>' for x in sec.splitlines()] + ['',
         f'Result (`results/fl/review/UNSEALED.json`, valid: {str(unseal["valid"]).lower()}): {unseal["n_sheets"]} '
         f'contact sheets ({unseal["n_control_sheets"]} of them control sheets), {unseal["n_windows"]} windows, '
         f'{unseal["n_flagged"]} flagged: {unseal["n_flagged_screen"]} {SCROLL} windows in {n_reg_flagged} regions, and '
         f'{unseal["n_flagged_control"]} control window, which overlaps his letters (rule: {unseal["rule"]}). '
         f'Sensitivity is low: {unseal["n_flagged_control_on_letters"]} of the '
         f'{unseal["n_control_windows_with_letters"]} control windows that hold letters was flagged, so a window '
         f'without a flag is not a window without ink.', ''])
    add(table(['mesh', 'window', 'sheet', 'slot', 'valid fraction', 'region'], fl_rows))
    L.append('')

    res = [x for x in nb['results'] if not x['control']]
    ctl = [x for x in nb['results'] if x['control']]
    passed = [x for x in res if x['passed']]
    if len(passed) != nb['n_pass_screen']:
        die('RESULT_v2 pass count disagrees with its rows')
    reg_pass = {reg_of.get((x['mesh'], x['sheet'], x['slot'])) for x in passed}
    rr = [x[k] for x in res for k in ('r_d1_d0', 'r_d1_d2')]
    v1 = json.load(open(NEIGHBOUR_V1)) if NEIGHBOUR_V1.is_file() else {}
    order = ['d+0', 'd+1', 'd+2', 'neg']
    ctl_desc = ', '.join(f'{c["scroll"]} {c["sheet"]} slot {c["slot"]}' for c in ctl)
    add(['## 7. Neighbour-depth check', '',
         f'Rule (`results/fl/neighbour/RESULT_v2.json`, valid: {str(nb["valid"]).lower()}): {nb["rule"]}. The control '
         f'window ({ctl_desc}) passed: '
         f'{", ".join(str(c["passed"]).lower() for c in ctl)}.', '',
         f'{len(passed)} of {len(res)} {SCROLL} windows pass, in {len(reg_pass)} of {len(regions)} regions. '
         f'Adjacent depths correlate at r {min(rr):.2f} to {max(rr):.2f} in every window, passing or failing, so '
         f'depth consistency is expected of any structure in these maps: **passing is a necessary condition, not '
         f'evidence of ink.**'
         + (' The check ran twice: the first run was void because its control failed, and this run followed a mask '
            'fix registered after the first run was scored.' if v1.get('valid') is False else ''), ''])
    add(table(['mesh', 'sheet', 'slot', 'region'] + order + ['passed'],
              [[f'`{x["mesh"]}`', x['sheet'], x['slot'], reg_of.get((x['mesh'], x['sheet'], x['slot']), '?')]
               + [x['marks_by_depth'].get(k, '') for k in order] + [str(x['passed']).lower()] for x in res]))
    L.append('')

    add(['## 8. Blind-side confirmation', ''])
    if confirm is None:
        add(['**pending.** The registered confirmation (the passing windows at d = -3 to +3 on the reading side and on '
             'the blind side, with a control crop) is running; `results/fl/confirm/RESULT.json` does not exist yet. '
             'This README is regenerated when it does.', ''])
    else:
        rs = confirm['results']
        ctrl = [x for x in rs if x['control']]
        cm = ctrl[0]['marks'] if ctrl else {}
        if confirm['valid']:
            add([f'**VALID** (`results/fl/confirm/RESULT.json`): the control window passed; {confirm["n_pass_screen"]} of '
                 f'{len(rs) - len(ctrl)} {SCROLL} windows met the registered pattern. Rule: {confirm["rule"]}.', ''])
        else:
            add([f'**VOID** (`results/fl/confirm/RESULT.json`). The control window (his letters, PHerc0343) did not pass: '
                 f'READ {cm.get("read")}, BLIND {cm.get("blind")}, BLIND_OWN {cm.get("blind_own")}, negative '
                 f'{cm.get("neg")}, and an unclear mark can only fail. By the registered rule ({confirm["rule"]}) the '
                 f'confirmation is VOID and none of the windows below is confirmed, including the {confirm["n_pass_screen"]} '
                 f'that met the pattern. It was the only attempt under this design.', ''])
        add(table(['scroll', 'mesh', 'sheet', 'slot', 'control', 'READ', 'BLIND', 'BLIND_OWN', 'negative', 'met the pattern'],
                  [[x['scroll'], x['mesh'], x['sheet'], x['slot'], str(x['control']).lower(), x['marks']['read'],
                    x['marks']['blind'], x['marks']['blind_own'], x['marks']['neg'], str(x['passed']).lower()] for x in rs]))
        add(['', f'Key sha256 `{confirm["key_sha256"]}`, record sha256 `{confirm["record_sha256"]}`.', ''])

    psc_url, psc_head = git(PSCREPO, 'remote', 'get-url', 'origin'), git(PSCREPO, 'rev-parse', '--short', 'HEAD')
    psc_lic, psc_name = licence_holder(PSCREPO)
    sw = SWEEP.read_text().split('## 6. Threats', 1)[1].split('\n## ', 1)[0]
    item = re.search(r'^2\. (.*?)(?=^\d+\. |\Z)', sw, re.S | re.M)
    item = ' '.join(item.group(1).split()) if item else 'see research/SWEEP_2026-10-24.md section 6'
    add(['## 9. Limits', '',
         '- Candidates are candidates. No letters are claimed and no text is read from these images.',
         f'- The meshes are {psc_name}\'s community meshes ({psc_url.removesuffix(".git")}, commit {psc_head}), not '
         f'curated segments.',
         f'- Contrast audit (our sweep of 2026-10-24, section 6 item 2): {item}',
         f'- The axis that sets the outward direction and the reading flips is estimated, not official (section 4).',
         f'- Reader V3 trained on PHerc0343P segments, so its control is not a held-out test (section 3).',
         f'- The review is not sensitive: {unseal["n_flagged_control_on_letters"]} of '
         f'{unseal["n_control_windows_with_letters"]} control windows with letters was flagged (section 6).',
         f'- Passing the neighbour-depth check is a necessary condition, not evidence of ink (section 7).', ''])

    vl = villa_data_licence()
    if vl['ours_excluded'] is True:
        which = (f'In villa\'s `vesuvius/src/vesuvius/install/configs/scrolls.yaml` Scrolls {vl["scrolls"][0]} to '
                 f'{vl["scrolls"][1]} are {", ".join(vl["named"])}, so {SCROLL} is not one of the exceptions.')
    else:
        which = f'Whether {SCROLL} falls under the exception could not be read from villa\'s files.'
    add(['## 10. Licence', '',
         '- Code: MIT.',
         f'- Images: derived from Vesuvius Challenge data of {SCROLL} '
         f'(`{orient.SCROLLS[SCROLL]["volume"].rsplit("/", 1)[0]}`). villa\'s `vesuvius/README.md`, section "Data '
         f'licence" (the licence text moved there in villa #1913, commit {vl["commit"]}; read at villa commit '
         f'{vl["head"]}), says all Vesuvius Challenge datasets are released under {vl["licence"]} ({vl["url"]}), except '
         f'Scrolls {vl["scrolls"][0]} to {vl["scrolls"][1]} and Fragments {vl["fragments"][0]} to '
         f'{vl["fragments"][1]}, which have their own EduceLab-Scrolls terms. {which} These images carry the data\'s '
         f'terms; we add no licence of our own to them.',
         f'- Meshes: {psc_name}, licence file "{psc_lic}". Recipe: {nl_name}, licence file "{nl_lic}". Reader V3: '
         f'see section 2.', ''])

    vol = orient.SCROLLS[SCROLL]['volume']
    four_k = ' '.join(k for k in ru.CKPT)
    all_k = ' '.join(keys)
    dd = ' '.join(str(d) for d in st0['ensemble']['depths'])
    add(['## 11. Reproduction', '',
         'The scripts below are published unchanged in `firstletters/screen/` of https://github.com/AndreasHad04/villa-apple-silicon '
         '(each sets ROOT in one line; the folder layout and the Python environments they expect are in '
         '`firstletters/screen/README.md`). From ROOT:', '',
         '```bash',
         f'# axis and per-mesh orientation (writes results/fl/axis_{SCROLL}.json and the inventory)',
         f'python ops/fl/orient.py --scroll {SCROLL}',
         f'# render 66 layers (render66.py functions) and the four public models\' reading-side maps, per mesh',
         f'python ops/fl/screen_unit.py --scroll {SCROLL} --mesh <mesh> --models {four_k} --depths {dd}',
         f'# the Reader V3 map: screen_unit.py with Reader V3 registered at run time',
         f'python ops/fl/rv3_screen_unit.py --scroll {SCROLL} --mesh <mesh> --models {rv3.MODEL} --depths {dd}',
         f'# the five-model images of this atlas: ensemble.py with Reader V3 appended by rv3_with.py',
         f'python ops/fl/rv3_with.py fl_ensemble --scroll {SCROLL} --mesh <mesh> --models {all_k} --depths {dd} '
         f'--out-dir results/fl/images5/{SCROLL}/<mesh>',
         f'# a standalone CPU render of one mesh, any OS',
         f'python public/firstletters/render66.py --mesh <mesh>.tifxyz --volume {vol} --out <mesh>_66.zarr '
         f'--cache-dir chunks',
         f'# this release',
         f'python ops/fl/atlas_pack.py build && python ops/fl/atlas_pack.py readme && '
         f'python ops/fl/atlas_pack.py zip && python ops/fl/atlas_pack.py verify',
         '```', ''])
    text = '\n'.join(L)
    return re.sub(r'\n{3,}', '\n\n', text)


def cmd_readme(a):
    rows, kinds, dtag, files = layout()
    plan = plan_zips(rows, files)
    text = readme_text(rows, kinds, dtag, files, plan)
    bad = [i for i, x in enumerate(text.splitlines(), 1) if any(c in x for c in DASHES)]
    if bad:
        die(f'README would contain an en or em dash on lines {bad[:10]}')
    write_atomic(OUT / 'README.md', text)
    entries = write_manifest(rows, dtag, files)
    print(f'readme: README.md {len(text.encode())} bytes, {len(text.splitlines())} lines, confirmation '
          f'{"present" if CONFIRM.is_file() else "pending"}; {MANIFEST} rewritten with {len(entries)} entries')


def write_zip(name, members, est, cap):
    dst = REL / name
    want = [(arc, p.stat().st_size) for arc, p in members]
    if dst.is_file():
        try:
            with zipfile.ZipFile(dst) as zf:
                have = [(i.filename, i.file_size) for i in zf.infolist()]
        except zipfile.BadZipFile:
            have = None
        if have == want:
            print(f'{name}: already complete, {len(want)} members, {dst.stat().st_size / 1e6:.1f} MB, kept')
            return dict(zip=name, bytes=dst.stat().st_size, members=len(want))
    free = shutil.disk_usage(REL).free
    if free - est < FREE_FLOOR:
        die(f'REFUSED {name}: free {free / 2 ** 30:.2f} GiB, the zip needs about {est / 2 ** 30:.3f} GiB and would '
            f'leave {(free - est) / 2 ** 30:.2f} GiB, under the {FREE_FLOOR / 2 ** 30:.0f} GiB floor')
    tmp = REL / f'{name}.partial'
    t0 = time.time()
    try:
        with open(tmp, 'wb') as fh:
            with zipfile.ZipFile(fh, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
                for arc, p in members:
                    zf.write(p, arc, compress_type=zipfile.ZIP_STORED)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    size = dst.stat().st_size
    if cap and size > cap:
        die(f'{name} is {size} bytes, over the {cap} cap')
    print(f'{name}: {len(members)} members, {size / 1e6:.1f} MB (estimate {est / 1e6:.1f}), '
          f'{time.time() - t0:.1f} s, free after {shutil.disk_usage(REL).free / 2 ** 30:.2f} GiB')
    return dict(zip=name, bytes=size, members=len(members))


def cmd_zip(a):
    rows, kinds, dtag, files = layout()
    plan = plan_zips(rows, files)
    man = read_manifest()
    small = small_files(rows, dtag, True)
    want = {arc for arc, _ in files} | set(small)
    if set(man) != want:
        die(f'{MANIFEST} does not list exactly the release files ({len(set(man) ^ want)} differ): run build, readme')
    for rel in small:
        if sha256_file(OUT / rel) != man[rel]:
            die(f'{rel} changed after the manifest was written: run readme')
    cache = HashCache()
    stale = [arc for arc, p in files if cache.get(p) != man[arc]]
    cache.save()
    if stale:
        die(f'{len(stale)} image files differ from the manifest, e.g. {stale[:3]}: run build')
    for f in REL.glob('*.zip.partial'):
        f.unlink()
    try:
        record = json.loads(ZIPS_JSON.read_text())
    except (OSError, ValueError):
        record = {}
    if a.only in (None, 'images'):
        for z in plan:
            r = write_zip(z['zip'], [(f'{NAME}/{arc}', p) for arc, p in z['members']], z['est_bytes'], ZIP_CAP)
            record[z['zip']] = dict(r, windows=z['windows'], meshes=z['meshes'])
    if a.only in (None, 'meta'):
        mem = [(f'{NAME}/{rel}', OUT / rel) for rel in ['README.md', 'index.html', MANIFEST]
               + [x for x in small if x.startswith('thumbs/')]]
        est = sum(p.stat().st_size + member_overhead(arc) for arc, p in mem) + 22
        record[INDEX_ZIP] = dict(write_zip(INDEX_ZIP, mem, est, None), windows='all', meshes=[])
    write_atomic(ZIPS_JSON, json.dumps(record, indent=1))


def cmd_verify(a):
    t0 = time.time()
    errs = []
    rows, kinds, dtag, files = layout()
    plan = plan_zips(rows, files)
    man_text = (OUT / MANIFEST).read_text()
    man = read_manifest(man_text)
    root = f'{NAME}/'
    names = [z['zip'] for z in plan] + [INDEX_ZIP]
    on_disk = sorted(p.name for p in REL.glob(f'{NAME}_*.zip'))
    if sorted(names) != on_disk:
        errs.append(f'zips on disk {on_disk} differ from the plan {sorted(names)}')
    where, hashes, sizes = {}, {}, {}
    for zn in names:
        p = REL / zn
        if not p.is_file():
            errs.append(f'missing {zn}')
            continue
        sizes[zn] = p.stat().st_size
        if zn != INDEX_ZIP and sizes[zn] > ZIP_CAP:
            errs.append(f'{zn} is {sizes[zn]} bytes, over the cap')
        with zipfile.ZipFile(p) as zf:
            for i in zf.infolist():
                if i.filename in where:
                    errs.append(f'{i.filename} is in {where[i.filename]} and {zn}')
                if i.compress_type != zipfile.ZIP_STORED:
                    errs.append(f'{zn}:{i.filename} is compressed')
                where[i.filename] = zn
                with zf.open(i) as f:
                    hashes[i.filename] = sha256_stream(f)
    bad_root = [k for k in where if not k.startswith(root)]
    if bad_root:
        errs.append(f'{len(bad_root)} members outside {root}, e.g. {bad_root[:3]}')
    src_of = dict(files)
    src_of.update({rel: OUT / rel for rel in small_files(rows, dtag, True)})
    n_ok = 0
    for rel, h in man.items():
        arc = root + rel
        sp = src_of.get(rel)
        if arc not in hashes:
            errs.append(f'manifest entry in no zip: {rel}')
        elif hashes[arc] != h:
            errs.append(f'zip member differs from the manifest: {rel}')
        elif sp is None:
            errs.append(f'manifest entry is not a release file: {rel}')
        elif sha256_file(sp) != h:
            errs.append(f'source file differs from the manifest: {sp}')
        else:
            n_ok += 1
    missing_from_man = sorted(set(src_of) - set(man))
    if missing_from_man:
        errs.append(f'{len(missing_from_man)} release files missing from the manifest, e.g. {missing_from_man[:3]}')
    unlisted = [k for k in where if k[len(root):] not in man and k != root + MANIFEST]
    if unlisted:
        errs.append(f'{len(unlisted)} zip members not in the manifest, e.g. {unlisted[:3]}')
    if root + MANIFEST not in hashes:
        errs.append(f'{MANIFEST} is in no zip')
    elif hashes[root + MANIFEST] != hashlib.sha256(man_text.encode()).hexdigest():
        errs.append(f'the {MANIFEST} inside {INDEX_ZIP} differs from {OUT / MANIFEST}')
    mesh_zip, per_mesh = collections.defaultdict(set), collections.defaultdict(list)
    for k, zn in where.items():
        parts = k[len(root):].split('/')
        if parts[0] == 'images' and len(parts) > 2:
            mesh_zip[parts[1]].add(zn)
            per_mesh[parts[1]].append('/'.join(parts[2:]))
    want = [r['mesh'] for r in rows]
    missing = sorted(set(want) - set(mesh_zip))
    extra = sorted(set(mesh_zip) - set(want))
    multi = sorted(m for m, s in mesh_zip.items() if len(s) != 1)
    if missing or extra or multi:
        errs.append(f'meshes missing {missing[:5]}, unexpected {extra[:5]}, in more than one zip {multi[:5]}')
    per = len(rows[0]['files'])
    badcount = [r['mesh'] for r in rows if sorted(per_mesh.get(r['mesh'], [])) != sorted(r['files'])]
    if badcount:
        errs.append(f'{len(badcount)} meshes without exactly their {per} files, e.g. {badcount[:3]}')
    page = (OUT / 'index.html').read_text()
    links = [html.unescape(x) for x in re.findall(r'(?:href|src)="([^"]*)"', page)]
    anchors = [x for x in links if x.startswith('#')]
    files_l = [x for x in links if not x.startswith('#')]
    ids = set(re.findall(r'id="([^"]+)"', page))
    unresolved = [x for x in files_l if root + x not in where] + [x for x in anchors if x[1:] not in ids]
    if unresolved:
        errs.append(f'{len(unresolved)} index.html links do not resolve, e.g. {unresolved[:3]}')
    if re.search(r'https?://|<script|@import|@font-face|url\(', page):
        errs.append('index.html references something external (script, font, stylesheet or URL)')
    mesh_links = collections.Counter(x.split('/')[1] for x in files_l if x.startswith('images/'))
    nolink = [m for m in want if mesh_links.get(m, 0) == 0]
    if nolink:
        errs.append(f'{len(nolink)} meshes have no link in index.html')
    dash_hits = [f for f in ('README.md', 'index.html', MANIFEST) if any(c in (OUT / f).read_text() for c in DASHES)]
    if dash_hits:
        errs.append(f'en or em dash in {dash_hits}')
    for e in errs:
        print('ERROR', e)
    status = 'PASS' if not errs else 'FAIL'
    img = [s for z, s in sizes.items() if z != INDEX_ZIP]
    print(f'VERIFY {status}: {n_ok} of {len(man)} manifest entries match the zip members and the source files; '
          f'{len(set(want) & set(mesh_zip)) - len(multi)} of {len(want)} meshes present exactly once, '
          f'{len(want) - len(badcount)} with exactly {per} files; {len(files_l) - len([x for x in unresolved if not x.startswith("#")])} '
          f'of {len(files_l)} index.html file links and {len(anchors) - len([x for x in unresolved if x.startswith("#")])} '
          f'of {len(anchors)} anchors resolve; {len(sizes)} zips ({len(img)} image, largest '
          f'{max(img) / 1e6 if img else 0:.1f} MB, cap {ZIP_CAP / 1e6:.0f} MB); {len(errs)} errors; '
          f'{time.time() - t0:.1f} s')
    return 0 if not errs else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('build')
    sub.add_parser('readme')
    z = sub.add_parser('zip')
    z.add_argument('--only', choices=('images', 'meta'))
    sub.add_parser('verify')
    a = ap.parse_args()
    me = Path(__file__).resolve()
    print(f'atlas_pack {a.cmd} pid {os.getpid()} sha256 {sha256_file(me)[:16]} mtime {time.ctime(me.stat().st_mtime)}',
          flush=True)
    rc = {'build': cmd_build, 'readme': cmd_readme, 'zip': cmd_zip, 'verify': cmd_verify}[a.cmd](a)
    sys.exit(rc or 0)


if __name__ == '__main__':
    main()
