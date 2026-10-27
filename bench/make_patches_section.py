#!/usr/bin/env python3
"""Generate the PATCHES README section (axis-free sheet patches on First Letters scrolls) from measured JSON.

Working tree: `python ops/make_patches_section.py --summary` rebuilds results/patches/patches_summary.json from every
patch folder research/uncontested/patch*/ that holds a validation.json (a folder without one is not included yet; the
scroll name comes from the folder's build.json or run.json and from the axis file its validation.json names, which must
agree), plus the umbilicus scores, each scroll's axis, the open-data metadata, the registrations and ink-screen results
under results/fl/, the public reproduction record, results/patches/survey_sheetcheck.json and the saved prior-art survey
under research/prior_art/. A folder whose build.json lists `existing_keys` is an expansion of its scroll's one other
folder: its patches join that scroll's table and totals, and its own screen (the registration that names its
validation.json, the review whose contact sheets hold exactly its patches) is reported after the scroll's. A folder that
holds a WRITEUP_EXCLUDE file is left out. `--write` splices the section into public/README.md between the PATCHES markers
and copies the summary and this script into the published layout (results/patches/, bench/).
Published repo: section_text(summary) regenerates the section from the shipped summary alone, which is what
bench/verify_claims.py checks.
"""
import argparse, hashlib, json, os, re, shutil, sys
from pathlib import Path

BEGIN, END = '<!-- PATCHES:BEGIN -->', '<!-- PATCHES:END -->'
OFF_CENTRE = 10          # a brightest render layer more than this many layers from the centre layer 33 is off centre
KEY_STEP = 16            # slices between the rows (keys) of a patch
VOL_NAMES = {'ome-zarr': 'the CT', 'surface-prediction-zarr': 'the m7 surface prediction', 'normal-grids': 'its normal grids'}
NEIGHBOUR_RULE = 'pass = present at d+1 and at d0 or d+2, absent in the 180 degree negative'
AMENDMENT1_RULE = 'pass = present in READ, absent in BLIND'
SHEET_RULE = 'pass when good_fraction >= '
EXT_FOLLOWUP = 'a follow-up of known candidates, not independent'
EXT_BEFORE_INK = 'before any ink map'


def _num(name):
    return int(name.split('_')[0][2:])


def _load(p):
    return json.loads(p.read_text()) if p.exists() else None


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fmt_list(xs):
    xs = [str(x) for x in xs]
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' and ' + xs[-1]


def _pl(n, word, plural=None):
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def _scroll_name(root, v, b, run):
    names = {x['scroll'] for x in (b, run) if x and x.get('scroll')}
    m = re.search(r'axis_\w+\.json', v.get('axis', ''))
    ax = _load(root / 'results/fl' / m.group(0)) if m else None
    if ax and ax.get('scroll'):
        names.add(ax['scroll'])
    if len(names) != 1:
        raise SystemExit(f'scroll name unclear for a patch folder: {sorted(names)}')
    return names.pop()


def _method(pr):
    return dict(step_z=pr['STEP_Z'], ray=pr['RAY'], dmax=pr['DMAX'], frac_sp=pr['FRAC_SP'], wmax=pr['WMAX'],
                sp_min=pr['SP_MIN'], min_match=pr['MIN_MATCH'], k2k=pr['K2K_MAX'], vertex=pr['COL_EVERY'] * pr['DS'],
                key_step=KEY_STEP)


def _criteria(v):
    return dict(on_sheet=v['criteria']['on_sheet_vertices_min'], k2k=v['criteria']['key_second_difference_max_max'],
                vertical_ratio=v['criteria']['vertical_on_track_vs_control'], outward=v['criteria']['outward_fraction_min'],
                box=int(re.search(r'Chebyshev distance (\d+)', v['on_sheet_rule']).group(1)))


def _selection(v, b, rows):
    e = v['extension']
    area = lambda names: round(sum(r['area'] for r in rows if r['name'] in names), 3)
    if 'passed_not_written_cap' in e:
        pr = b['params']
        return dict(kind='first_plus_extension', first_key=pr['Z0'], n_first=len(b['selected']),
                    area_first=area(b['selected']), sep=pr['SEP_DEG_USED'], ext_keys=e['keys'],
                    validated=e['validated'], passed=e['passed'], duplicates=len(e['refused_duplicate_of_new']),
                    n_ext=len(e['accepted']), n_sector=len(e['sector_rule']['added']),
                    n_cap=len(e['accepted']) - len(e['sector_rule']['added']),
                    sector=[int(float(x)) for x in re.search(r'at (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?) degrees',
                                                             e['sector_rule']['rule']).groups()],
                    not_written=len(e['passed_not_written_cap']), area_ext=e['new_area_cm2'])
    pk = e['per_key']
    return dict(kind='pooled_keys', keys=e['keys'], built=sum(k['n_selected'] for k in pk.values()),
                passed=sum(k['n_pass'] for k in pk.values()), duplicates=len(e['refused_duplicate']),
                failed=len(e['failed_gates']), kept=len(e['accepted']),
                sep=sorted({k['separation_deg'] for k in pk.values()}), duplicate_rule=e['duplicate_rule'])


def _ink(root, scroll):
    suf = scroll[len('PHerc'):] if scroll.startswith('PHerc') else scroll
    fl = root / 'results/fl'
    out = dict(prereg=None, review=None, neighbour=None, confirm=None)
    pre = fl / f'PREREGISTRATION_FL_{suf}.md'
    if pre.exists():
        t, sha = pre.read_text(), _sha(pre)
        m = re.search(r'^## Amendment 1\b.*?^- The rule for these windows: (.*?)(?=^- |^## |\Z)', t, re.M | re.S)
        am = ' '.join(m.group(1).split()) if m else None
        if am and not am.startswith(AMENDMENT1_RULE):
            raise SystemExit(f'{pre.name}: the Amendment 1 rule is not the one section_text paraphrases; update the sentence')
        rec = fl / 'prereg.sha256'
        out['prereg'] = dict(file=str(pre.relative_to(root)), sha256=sha, recorded=rec.exists() and sha in rec.read_text(),
                             amendment1_rule=am)
    u = _load(fl / f'review_{suf}' / 'UNSEALED.json')
    if u:
        by = {}
        for f in u['flagged']:
            if not f['control']:
                by.setdefault(f['mesh'], []).append(f['window'])
        out['review'] = dict(valid=u['valid'], rule=u['rule'], n_control_windows_with_letters=u['n_control_windows_with_letters'],
                             n_flagged_control_on_letters=u['n_flagged_control_on_letters'],
                             n_flagged_screen=u['n_flagged_screen'], n_regions=len(by),
                             regions=[[k, sorted(w)] for k, w in by.items()])
    nb = _load(fl / f'neighbour_{suf}' / 'RESULT.json')
    if nb:
        res = nb['results']
        out['neighbour'] = dict(valid=nb['valid'], rule=nb['rule'], n_screen=sum(not r['control'] for r in res),
                                n_pass_screen=nb['n_pass_screen'], n_control=nb['n_control'],
                                control_passed=sum(bool(r['passed']) for r in res if r['control']),
                                depths=sorted({int(k[1:]) for r in res for k in r['marks_by_depth'] if k.startswith('d')}))
    cdir = fl / f'confirm_{suf}'
    cf = _load(cdir / 'RESULT.json')
    if cf:
        res = cf['results']
        out['confirm'] = dict(state='done', valid=cf['valid'], rule=cf['rule'], n_screen=sum(not r['control'] for r in res),
                              n_pass_screen=cf['n_pass_screen'], n_control=cf['n_control'],
                              control_passed=sum(bool(r['passed']) for r in res if r['control']))
    elif cdir.exists():
        out['confirm'] = dict(state='running')
    fc = _load(cdir / 'facecheck.json')
    if fc:
        pair = '/'.join(fc['models'])
        rows = []
        for c in fc['crops']:
            a, b = c['own_valid_mask']['pairs'][pair], c['pr_sheet_check']['pairs'][pair]
            rows.append(dict(scroll=c['scroll'], mesh=c['mesh'], control=c['scroll'] != scroll,
                             reading=a['best_face'] == fc['reading_side'], asym=a['face_asymmetry'],
                             consistency=a['face_consistency'], same_side_both_masks=a['best_face'] == b['best_face']))
        out['facecheck'] = dict(models=fc['models'], rows=rows)
    return out


def _rows(v, seed0, sh):
    rows = []
    for n in sorted(v['patches'], key=_num):
        p, vert, s = v['patches'][n], v['patches'][n]['vertical'], sh.get(n)
        rows.append(dict(name=n, seed_key=p.get('seed_key', seed0), z=p['keys'], angle=p['angle_deg'], area=p['area_cm2'],
                         on_sheet=p['m7']['on_sheet_vertices'], second_diff_max=p['keys_check']['second_diff_max'],
                         vertical=vert['on_track_of_covered_all'],
                         control=max(x or 0 for x in vert['control_half_spacing_shift_on_track'].values()),
                         outward=p['orientation']['outward_fraction_chosen'], switches=vert['switch_events'],
                         lines=vert['lines'], peak=(p.get('render') or {}).get('peak_layer'), passed=p['pass'],
                         sheet=None if s is None else dict(good_fraction=s['good_fraction'], tiles=s['tiles'],
                                                           passes=s['passes'])))
    return rows


def _is_expansion(b):
    """a folder traced around the existing patches of its scroll: its build record lists their seed keys"""
    return bool((b or {}).get('existing_keys'))


def _ink_expansion(root, d, v):
    """the screen of an expansion folder: the registration that names its validation.json, and the review whose contact
    sheets were built from exactly its patches (its key is read only once UNSEALED.json exists)"""
    fl, folder, names = root / 'results/fl', str(d.relative_to(root)), set(v['patches'])
    pres = [p for p in sorted(fl.glob('PREREGISTRATION_FL_*.md')) if f'{folder}/validation.json' in p.read_text()]
    revs = [q.parent for q in sorted(fl.glob('review_*/build.json'))
            if {m for m in (_load(q).get('meshes') or []) if isinstance(m, str)} == names]
    if len(pres) > 1 or len(revs) > 1:
        raise SystemExit(f'{d.name}: {len(pres)} registrations and {len(revs)} reviews match it; at most one of each')
    out = dict(prereg=None, review=None)
    if pres:
        t, sha, rec = ' '.join(pres[0].read_text().split()), _sha(pres[0]), fl / 'prereg.sha256'
        if EXT_FOLLOWUP not in t or EXT_BEFORE_INK not in t:
            raise SystemExit(f'{pres[0].name}: it no longer says the screen was registered before any ink map as a follow-up '
                             'of known candidates; update the sentences that paraphrase it')
        out['prereg'] = dict(file=str(pres[0].relative_to(root)), sha256=sha, recorded=rec.exists() and sha in rec.read_text())
    u = _load(revs[0] / 'UNSEALED.json') if revs else None
    if u:
        win = {}
        for s in json.loads((revs[0] / 'key.json').read_text())['sheets'].values():
            if not s['control']:
                win[s['mesh']] = win.get(s['mesh'], 0) + len(s['windows'])
        if not set(win) <= names:
            raise SystemExit(f'{revs[0].name}: its key holds windows of patches outside {d.name}')
        by = {}
        for f in u['flagged']:
            if not f['control']:
                by.setdefault(f['mesh'], []).append(f['window'])
        n_with = sum(1 for k in win.values() if k)
        out['review'] = dict(folder=str(revs[0].relative_to(root)), valid=u['valid'], rule=u['rule'],
                             n_control_windows_with_letters=u['n_control_windows_with_letters'],
                             n_flagged_control_on_letters=u['n_flagged_control_on_letters'],
                             n_flagged_screen=u['n_flagged_screen'], n_regions=len(by),
                             regions=[[k, sorted(w)] for k, w in by.items()], n_windows_screen=sum(win.values()),
                             n_patches=len(names), n_patches_with_windows=n_with, n_patches_without_windows=len(names) - n_with)
    return out


def _expansion(root, d, name, sh, v0, ink0, md):
    """one expansion folder of a scroll: (what the write-up says about it, its table rows, its build parameters)"""
    v = json.loads((d / 'validation.json').read_text())
    b = _load(d / 'build.json') or {}
    run = _load(d / 'run.json')
    if _scroll_name(root, v, b, run) != name:
        raise SystemExit(f'{d.name}: not a folder of {name}')
    rows = _rows(v, None, sh)
    e, first = v['extension'], set(v0['patches'])
    if set(e['accepted']) != set(v['patches']) or first & set(v['patches']):
        raise SystemExit(f'{d.name}: its accepted list is not its patches, or it reuses a patch name of {name}')
    dup, reb = e['refused_duplicate'], e['duplicate_control']
    parts = [len(e[k]) for k in ('accepted', 'refused_cap', 'refused_duplicate', 'failed_gates', 'failed_bars',
                                 'failed_build', 'failed_validate')]
    listed = sum(k['n_in_sector_geometry_ok'] for k in e['per_key'].values())
    built = len(b['candidates_built'])
    if not sum(parts) == built == listed:
        raise SystemExit(f'{d.name}: {parts} do not add up to the {built} candidates built ({listed} listed)')
    regions = sorted(e['per_region'])
    flagged = sorted(m for m, _ in ((ink0 or {}).get('review') or {}).get('regions') or [])
    if regions != flagged:
        raise SystemExit(f'{d.name}: its regions {regions} are not the regions the {name} screen flagged {flagged}')
    prefix = {n.split('_')[0].rstrip('0123456789') for n in v['patches']}
    area = round(sum(r['area'] for r in rows), 3)
    if len(prefix) != 1 or area != e['area_cm2_total'] or v['n_pass'] != len(rows):
        raise SystemExit(f'{d.name}: patch names {sorted(prefix)}, area {area} against {e["area_cm2_total"]}, '
                         f'{v["n_pass"]} of {len(rows)} passing')
    sel = dict(kind='sector_expansion', keys=e['keys'], n_keys_searched=len(e['keys']), sector=e['sector_deg'],
               built=built, failed_bars=len(e['failed_bars']), failed_validation=len(e['failed_gates']),
               errors=len(e['failed_build']) + len(e['failed_validate']), duplicates=len(dup),
               duplicates_of_first=sum(x['duplicate_of'] in first for x in dup.values()), n_first=len(first),
               rebuilt_first=len(reb), rebuilt_identical_refused=sum(bool(x['identical']) and x['status'] == 'refused_duplicate'
                                                                     and x['rebuilt'] in dup for x in reb),
               passed_not_duplicate=len(e['accepted']) + len(e['refused_cap']), kept=len(e['accepted']), cap=e['cap'],
               per_region=[[k, e['per_region'][k]] for k in regions], gates=e['gates'], duplicate_rule=e['duplicate_rule'],
               cap_rule=e['cap_rule'])
    vol = next(iter(md['volumes'].values()))
    cli = dict(band=(run or {}).get('band_arg'), voxel_um=vol['properties']['pixel_size_um'])
    out = dict(folder=str(d.relative_to(root)), prefix=prefix.pop(), n=len(rows), n_pass=v['n_pass'], area=area, cli=cli,
               n_sheet_unscored=sum(r['sheet'] is None for r in rows), selection=sel, ink=_ink_expansion(root, d, v),
               script_sha256=(run or {}).get('sheet_patches_sha256'))
    return out, rows, [r['params'] for r in (b.get('per_key_listing_records') or {}).values()], _criteria(v)


def _scroll(root, d, md_all, sheets, ext_dirs=()):
    v = json.loads((d / 'validation.json').read_text())
    b = _load(d / 'build.json') or {}
    run = _load(d / 'run.json')
    name = _scroll_name(root, v, b, run)
    params = [b['params']] if 'params' in b else [r['params'] for r in (b.get('per_key_records') or {}).values()]
    seed0 = b['params']['Z0'] if 'params' in b else None
    sc = ((sheets or {}).get('scrolls') or {}).get(name)
    sh = {x['patch']: x for x in sc['patches']} if sc else {}
    first_rows = _rows(v, seed0, sh)
    ink = _ink(root, name)
    exps = [_expansion(root, e, name, sh, v, ink, md_all[name]) for e in ext_dirs]
    params += [p for x in exps for p in x[2]]
    rows = first_rows + [r for x in exps for r in x[1]]
    crit = _criteria(v)
    md = md_all[name]
    vol_types = sorted({x['type'] for vol in md['volumes'].values() for x in vol.get('data') or []})
    types = sorted(set(vol_types) | {x['type'] for x in md['sample'].get('data') or []})
    ax = json.loads((root / 'results/fl' / f'axis_{name}.json').read_text())
    top = sorted(rows, key=lambda r: (-r['switches'], _num(r['name'])))[:4]
    limits = dict(top_switches=[[r['name'], r['switches'], r['lines']] for r in top], n_top=len(top),
                  off_centre=[[r['name'], r['peak']] for r in rows if r['peak'] is not None and abs(r['peak'] - 33) > OFF_CENTRE],
                  at_off_centre=[[r['name'], r['peak']] for r in rows
                                 if r['peak'] is not None and abs(r['peak'] - 33) == OFF_CENTRE],
                  at_bar=[[r['name'], r['second_diff_max']] for r in rows if r['second_diff_max'] >= crit['k2k']],
                  not_rendered=[r['name'] for r in rows if r['peak'] is None],
                  axis_hull_median=ax['check']['distance_median'], axis_hull_max=ax['check']['distance_max'])
    sheet = None
    if sc:
        fails = [r for r in rows if r['sheet'] and not r['sheet']['passes']]
        topn = {r['name'] for r in top}
        sheet = dict(n=sc['n'], n_pass=sc['n_pass'], n_scored=sum(r['sheet'] is not None for r in rows),
                     failing=[r['name'] for r in fails], n_failing=len(fails),
                     few_tiles=[[r['name'], r['sheet']['tiles'], r['sheet']['passes']] for r in rows
                                if r['sheet'] and r['sheet']['tiles'] <= 1],
                     top_switch_passes=[[r['name'], (r['sheet'] or {}).get('passes')] for r in top],
                     n_failing_outside_top=sum(r['name'] not in topn for r in fails),
                     max_switches_failing_outside_top=max([r['switches'] for r in fails if r['name'] not in topn],
                                                          default=None))
    sel = _selection(v, b, first_rows)
    cli = None
    if sel['kind'] == 'pooled_keys' and run and b.get('per_key_records'):
        vol = next(iter(md['volumes'].values()))
        cli = dict(band=[run['band'][0], run['band'][1] - KEY_STEP, KEY_STEP], voxel_um=vol['properties']['pixel_size_um'],
                   first_index={k: min(_num(n) for n in r['selected']) for k, r in b['per_key_records'].items()})
    rp = _load(d / 'public_repro.json')
    repro = None
    if rp:
        vr = (rp.get('validate_5760') or {}).get('rows') or {}
        repro = dict(script_sha256=rp['public_sha256'], grids=sorted(rp['grids'], key=_num), n_grids=len(rp['grids']),
                     n_validated=len(vr),
                     grids_byte_identical=rp['grids_all_bytes_identical'],
                     validated=sorted(vr, key=_num), validation_fields=sum(x['fields_compared'] for x in vr.values()),
                     validation_differing=sum(len(x['differing']) for x in vr.values()),
                     axis_points=(rp.get('axis') or {}).get('n_points'),
                     axis_identical=bool((rp.get('axis') or {}).get('control_points_identical')) and
                     bool((rp.get('axis') or {}).get('sections_identical')))
    keys_used = sorted({r['seed_key'] for r in rows if r['seed_key'] is not None})
    out = dict(
        scroll=name, folder=str(d.relative_to(root)),
        missing=dict(segments=len(md.get('segments', {})), data_types=types, volume_types=vol_types,
                     umbilicus=any('umbilicus' in t for t in types), lasagna=any('lasagna' in t for t in types)),
        rows=rows, selection=sel, cli=cli, limits=limits, sheet_check=sheet, ink=ink, repro=repro,
        script_sha256=(run or {}).get('sheet_patches_sha256') or (repro or {}).get('script_sha256'),
        totals=dict(n=len(rows), n_pass=v['n_pass'] + sum(x[0]['n_pass'] for x in exps), area=round(sum(r['area'] for r in rows), 3),
                    largest=max(r['area'] for r in rows), z_min=min(r['z'][0] for r in rows),
                    z_max=max(r['z'][1] for r in rows), n_keys=len(keys_used),
                    on_sheet_min=min(r['on_sheet'] for r in rows), outward_min=min(r['outward'] for r in rows),
                    vertical_over_control_min=round(min(r['vertical'] / max(r['control'], 1e-9) for r in rows), 1)))
    if exps:
        out['expansions'] = [x[0] for x in exps]
    return out, [_method(x) for x in params], [crit] + [x[3] for x in exps]


def _prior(root, names):
    d = root / 'research/prior_art'
    sv = sorted(d.glob('*survey_README*.md')) if d.exists() else []
    if not sv:
        return None
    t = sv[-1].read_text()
    lines = t.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('| scroll |') and 'survey patches traced' in l)
    col = {h.strip(): k for k, h in enumerate(lines[i].split('|')[1:-1])}
    table = {}
    for l in lines[i + 2:]:
        if not l.startswith('|'):
            break
        c = [x.strip() for x in l.split('|')[1:-1]]
        table[c[0]] = c
    per = {}
    for n in names:
        m = re.fullmatch(r'PHerc(\d+)(\w*)', n)
        label = f'PHerc. {int(m.group(1))}{m.group(2)}' if m else n
        c = table.get(label)
        per[n] = None if c is None else dict(label=label, traced=int(c[col['survey patches traced']]),
                                             passed=int(c[col['pass sheet check']]), inked=int(c[col['inked']]))
    body = sorted(d.glob('villa_*_body_*.md'))
    gh = re.search(r'githubusercontent\.com/([\w-]+)/([\w.-]+?)/', body[-1].read_text()) if body else None
    return dict(source=str(sv[-1].relative_to(root)),
                n_scrolls=int(re.search(r'Survey of the (\d+) First Letters scrolls', t).group(1)),
                author=gh.group(1) if gh else None, repo=gh.group(2) if gh else None,
                url=f'https://github.com/{gh.group(1)}/{gh.group(2)}' if gh else None,
                pr=int(re.search(r'villa_(\d+)_body', body[-1].name).group(1)) if body else None, per_scroll=per)


def build_summary(root: Path) -> dict:
    p2 = root / 'research/uncontested'
    md_all = json.loads((p2 / 'dl/metadata.json').read_text())['samples']
    sheets = _load(root / 'results/patches/survey_sheetcheck.json')
    groups = {}
    for d in sorted(p2.glob('patch*')):
        if d.is_dir() and (d / 'validation.json').exists() and not (d / 'WRITEUP_EXCLUDE').exists():
            v, b = json.loads((d / 'validation.json').read_text()), _load(d / 'build.json') or {}
            groups.setdefault(_scroll_name(root, v, b, _load(d / 'run.json')), []).append((d, _is_expansion(b)))
    if not groups:
        raise SystemExit('no patch folder with a validation.json')
    built = []
    for name, ds in groups.items():
        first = [d for d, x in ds if not x]
        if len(first) != 1:
            raise SystemExit(f'{name}: {len(first)} patch folders that are not an expansion; exactly one is needed')
        built.append(_scroll(root, first[0], md_all, sheets, [d for d, x in ds if x]))
    built.sort(key=lambda x: (-x[0]['totals']['area'], x[0]['scroll']))
    scrolls = [x[0] for x in built]
    methods = [m for x in built for m in x[1]]
    crits = [c for x in built for c in x[2]]
    bar = re.search(r'Bar: median <= (\d+) and max <= (\d+) level-0 voxels', (p2 / 'PHASE2_UMBILICUS.md').read_text())
    umb = {}
    for k in ('v1', 'v2'):
        c = json.loads((p2 / 'phase2' / f'score_{k}.json').read_text())['checks_curve']
        umb[k] = dict(planes=c['planes'], scrolls=c['scrolls'], median=c['median_l0'], max=c['max_l0'], passed=c['PASS'],
                      per_scroll=sorted(c['per_scroll']))
    pub_sha = _sha(root / 'public/uncontested/sheet_patches.py')
    rows = [r for s in scrolls for r in s['rows']]
    scored = [s['sheet_check'] for s in scrolls if s['sheet_check']]
    return dict(
        scrolls=scrolls,
        totals=dict(n_scrolls=len(scrolls), n=len(rows), n_pass=sum(s['totals']['n_pass'] for s in scrolls),
                    area=round(sum(r['area'] for r in rows), 3), on_sheet_min=min(r['on_sheet'] for r in rows),
                    outward_min=min(r['outward'] for r in rows),
                    vertical_over_control_min=min(s['totals']['vertical_over_control_min'] for s in scrolls),
                    sheet_scored=sum(x['n_scored'] for x in scored), sheet_pass=sum(x['n_pass'] for x in scored),
                    sheet_fail=sum(len(x['failing']) for x in scored)),
        umbilicus=dict(bar_median=int(bar.group(1)), bar_max=int(bar.group(2)), **umb),
        params=methods[0], params_same=all(m == methods[0] for m in methods),
        criteria=crits[0], criteria_same=all(c == crits[0] for c in crits),
        script=dict(sha256=pub_sha, same_for_all=all(s['script_sha256'] in (pub_sha, None) for s in scrolls) and
                    all(e['script_sha256'] in (pub_sha, None) for s in scrolls for e in s.get('expansions') or [])),
        survey_sheet_check=None if sheets is None else dict(
            rule=sheets['rule'], layer=sheets['layer'], inkcheck_sha256=sheets['inkcheck_sha256'],
            positive_control=sheets.get('positive_control'), negative_control=sheets.get('negative_control')),
        prior_art=_prior(root, [s['scroll'] for s in scrolls]),
    )


def _row(r):
    sc = r['sheet']
    sheet = 'not scored' if sc is None else \
        f"{sc['good_fraction']:g} ({sc['tiles']} tiles), {'pass' if sc['passes'] else 'fail'}"
    return (f"| {r['name']} | {r['seed_key']} | {r['z'][0]} to {r['z'][1]} | {r['angle']:.1f} | {r['area']:.3f} | "
            f"{r['on_sheet']:.4f} | {r['second_diff_max']:.2f} | {r['vertical']:.4f} | {r['control']:.4f} | "
            f"{r['outward']:.4f} | {r['switches']} / {r['lines']} | "
            f"{r['peak'] if r['peak'] is not None else 'not rendered'} | {sheet} |")


def _scroll_row(x):
    t, sc = x['totals'], x['sheet_check']
    return (f"| {x['scroll']} | {t['n']} | {t['n_pass']} | {t['area']:.3f} | {t['largest']:.3f} | {t['z_min']} to "
            f"{t['z_max']} | {t['n_keys']} | {'not scored' if sc is None else str(sc['n_pass']) + ' of ' + str(sc['n_scored'])} |")


def _script_text(s):
    sc, sha = s['scrolls'], s['script']['sha256']
    runs = [(x['scroll'], x['script_sha256'], x['repro']) for x in sc] + \
        [(f"the {x['scroll']} expansion", e['script_sha256'], None) for x in sc for e in x.get('expansions') or []]
    diff = [(n, h) for n, h, _ in runs if h and h != sha]
    if diff:
        return ('Not every run used the shipped `sheet_patches.py`: ' +
                '; '.join(f"{n} sha256 `{h[:16]}`" for n, h in diff) + '.')
    made = [n for n, h, r in runs if r is None and h]
    rep = [x for x in sc if x['repro'] and x['repro']['grids_byte_identical']]
    unrec = [n for n, h, _ in runs if not h]
    parts = ([f"it traced {_fmt_list(made)} as shipped"] if made else []) + \
        [f"it rebuilt {len(x['repro']['grids'])} of the {x['scroll']} patches byte for byte from the same inputs" for x in rep]
    who = 'every scroll' if not unrec else _fmt_list(made + [x['scroll'] for x in rep])
    t = f"The same file (sha256 `{sha[:16]}`) served {who} unchanged: {'; '.join(parts)}."
    return t + (f" The run record of {_fmt_list(unrec)} does not give the script's hash yet." if unrec else '')


def _related_text(s):
    pa, ssc, sc = s['prior_art'], s['survey_sheet_check'], s['scrolls']
    if pa is None:
        return ''
    cov = [(x['scroll'], pa['per_scroll'].get(x['scroll'])) for x in sc]
    counts = '; '.join(f"{c['label']} {c['traced']} traced and {c['passed']} passing its sheet check" for _, c in cov if c)
    t = (f"**Related work.** {pa['author']} published a survey of all {pa['n_scrolls']} First Letters scrolls "
         f"([{pa['repo']}]({pa['url']})) and opened villa PR #{pa['pr']} from it. It traces patches with VC3D's "
         f"`vc_grow_seg_from_seed` and the remote normal grids, and scores each with its own sheet check, which flags parts "
         f"of a patch that cut across windings. Its counts for the scrolls covered here: {counts}.")
    differ = [(n, c) for n, c in cov if c and (c['passed'] == 0) != (next(x for x in sc if x['scroll'] == n)['totals']['n_pass'] == 0)]
    for n, c in differ:
        x = next(y for y in sc if y['scroll'] == n)
        t += (f" The survey and this work differ on {n}: none of its {c['traced']} traces passed its sheet check, while "
              f"the tracer here produced {x['totals']['n_pass']} patches that pass the gates above.")
    scored = [x for x in sc if x['sheet_check']]
    if ssc and scored:
        m = re.match(r'pass when good_fraction >= ([\d.]+) \((\d+) px tiles, median crossness >= ([\d.]+)\)', ssc['rule'])
        rule = (f"at least {m.group(1)} of the {m.group(2)} px tiles with a median crossness of at least {m.group(3)}"
                if m and ssc['rule'].startswith(SHEET_RULE) else f"`{ssc['rule']}`")
        pc = ssc['positive_control']
        t += (f" Its sheet check, run here with its own `inkcheck.py` unchanged and its published pass rule ({rule}) on "
              f"the middle layer of every render, passes " +
              _fmt_list([f"{x['sheet_check']['n_pass']} of the {x['sheet_check']['n_scored']} {x['scroll']} patches"
                         for x in scored]) + '.')
        uns = [(x['scroll'], e) for x in scored for e in x.get('expansions') or [] if e['n_sheet_unscored']]
        if uns:
            t += (' It has not been run on ' +
                  _fmt_list([f"the {e['n']} {n} expansion patches" if e['n_sheet_unscored'] == e['n'] else
                             f"{e['n_sheet_unscored']} of the {e['n']} {n} expansion patches" for n, e in uns]) + '.')
        if pc:
            t += (f" On the positive control, a PHerc0343 crop over known letters, it gives a good fraction of "
                  f"{pc['good_fraction']:g} over {pc['tiles']} tiles and {'passes' if pc['passes'] else 'FAILS'}.")
        t += ' No negative control was run.' if not ssc.get('negative_control') else ''
        fail = [x for x in scored if x['sheet_check']['failing']]
        if fail:
            t += (" The two checks disagree: the gates here pass every patch, while its sheet check fails " +
                  _fmt_list([f"{len(x['sheet_check']['failing'])} on {x['scroll']}" for x in fail]) + '.')
        for x in scored:
            q = x['sheet_check']
            if len(q['failing']) < 2:
                continue
            ok = [n for n, p in q['top_switch_passes'] if p]
            bad = [n for n, p in q['top_switch_passes'] if p is False]
            same = not ok and not q['n_failing_outside_top']
            t += (f" On {x['scroll']} its failures and the switch-event counter here "
                  f"{'single out the same patches' if same else 'do not single out the same patches'}: of the "
                  f"{len(q['top_switch_passes'])} with the most switch events, " +
                  ' while '.join(([f"{_fmt_list(ok)} pass{'es' if len(ok) == 1 else ''} its check"] if ok else []) +
                            ([f"{_fmt_list(bad)} fail{'s' if len(bad) == 1 else ''} it"] if bad else [])))
            if q['n_failing_outside_top']:
                t += (f", and the other {q['n_failing_outside_top']} failing patches have at most "
                      f"{q['max_switches_failing_outside_top']} switch events each")
            t += '.' if same else (f". The two criteria measure different things, and which is right on {x['scroll']} "
                                   "is not settled.")
        few = [x for x in scored if x['sheet_check']['few_tiles']]
        if few:
            t += (" Its rule degenerates on patches with 0 or 1 scoreable tiles, where one tile or none decides: " +
                  '; '.join(f"on {x['scroll']} " + _fmt_list([f"{n} ({k} {'tile' if k == 1 else 'tiles'}, "
                                                             f"{'pass' if ok else 'fail'})"
                                                             for n, k, ok in x['sheet_check']['few_tiles']])
                            for x in few) + '.')
    return t + (" Its face check, which asks whether the ink models agree on one face of the sheet more than on the "
                "other (forward against reverse maps), is close in spirit to the blind-side stage below.")


def _missing_text(sc):
    groups = {}
    for x in sc:
        m = x['missing']
        groups.setdefault((m['segments'], m['umbilicus'], m['lasagna'], tuple(m['volume_types'])), []).append(x['scroll'])
    out = []
    for (seg, umb, las, vt), ns in groups.items():
        vols = _fmt_list([VOL_NAMES.get(v, v) for v in sorted(vt, key=lambda v: (list(VOL_NAMES) + [v]).index(v))])
        out.append(f"for {_fmt_list(ns)} it lists {seg or 'no'} segments, {'an' if umb else 'no'} umbilicus and "
                   f"{'' if las else 'no '}lasagna data, and the volume comes with {vols} only")
    return '; '.join(out)


def _selection_text(x):
    return ' '.join([_selection_first(x)] + [_expansion_selection_text(e) for e in x.get('expansions') or []])


def _expansion_selection_text(e):
    q = e['selection']
    regs = _fmt_list([f"{k} nearer {m}" for m, k in q['per_region']])
    t = (f"The last {e['n']} ({e['area']:.3f} cm2, the {e['prefix']} patches) were traced around the candidate regions of "
         f"the ink screen below, {regs}: every seed candidate whose geometry the tracer accepts at the "
         f"{q['n_keys_searched']} keys from z {min(q['keys'])} to {max(q['keys'])} with an angle of {q['sector'][0]:g} to "
         f"{q['sector'][1]:g} degrees, {q['built']} in all; {q['failed_bars']} failed the tracer's own bars, "
         f"{q['failed_validation']} failed validation, " + (f"{q['errors']} did not run, " if q['errors'] else '') +
         f"{q['duplicates']} were refused as duplicates ({q['duplicates_of_first']} of them of the {q['n_first']} earlier "
         "patches")
    if q['rebuilt_first']:
        t += (f"; the {q['rebuilt_first']} candidates that rebuild an earlier patch came out identical to it and are among "
              "them" if q['rebuilt_identical_refused'] == q['rebuilt_first'] else
              f"; {q['rebuilt_identical_refused']} of the {q['rebuilt_first']} candidates that rebuild an earlier patch came "
              "out identical to it and are among them")
    if q['passed_not_duplicate'] > q['kept']:
        return t + f"), and the {q['kept']} largest of the other {q['passed_not_duplicate']} were kept, a cap of {q['cap']}."
    return t + f"), and the other {q['kept']} were kept."


def _selection_first(x):
    q = x['selection']
    if q['kind'] == 'first_plus_extension':
        return (f"The first {q['n_first']} ({q['area_first']:.3f} cm2) come from seeds at z {q['first_key']}, chosen at "
                f"least {q['sep']:g} degrees apart. The other {q['n_ext']} ({q['area_ext']:.3f} cm2) come from seeds at z "
                f"{_fmt_list(q['ext_keys'])}: {q['validated']} candidates validated, {q['passed']} passed, "
                f"{q['duplicates']} refused as duplicates of a patch already accepted, {q['n_cap']} written largest first "
                f"and {q['n_sector']} more from the {q['sector'][0]} to {q['sector'][1]} degree sector, which had none; "
                f"{q['not_written']} passing candidates were not written.")
    failed = f", {q['failed']} failed" if q['failed'] else ''
    return (f"They come from seeds at z {_fmt_list(q['keys'])}, the seeds of one key at least "
            f"{' or '.join(f'{v:g}' for v in q['sep'])} degrees apart: one `build` and `validate` per key wrote "
            f"{q['built']} patches and {q['passed']} passed every gate{failed}; pooled across keys, largest first, "
            f"{q['duplicates']} {'was' if q['duplicates'] == 1 else 'were'} refused as duplicates of a larger patch, so "
            f"{q['kept']} were kept.")


def _depths(ds):
    return _fmt_list(['0' if d == 0 else f'+{d}' for d in ds])


def _expansion_ink_lines(x):
    out = []
    for e in x.get('expansions') or []:
        n, k = x['scroll'], e['ink']
        pre, rv = k['prereg'], k['review']
        line = (f"- {n} expansion, the {e['n']} {e['prefix']} patches" +
                (f" (registration sha256 `{pre['sha256'][:16]}`)" if pre else '') + ': ')
        if pre:
            line += (f"registered before any ink map of them existed, as a follow-up of the {n} candidates above and not as an "
                     "independent test. ")
        if rv is None:
            out.append(line + ('The review' if pre else 'Not registered yet; the review') + ' has not run yet.')
            continue
        a, b = rv['n_flagged_control_on_letters'], rv['n_control_windows_with_letters']
        line += f"The review is {'VALID' if rv['valid'] else 'VOID'} ({a} of the {b} control windows on known letters flagged)"
        if not rv['valid']:
            out.append(line + ', so it counts for nothing under its rule.')
            continue
        cover = (f"{rv['n_windows_screen']} windows from {rv['n_patches_with_windows']} of the {rv['n_patches']} patches" +
                 (f" (the other {rv['n_patches_without_windows']} yielded no window)" if rv['n_patches_without_windows'] else ''))
        if rv['n_flagged_screen'] == 0:
            line += (f" and flagged none of the {cover}: a null." +
                     (f" With only {a} of the {b} control windows on known letters flagged, its sensitivity is low, so the "
                      "null is weak evidence against text continuing there, not proof." if 2 * a < b else
                      " The null is evidence against text continuing there, not proof."))
        else:
            regs = '; '.join(f"{m} {'windows' if len(w) > 1 else 'window'} {_fmt_list(w)}" for m, w in rv['regions'])
            line += (f" and flagged {_pl(rv['n_flagged_screen'], 'window')} of the {cover}, in "
                     f"{_pl(rv['n_regions'], 'region')} ({regs}); the later checks have not run on them yet.")
        reg = [m.split('_')[0] for m, _ in e['selection']['per_region']]
        out.append(line + f" The results for {_fmt_list(reg)} above are unchanged.")
    return out


def _ink_text(s):
    sc = s['scrolls']
    reg = [x['scroll'] for x in sc if x['ink']['prereg']]
    t = ("**Ink screen.** Five-model ink images of every patch at d = +1 were reviewed blind, with known-letter windows "
         "from PHerc0343 shuffled in unmarked; a review counts as VALID only if a flagged control window overlaps the "
         "known letters.")
    if reg:
        t += (" Each scroll's screen was registered before any ink map of its patches existed." if len(reg) == len(sc) else
              f" The screens of {_fmt_list(reg)} were registered before any ink map of their patches existed.")
    out = [t, '']
    miss = False
    for x in sc:
        k, n = x['ink'], x['scroll']
        pre, rv = k['prereg'], k['review']
        line = f"- {n}" + (f" (registration sha256 `{pre['sha256'][:16]}`)" if pre else '') + ': '
        if rv is None:
            out.append(line + ('registered; ' if pre else 'not registered yet; ') + 'the review has not run yet.')
            out += _expansion_ink_lines(x)
            continue
        miss = miss or rv['n_flagged_control_on_letters'] < rv['n_control_windows_with_letters']
        line += (f"the review is {'VALID' if rv['valid'] else 'VOID'} ({rv['n_flagged_control_on_letters']} of the "
                 f"{rv['n_control_windows_with_letters']} control windows on known letters flagged)")
        if rv['n_flagged_screen'] == 0:
            out.append(line + f" and flagged no {n} window: a null, so nothing goes on to the later checks.")
            out += _expansion_ink_lines(x)
            continue
        regs = '; '.join(f"{m} {'windows' if len(w) > 1 else 'window'} {_fmt_list(w)}" for m, w in rv['regions'])
        line += (f" and flagged {_pl(rv['n_flagged_screen'], n + ' window')} in {_pl(rv['n_regions'], 'region')} "
                 f"({regs}).")
        nb, cf = k['neighbour'], k['confirm']
        if nb:
            how = (f"a window passes when it is present at d = +1 and at d = 0 or +2 and absent in the 180 degree "
                   f"negative" if nb['rule'].startswith(NEIGHBOUR_RULE) else f"rule: `{nb['rule']}`")
            line += (f" The neighbour-depth check (the same models at d = {_depths(nb['depths'])}; {how}) is "
                     f"{'VALID' if nb['valid'] else 'VOID'}, and {nb['n_pass_screen']} of {nb['n_screen']} windows pass it.")
        else:
            line += ' The neighbour-depth check has not run yet.'
        if nb and nb['n_pass_screen']:
            if cf is None:
                line += ' The blind-side confirmation has not started.'
            elif cf['state'] == 'running':
                line += ' The blind-side confirmation is running; its result is not in yet.'
            else:
                line += (f" The blind-side confirmation is {'VALID' if cf['valid'] else 'VOID'}: the control passed "
                         f"{cf['control_passed']} of {cf['n_control']}, and {cf['n_pass_screen']} of {cf['n_screen']} "
                         f"windows pass it.")
            if pre and pre['amendment1_rule']:
                line += (" Under Amendment 1 of the registration, a window passes the blind-side confirmation only if the "
                         "blinded reviewer marks it present in the reading-side image and absent both in the blind-side "
                         "image (the same models in the opposite depth order, shown with the reading side's statistics) "
                         "and in the 180 degree negative, with the control required to pass and one attempt allowed.")
        fk = k.get('facecheck')
        if fk:
            def _fc(r):
                nm = 'the control' if r['control'] else r['mesh'].split('_')[0]
                return f"{nm} (face asymmetry {r['asym']}, face consistency {r['consistency']})"
            rd = [r for r in fk['rows'] if r['reading']]
            bl = [r for r in fk['rows'] if not r['reading']]
            line += (" Description only, registered as Amendment 2 before the blind-side images were seen: the face "
                     "check of villa PR #2318, run unchanged on the dnative and seed42 maps, puts the two models' "
                     "agreement on the reading side for " + _fmt_list([_fc(r) for r in rd]))
            if bl:
                line += ', and on the blind side for ' + _fmt_list([_fc(r) for r in bl])
            same = all(r['same_side_both_masks'] for r in fk['rows'])
            sup = [r['mesh'].split('_')[0] for r in rd if not r['control']]
            nsup = [r['mesh'].split('_')[0] for r in bl if not r['control']]
            line += ('; the same side comes out with its own sheet check as the mask' if same else '') + '.'
            if sup or nsup:
                line += ' So it supports ' + (_fmt_list(sup) if sup else 'no window') + (
                    ' and goes against ' + _fmt_list(nsup) if nsup else '') + '; it gates nothing.'
        out.append(line)
        out += _expansion_ink_lines(x)
    miss = miss or any(e['ink']['review']['n_flagged_control_on_letters'] < e['ink']['review']['n_control_windows_with_letters']
                       for x in sc for e in x.get('expansions') or [] if e['ink']['review'])
    fig = Path(__file__).resolve().parents[1] / 'public/results/patches/candidates_PHerc0483B.png'
    if not fig.exists():
        fig = Path(__file__).resolve().parents[1] / 'results/patches/candidates_PHerc0483B.png'
    if fig.exists():
        out += ['', '![PHerc0483B blind-side confirmation crops](results/patches/candidates_PHerc0483B.png)', '',
                'The crops the blinded reviewer judged in the blind-side confirmation, each about 1 cm: the CT, the '
                'four-model ensemble on the reading side and the same models on the blind side (reading-side '
                'statistics), for the known-letter control and the two PHerc0483B candidate regions.']
    out += ['', "Flagged windows are candidates, places to look, not readings, and no letters are claimed." +
            (" A window without a flag is not a window without ink: the reviews miss some of the control's known letters."
             if miss else '')]
    return '\n'.join(out)


def _limits_text(s):
    sc, c = s['scrolls'], s['criteria']
    sw = '; '.join(f"on {x['scroll']} " + _fmt_list([f"{n} ({a} on {b} lines)" for n, a, b in x['limits']['top_switches']])
                   for x in sc)
    ocs = [(x['scroll'], x['limits']['off_centre']) for x in sc if x['limits']['off_centre']]
    oc = '; '.join(f"on {n} " + _fmt_list([f"{m} ({k})" for m, k in l]) for n, l in ocs) if ocs else 'on no patch'
    ats = [(x['scroll'], x['limits'].get('at_off_centre')) for x in sc if x['limits'].get('at_off_centre')]
    if ats:
        oc += (f", and exactly {OFF_CENTRE} layers from it " +
               '; '.join(f"on {n} " + _fmt_list([f"{m} ({k})" for m, k in l]) for n, l in ats))
    n_oc = sum(len(l) for _, l in ocs) + sum(len(l) for _, l in ats)
    nr = [(x['scroll'], x['limits']['not_rendered']) for x in sc if x['limits']['not_rendered']]
    nrt = (' ' + '; '.join(f"On {n}, {_fmt_list(l)} {'is' if len(l) == 1 else 'are'} not rendered yet." for n, l in nr)) if nr else ''
    out = ['**Limits.**',
           "- Switch events are counted, not gated. The detector (the vertical track matched to the surface changes while "
           "the old one carries on beside it) has not been calibrated, and it fires the same way where a vertical track "
           f"crosses a bridge between touching sheets. The most: {sw}.",
           f"- The brightest layer of the 66-layer render is more than {OFF_CENTRE} layers from the centre layer 33 {oc}, "
           f"so on {'that patch' if n_oc == 1 else 'those patches'} the centre layer may not sit on the sheet.{nrt}"]
    bars = [(x['scroll'], n, v) for x in sc for n, v in x['limits']['at_bar']]
    if bars:
        out.append("- " + _fmt_list([f"{n} on {sn} reaches the second-difference bar exactly ({v:.2f} voxels)"
                                     for sn, n, v in bars]) +
                   f"; the gate refuses only values over {c['k2k']:g} voxels, so {'it passes' if len(bars) == 1 else 'they pass'}.")
    ax = ''.join(f"; {x['limits']['axis_hull_median']:g} and {x['limits']['axis_hull_max']:g} on {x['scroll']}" for x in sc[1:])
    out.append("- The axis is coarse and has never been tested against a reference: the centroid of the m7 tracks' convex "
               f"hull sits a median of {sc[0]['limits']['axis_hull_median']:g} voxels from it over the band's z keys and "
               f"{sc[0]['limits']['axis_hull_max']:g} at worst on {sc[0]['scroll']}{ax}. It only chooses the outward side, "
               "and every patch's chosen side reaches the bar above.")
    t = s['totals']
    if t['sheet_fail']:
        out.append(f"- The survey's sheet check fails {t['sheet_fail']} of the {t['sheet_scored']} patches scored here, "
                   "although every patch here passes the gates above (Related work); which criterion is right on those "
                   "patches is not settled.")
    out.append("- These are surfaces, not text. No letters are claimed.")
    return '\n'.join(out)


def _repro_text(x):
    r = x['repro']
    t = (f"Run from this repository on the same inputs, `sheet_patches.py` rebuilt {_fmt_list(r['grids'])} with x, y and "
         f"z grids {'byte-identical to' if r['grids_byte_identical'] else 'DIFFERENT from'} the ones in the {x['scroll']} table")
    if r['validated']:
        t += (f"; `validate` on the first {len(r['validated'])} gave {r['validation_fields'] - r['validation_differing']}"
              f" of {r['validation_fields']} compared fields identical")
    if r['axis_points']:
        t += (f"; `axis` rebuilt the {r['axis_points']} control points of the coarse axis and its cross-sections "
              f"{'exactly' if r['axis_identical'] else 'with DIFFERENCES'}")
    return t + '.'


def _howto_text(s):
    out = ['**How to reproduce.** CPU only.']
    for x in s['scrolls']:
        if not x['repro']:
            continue
        n = x['scroll']
        out += ['', f"For {n}, from its surface tracks and m7 prediction:", '', '```',
                f"python uncontested/export_tracks.py {n}_tracks.dbm tracks_export.npz",
                "python uncontested/sheet_patches.py axis --tracks tracks_export.npz --axis axis.json --sections level5_sections.npz",
                "python uncontested/sheet_patches.py build --tracks tracks_export.npz --m7 sp_m7.zarr/0 --axis axis.json \\",
                "    --sections level5_sections.npz --out patches --record build.json",
                "python uncontested/sheet_patches.py validate --tracks tracks_export.npz --m7 sp_m7.zarr/0 --axis axis.json \\",
                "    --out patches --record build.json --validation validation.json", '```', '',
                f"The defaults reproduce {n}'s run; `build --key K` seeds at another z, and `python "
                f"uncontested/sheet_patches.py --help` lists every option. {_repro_text(x)}"]
    sha = s['script']['sha256']
    for x in s['scrolls']:
        if x['repro'] or x['cli'] is None:
            continue
        q, cl = x['selection'], x['cli']
        same = x['script_sha256'] == sha
        what = ('a further run of the same file, unchanged' if same else 'a run of a DIFFERENT file' if x['script_sha256']
                else 'a further run whose script hash is not recorded')
        out += ['', f"{x['scroll']} is {what} "
                f"(its run record gives sha256 `{(x['script_sha256'] or 'unrecorded')[:16]}`): `axis`, then `build` and "
                f"`validate` once per seed key with `--scroll {x['scroll']} --band {cl['band'][0]} {cl['band'][1]} "
                f"{cl['band'][2]} --voxel-um {cl['voxel_um']:g} --key K --first-index N` (N numbers the patches of each "
                f"key), then the patches of all keys pooled: a patch was {q['duplicate_rule']}. The pooling is a short "
                f"merge over the per-key outputs and is not part of the script."]
    for x in s['scrolls']:
        for e in x.get('expansions') or []:
            q, h, cl = e['selection'], e['script_sha256'], e['cli']
            what = ('a further run of the same file, unchanged' if h == sha else 'a run of a DIFFERENT file' if h
                    else 'a further run whose script hash is not recorded')
            opts = (f"--scroll {x['scroll']} --band {' '.join(str(b) for b in cl['band'])} --voxel-um {cl['voxel_um']:g} --key K"
                    if cl['band'] else '--key K')
            out += ['', f"The {x['scroll']} expansion is {what} (its run record gives sha256 `{(h or 'unrecorded')[:16]}`): "
                    f"`build {opts}` once per seed key to list its candidates, then `build` with `--candidate I --name N` "
                    f"added and `validate` with the same options once per candidate in the sector, then a merge over their "
                    f"outputs that is not part of the script: the gates were {q['gates']}; a patch was {q['duplicate_rule']}; "
                    f"and {q['cap_rule']}."]
    return '\n'.join(out)


def section_text(s: dict) -> str:
    sc, t, u, p, c = s['scrolls'], s['totals'], s['umbilicus'], s['params'], s['criteria']
    head = f"{t['n']} validated sheet patches" if t['n'] == t['n_pass'] else f"{t['n']} sheet patches, {t['n_pass']} validated"
    per = _fmt_list([f"{x['scroll']} {x['totals']['n']} ({x['totals']['area']:.3f} cm2)" for x in sc])
    hdr = ('| patch | seed z | z range | angle (deg) | area (cm2) | on sheet | second difference max (voxels) | vertical '
           'tracks on surface | same, shifted half a sheet | outward | switch events / lines | brightest render layer | '
           'survey sheet check: good fraction (tiles), result |\n|' + '---|' * 13)
    tables = '\n\n'.join(f"### {x['scroll']}\n\n{hdr}\n" + '\n'.join(_row(r) for r in x['rows']) +
                         f"\n\n**Totals.** {x['totals']['n']} patches, {x['totals']['n_pass']} passing every gate, "
                         f"{x['totals']['area']:.3f} cm2 in all, between z {x['totals']['z_min']} and {x['totals']['z_max']}. "
                         f"{_selection_text(x)} Across all {x['totals']['n']}: on-sheet fraction at least "
                         f"{x['totals']['on_sheet_min']:.4f}, outward fraction at least {x['totals']['outward_min']:.4f}, "
                         f"and the vertical tracks meet the surface at least {x['totals']['vertical_over_control_min']:g} "
                         f"times as often as its shifted copy." for x in sc)
    shc = s['totals']
    all_row = (f"| all | {t['n']} | {t['n_pass']} | {t['area']:.3f} | | | | "
               f"{str(shc['sheet_pass']) + ' of ' + str(shc['sheet_scored']) if shc['sheet_scored'] else 'not scored'} |")
    return f"""{BEGIN}
## Sheet patches without an umbilicus on First Letters scrolls

**What this is.** {head}, {t['area']:.3f} cm2 in all, on {t['n_scrolls']} First Letters scrolls: {per}. They come
from a CPU tracer that needs no umbilicus and no VC3D: `uncontested/sheet_patches.py` walks the organisers' m7 surface
prediction, seeded by the surface tracks that villa's `extract_surface_tracks.py` makes from it, and writes tifxyz that
villa's renderers and ink models read like any segment. {_script_text(s)} Every patch passes four gates: at least
{c['on_sheet']:g} of its vertices on the predicted sheet, no jump over {c['k2k']:g} voxels between rows, vertical tracks
that meet it at least {c['vertical_ratio']:g} times as often as a copy shifted half a sheet, and at least
{c['outward']:g} of its normals pointing outward (Validation, below).

| scroll | patches | passing every gate | area (cm2) | largest patch (cm2) | z range | seed keys | pass the survey's sheet check |
|---|---|---|---|---|---|---|---|
{chr(10).join(_scroll_row(x) for x in sc)}
{all_row}

{_related_text(s)}

**What the open data lacks.** These are First Letters scrolls. In the open-data `metadata.json`, {_missing_text(sc)}.
villa's spiral fitting needs an umbilicus to start, so it cannot run on them as published.

**Why not estimate an umbilicus first.** Two automatic estimators that read the m7 surface prediction were registered
and then scored blind on the {u['v1']['scrolls']} scrolls that have a published umbilicus
({_fmt_list(u['v1']['per_scroll'])}), against a bar of a median distance of at most {u['bar_median']} and a maximum of at
most {u['bar_max']} level-0 voxels. The first gave a median of {u['v1']['median']} and a maximum of {u['v1']['max']} over
{u['v1']['planes']} planes, the second a median of {u['v2']['median']} and a maximum of {u['v2']['max']} over
{u['v2']['planes']} planes. Both {'failed' if not (u['v1']['passed'] or u['v2']['passed']) else 'did not fail'}, so no
umbilicus was written and the patches below do without one.

**Method.** `uncontested/sheet_patches.py` traces local sheet patches through the m7 surface prediction, seeded by the
surface tracks that villa's `extract_surface_tracks.py` makes from it (`uncontested/export_tracks.py` turns their DBM
into one npz). A seed is a long horizontal track at one z, joined to others only across clean end-to-end breaks. The
walk moves every sample of the curve along its in-plane normal, {p['step_z']} slices at a time, to the nearest run of
predicted voxels on a ray of +-{p['ray']:g} voxels. A sample is accepted only when that run is within {p['dmax']:g}
voxels and within {p['frac_sp']:g} of the gap to the neighbouring sheet, at most {p['wmax']:g} voxels wide, and the
neighbouring sheet is at least {p['sp_min']:g} voxels away; a step is refused when fewer than {p['min_match']:g} of the
samples are accepted. Where the surface jumps between rows {p['key_step']} slices apart (second difference over
{p['k2k']:g} voxels) those columns are cut. An axis is used for one thing only, choosing which side of each patch faces
outward, and that axis is coarse: the centroid of the scan's cross-section. Patches are tifxyz with one vertex every
{p['vertex']:g} voxels, so `firstletters/render66.py` and villa's tools read them like any segment. CPU only.
{'Every scroll was traced with the same parameters.' if s['params_same'] else 'The parameters differ between scrolls; the summary JSON has them.'}

**Validation.** A patch is kept only if at least {c['on_sheet']:g} of its vertices lie on the predicted sheet (a
predicted voxel within {c['box']} voxels), the second difference between rows {p['key_step']} slices apart never exceeds
{c['k2k']:g} voxels, the vertical tracks meet the surface at least {c['vertical_ratio']:g} times as often as they meet
the same surface shifted half a sheet sideways, and at least {c['outward']:g} of its normals point outward on the
chosen side. {'The gates are the same on every scroll.' if s['criteria_same'] else 'The gates differ between scrolls; the summary JSON has them.'}

{tables}

{_ink_text(s)}

{_limits_text(s)}

{_howto_text(s)}
{END}"""


def write_atomic(path: Path, text: str):
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(text)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', action='store_true', help='rebuild results/patches/patches_summary.json')
    ap.add_argument('--write', action='store_true', help='splice into public/README.md and copy to the published layout')
    a = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    sp = root / 'results/patches/patches_summary.json'
    if a.summary:
        sp.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(sp, json.dumps(build_summary(root), indent=1) + '\n')
    s = json.loads(sp.read_text())
    text = section_text(s)
    if a.write:
        rp = root / 'public/README.md'
        r = rp.read_text()
        r = r.split(BEGIN)[0] + text + r.split(END, 1)[1] if BEGIN in r else r.rstrip('\n') + '\n\n' + text + '\n'
        write_atomic(rp, r)
        (root / 'public/results/patches').mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, root / 'public/results/patches/patches_summary.json')
        shutil.copy2(Path(__file__).resolve(), root / 'public/bench/make_patches_section.py')
    else:
        sys.stdout.write(text + '\n')


if __name__ == '__main__':
    main()
