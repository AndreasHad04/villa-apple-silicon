#!/usr/bin/env python3
"""Generate the NL343 README section (the PHerc. 343 First Letters render without VC3D) from measured JSON.

Working tree: `python ops/make_nl343_section.py --summary` rebuilds results/nieuwlaar/nl343_summary.json from
results/nieuwlaar/g2.json and results/nieuwlaar/arm_S2/result.json; `--write` splices the section into
public/README.md between the NL343 markers. Published repo: section_text(summary) regenerates the section from the
shipped summary alone, which is what bench/verify_claims.py checks.
"""
import argparse, hashlib, json, sys
from pathlib import Path

BEGIN, END = '<!-- NL343:BEGIN -->', '<!-- NL343:END -->'


def build_summary(root: Path) -> dict:
    g2 = json.loads((root / 'results/nieuwlaar/g2.json').read_text())
    s2 = json.loads((root / 'results/nieuwlaar/arm_S2/result.json').read_text())
    e1 = s2['images']['d+1']['endpoint']
    f34 = g2['f']['34']
    out = root / 'results/nieuwlaar/repo/outputs'
    side = json.loads((out / 'concat_w047-w048_R5B2_z9500-11000.json').read_text())
    ri = json.loads((root / 'data/nieuwlaar/concat_w047-w048_R5B2_z9500-11000_66.zarr.render.json').read_text())
    his_png = hashlib.sha256((out / 'concat_w047-w048_R5B2_z9500-11000.png').read_bytes()).hexdigest()
    return dict(
        render=dict(chunks=ri['info']['chunks_needed'], gb=round(ri['info']['bytes_fetched'] / 1e9, 2),
                    seconds_compute=ri['info']['seconds_compute']),
        ct=dict(lo=side['ct']['stretch']['lo'], hi=side['ct']['stretch']['hi'], dim=side['overlay']['ct_dim']),
        g2=dict(valid_px=g2['valid_px'], grey_px=g2['grey_set_px'], grey_share=g2['grey_set_over_valid'],
                f34=f34, f33=g2['f']['33'], f35=g2['f']['35'], f31_flipped=g2['f']['31'],
                off_by_more_than_1=int(round((1 - f34) * g2['grey_set_px'])),
                sha_layer34_equal=g2['sha_layer34_ours'] == g2['sha_layer34_his'],
                pass_f34=g2['pass_f34'], discriminates=g2['discriminates']),
        s2=dict(verdict=s2['verdict'], n_C=e1['n_C'], r_alpha=e1['r_alpha'], r_alpha_letters=e1['r_alpha_letters'],
                r_lum=e1['r_lum'], boxes_agree=e1['boxes_agree'], boxes_total=len(e1['per_box']),
                r_alpha_d0=s2['images']['d+0']['endpoint']['r_alpha'],
                r_alpha_d2=s2['images']['d+2']['endpoint']['r_alpha'],
                image_sha_equal_his=s2['images']['d+1']['sha256'] == his_png),
        prereg_sha256=hashlib.sha256((root / 'results/nieuwlaar/PREREGISTRATION_MAC.md').read_bytes()).hexdigest(),
        prereg_original_sha256=(root / 'results/nieuwlaar/prereg.sha256').read_text().split()[0],
        renderer_sha256=hashlib.sha256((root / 'ops/render/render_tifxyz.py').read_bytes()).hexdigest(),
        grids=_grids(root),
    )


def _grids(root: Path) -> dict:
    ck = json.loads((root / 'results/render66_flags/check.json').read_text())
    fo = json.loads((root / 'results/render_forensics/forensics.json').read_text())
    rows = fo['A_per_layer']
    return dict(default_identical=ck['a_default_identical']['array_equal'], passed=ck['pass'],
                identical_median=ck['b_flip31']['official']['identical_median'],
                corr_median=ck['b_flip31']['official']['corr_median'], mask_px=ck['b_flip31']['official']['mask_px'],
                noflip_corr_median=ck['c_noflip31']['official']['corr_median'],
                pair_mad=round(sum(r['pair_u8_mad'] for r in rows) / len(rows), 3))


def section_text(s: dict) -> str:
    g, k, c, rr = s['g2'], s['s2'], s['ct'], s['render']
    return f"""{BEGIN}
## The PHerc. 343 First Letters image, reproduced without VC3D or an NVIDIA GPU

Erwin Nieuwlaar's First Letters result on PHerc. 343 ([pherc343-first-letters](https://github.com/Nieuwlaar/pherc343-first-letters),
MIT) renders its segment with VC3D build `41701aa`, which can no longer be downloaded
([villa #2049](https://github.com/ScrollPrize/villa/issues/2049)). `firstletters/render66.py` does that step in Python
on the CPU, in the convention of his `mesh/render_66_layers.sh` (66 layers one voxel apart along the normal,
d = layer - 33). The sampling is `firstletters/render_tifxyz.py`, written from villa's `vc_render_tifxyz` sources.

```
python firstletters/render66.py --mesh outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz \\
    --volume https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr/0 \\
    --out concat_w047-w048_R5B2_z9500-11000_66.zarr --cache-dir chunks_0343
```

{_grids_text(s)}

By default the output is a 66 x H x W uint8 zarr that villa's ink CLI and his `hecate_prep.py` read as it is (that is how the
maps below were made). It needs numpy, tifffile, urllib3, zarr and numcodecs, and fetches {rr['chunks']:,} chunks
({rr['gb']:.2f} GB) of the open-data volume once.

**The render, checked against his published image.** His PNG shows CT layer 34 in grey wherever no ink is drawn,
so every such pixel can be predicted from our render with his own stretch ({c['lo']:g} to {c['hi']:g}) and dimming ({c['dim']:g}).

| our layer | share of the compared grey pixels within 1 grey level of his |
|---|---|
| 34, the layer his image uses | {g['f34']:.7f} |
| 33 | {g['f33']:.4f} |
| 35 | {g['f35']:.4f} |
| 31 (normal flipped) | {g['f31_flipped']:.4f} |

{g['grey_px']:,} pixels compared ({100 * g['grey_share']:.2f}% of the {g['valid_px']:,} valid ones; the rest carry ink
or his row lines); {g['off_by_more_than_1']} of them differ by more than 1. The layer is not bit-identical to his
(its sha256 differs from the one his sidecar records).

**The ink, on Apple Silicon.** The four models of his ensemble (dense_native, ink_9um seeds 42 and 43, Hecate 9.6 um)
ran on an M1 Max (MPS, fp32) at d = +1 on both tilings of our render, and his own `make_submission_image.py` made the
image from our maps with the statistics published in his sidecar:

| measured against his image | value |
|---|---|
| Pearson r of ink opacity, {k['n_C']:,} compared pixels | {k['r_alpha']:.5f} |
| the same inside his {k['boxes_total']} letter boxes | {k['r_alpha_letters']:.5f} |
| Pearson r of luminance | {k['r_lum']:.5f} |
| letter boxes whose ink fraction is within 0.8 to 1.25 of his | {k['boxes_agree']} of {k['boxes_total']} |
| depth control: opacity r from our d = 0 and d = +2 maps instead | {k['r_alpha_d0']:.4f} and {k['r_alpha_d2']:.4f} |

Pre-registered before any ink model ran on this render: `results/nieuwlaar/PREREGISTRATION_MAC.md` (original text
sha256 `{s['prereg_original_sha256'][:16]}`; one amendment, after two timing runs, changed no endpoint, threshold or arm).
**Limits.** This uses his published statistics. The pre-registered primary recomputes them from our own maps at all
31 depths (248 model runs) and has not finished yet; it will be reported here whatever it shows. Our image is not
bit-identical to his{'' if not k['image_sha_equal_his'] else ' (it is, by sha256)'}.
{END}"""


def _grids_text(s: dict) -> str:
    g = s['grids']
    return ("**Other grids.** `--num-slices N` renders N layers centred at (N - 1) / 2, as `vc_render_tifxyz` does, and "
            "`--flip-normals` samples along -n, so `--num-slices 31 --flip-normals` gives VC3D's default 31-slice render "
            "with flipped normals. On PHerc0800 segment 20251029010146 that reproduces the published VC3D render: "
            f"{100 * g['identical_median']:.2f}% of the {g['mask_px']:,} compared voxels per layer have the same value "
            "(median over the 31 layers; the others differ by one grey level), Pearson r "
            f"{g['corr_median']:.7f}, and without `--flip-normals` r is {g['noflip_corr_median']:.3f}. The default 66 "
            "layers sit half a voxel from VC3D's 31, so a comparison between the two grids has to average two "
            f"neighbouring layers, which leaves about {g['pair_mad']:.2f} grey levels of difference even between two "
            "renders from this same renderer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', action='store_true', help='rebuild results/nieuwlaar/nl343_summary.json')
    ap.add_argument('--write', action='store_true', help='splice the section into public/README.md')
    a = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    sp = root / 'results/nieuwlaar/nl343_summary.json'
    if a.summary:
        sp.write_text(json.dumps(build_summary(root), indent=1) + '\n')
    s = json.loads(sp.read_text())
    text = section_text(s)
    if a.write:
        rp = root / 'public/README.md'
        r = rp.read_text()
        if BEGIN in r:
            r = r.split(BEGIN)[0] + text + r.split(END, 1)[1]
        else:
            r = r.rstrip('\n') + '\n\n' + text + '\n'
        rp.write_text(r)
    else:
        sys.stdout.write(text + '\n')


if __name__ == '__main__':
    main()
