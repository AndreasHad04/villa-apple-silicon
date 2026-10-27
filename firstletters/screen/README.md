# First Letters screen scripts (exact versions)

These are the scripts that made the PHerc0800 ink atlas release (`pherc0800-atlas-v1`) and the axis-free patch screens,
copied unchanged from our working tree. They are published so every step of the release can be read and re-run; they
are orchestration around public pieces: Erwin Nieuwlaar's PHerc0343 recipe
(https://github.com/Nieuwlaar/pherc343-first-letters, commit e8370f1), villa's `vesuvius` package, the public 9 um ink
models and Reader V3, and this repository's `firstletters/render66.py` and `firstletters/render_tifxyz.py`.

Each script sets `ROOT = Path('/Users/andreashad04/money/vesuv')` in one line. Change it to your own folder, and lay that
folder out as the scripts expect:

| path under ROOT | what |
|---|---|
| `ops/fl/`, `ops/nieuwlaar/` | these scripts |
| `public/firstletters/` | `render66.py` and `render_tifxyz.py` from this repository |
| `results/nieuwlaar/repo/` | a clone of Nieuwlaar's repository (its `ink/scripts` gives `ensemble_maps.py` and `make_submission_image.py`) |
| `models/nieuwlaar_ckpt/`, `models/reader_v3/` | the checkpoints named in `ops/nieuwlaar/run_unit.py` (`CKPT`) and `ops/fl/rv3_unit.py` (sha256 pinned there) |
| `scratch_envs/env212/`, `villa-inkcopy/vesuvius/src/` | the Python with torch 2.12.1 and the villa source that `run_unit.py` runs the villa models with |
| `env/` | the Python (torch 2.14.0) that runs hecate and these scripts |
| `data/fl/`, `results/fl/` | inputs (meshes, patches) and outputs, created as the scripts run |

Later additions to these files did not change what the atlas was made with: `orient.py` gained entries for other scrolls
(its PHerc0800 entry is unchanged), and `screen_unit.py`, `run_unit.py` and `ensemble.py` gained the blind-side options
(`--directions`, `--direction`, `--blind`, `--stats-side`) with defaults that reproduce the earlier maps byte for byte.

The commands that made the release are in its README, section 11. MIT licence, as the rest of this repository.

sha256 of each file as published:

- `ops/fl/atlas_pack.py` 6cecaa0a32270cbf
- `ops/fl/ensemble.py` fa87e28192a6f2b6
- `ops/fl/orient.py` 09ffa7c7b6021351
- `ops/fl/rv3_screen_unit.py` 36de620b92fcaf3a
- `ops/fl/rv3_unit.py` 0e4415097780eb6d
- `ops/fl/rv3_with.py` c0d3d25ccd691e51
- `ops/fl/screen_unit.py` 143f1dc49f5867a9
- `ops/nieuwlaar/run_unit.py` 09c205b50f432aa9
