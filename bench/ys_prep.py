"""ARM Y, stage 2: villa's production 9 um recipe on every window of the published volume.

For window z0 the recipe is handed planes [z0, z0+84), so its own centred slice
of 84 planes is the whole input. The production window is whatever
`centered_slice(109, 84)` returns (13), and C4 checks that window bit for bit
against the recipe run on the full 109-plane crop, so the sweep is known to use
the same arithmetic as production.

    env/bin/python3 ops/ys_prep.py [w00 ag144 ag174]

Writes data/ys/<seg>/wNN/ct.zarr, a plain (21,H,W) uint8 array, staged then
renamed; data/ys/<seg>/prep.json records C4 and each window's source planes.
"""
import json, pathlib, shutil, sys, tempfile
import numpy as np, zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "villa" / "vesuvius" / "src"))
from vesuvius.ink_detection.preprocessing import prepare_9um_isotropic_input as P9  # noqa: E402

WINDOWS = [1, 5, 9, 13, 17, 21, 25]  # pre-registered, results/ys/PREREGISTRATION_ARMY.md


def prepare(src, workdir):
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="prep.", dir=workdir))
    try:
        zarr.open_array(str(scratch / "src.zarr"), mode="w", shape=src.shape, chunks=(src.shape[0], 128, 128), dtype=np.uint8, zarr_format=2)[:] = src
        out = P9.prepare_isotropic_input(str(scratch / "src.zarr"), scratch / "prep.zarr", level="2", workers=4)
        node = zarr.open(str(out), mode="r")
        arr = node if hasattr(node, "shape") else node[sorted(node.array_keys())[0]]
        return np.asarray(arr[:]), dict(node.attrs)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def write_ct(dest, ct):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=dest.name + ".", dir=dest.parent))
    zarr.open_array(str(tmp / "ct.zarr"), mode="w", shape=ct.shape, chunks=(ct.shape[0], 128, 128), dtype=np.uint8, zarr_format=2)[:] = ct
    if dest.exists(): shutil.rmtree(dest)
    tmp.rename(dest)


def main(segs):
    for seg in segs:
        sd = ROOT / "data" / "ys" / seg
        raw = np.asarray(zarr.open_array(str(sd / "raw.zarr"), mode="r")[:])
        D = raw.shape[0]
        prod0, prod1 = P9.centered_slice(D, P9.INPUT_Z)
        assert prod0 in WINDOWS and prod1 - prod0 == P9.INPUT_Z, (prod0, prod1)
        rec = dict(depth=D, input_z=P9.INPUT_Z, format=P9.FORMAT_TAG, production_window=[prod0, prod1], windows={})
        full, full_attrs = prepare(raw, sd)  # production, untouched: the recipe picks its own planes
        for z0 in WINDOWS:
            dest = sd / f"w{z0:02d}"
            ct, attrs = prepare(raw[z0:z0 + P9.INPUT_Z], sd)
            assert attrs["source_z_slice"] == [0, P9.INPUT_Z], attrs["source_z_slice"]
            assert ct.shape == (P9.OUTPUT_Z,) + raw.shape[1:], ct.shape
            if z0 == prod0:
                same = bool(np.array_equal(ct, full))
                rec["C4"] = dict(pass_=same, production_source_z_slice=full_attrs["source_z_slice"],
                                 max_abs_diff=int(np.abs(ct.astype(np.int16) - full.astype(np.int16)).max()))
                assert same, f"C4 FAILED on {seg}: window {z0} differs from the production recipe"
            write_ct(dest, ct)
            rec["windows"][f"w{z0:02d}"] = dict(source_planes=[z0, z0 + P9.INPUT_Z], centre=z0 + P9.INPUT_Z / 2 - 0.5,
                                               offset_from_production=z0 - prod0)
            print(f"{seg} w{z0:02d} planes [{z0},{z0 + P9.INPUT_Z}) -> {ct.shape}", flush=True)
        json.dump(rec, open(sd / "prep.json", "w"), indent=1)
        print(f"{seg}: C4 {'PASS' if rec['C4']['pass_'] else 'FAIL'}, production planes {rec['C4']['production_source_z_slice']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["w00", "ag144", "ag174"])
