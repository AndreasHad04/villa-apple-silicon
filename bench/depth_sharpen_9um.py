"""Depth-sharpen the 21-layer input window of a native 1.2 m surface volume before running ink_9um on it.

The eligible First Letters volumes are 1.2 m scans, and their rendered layers are much more correlated with
their neighbours than the 2.4 um-derived layers ink_9um was mostly trained on. This applies a fixed depth
gain and a fixed intensity map, both fitted WITHOUT labels on paired PHerc0841 renders (see ARM F in the
README), to each pixel's 21-sample depth profile. Nothing is fitted on the input you give it.

    python depth_sharpen_9um.py IN.zarr OUT.zarr [--no-map] [--layer-um 9.362]

IN.zarr is a (21, H, W) uint8 zarr v2 array: the centred 21 layers of a native surface volume, as
villa's centred_slice(depth, 21) chooses them. OUT.zarr is written the same shape, chunks (21, 128, 128).
"""
import argparse, json, pathlib
import numpy as np, zarr

P = json.load(open(pathlib.Path(__file__).with_name("depth_sharpen_9um.json")))


def sharpen(vol, layer_um, use_map=True):
    D = vol.shape[0]; assert D == P["window_layers"], f"expected {P['window_layers']} layers, got {D}"
    v = vol.astype(np.float64); v = np.concatenate([v[::-1], v, v[::-1]], axis=0)
    f = np.fft.rfftfreq(3 * D, d=layer_um); G = np.interp(f, P["fz_cycles_per_um"], P["g_depth"], right=P["g_depth"][-1])
    x = np.fft.irfft(np.fft.rfft(v, axis=0) * G[:, None, None], n=3 * D, axis=0)[D:2 * D]
    if use_map: x = np.interp(x, P["map_src"], P["map_tgt"])
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--no-map", action="store_true"); ap.add_argument("--layer-um", type=float, default=9.362)
    a = ap.parse_args(); src = np.asarray(zarr.open_array(a.inp, mode="r")[:])
    out = sharpen(src, a.layer_um, not a.no_map)
    zarr.open_array(a.out, mode="w", shape=out.shape, chunks=(out.shape[0], 128, 128), dtype=np.uint8, zarr_format=2)[:] = out
    print("wrote", a.out, out.shape)
