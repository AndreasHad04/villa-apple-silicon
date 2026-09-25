"""Dump the ARM F depth-sharpening parameters (depth gain and intensity map, NDH), fitted on ALL THREE PHerc0841
pairs without labels, exactly as ops/f5.py make() fits them, for the standalone tool bench/depth_sharpen_9um.py."""
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "ops"))
import fs_make as M  # noqa: E402
fits = [M.fit_pair(s) for s in M.SEGS]; fz, gz = M.z_gain_curve(fits); data = {s: M.load(s) for s in M.SEGS}
src = np.concatenate([M.apply_depth(data[s][0].astype(np.float64), fz, gz, M.DZN)[:, data[s][2]].ravel() for s in M.SEGS])
tgt = np.concatenate([data[s][1][:, data[s][3]].ravel() for s in M.SEGS]); qs, qt = M.quantile_map(src, tgt)
out = dict(what="ARM F depth sharpening (NDH) for ink_9um on native 1.2 m surface volumes; fitted without labels on PHerc0841 "
                "w00, ag144, ag174, 9.366 um 1.2 m 113 keV against the same segments' 2.403 um 0.22 m 77 keV volumes through villa's 9 um recipe",
           depth_um_per_layer=M.DZN, fz_cycles_per_um=fz.tolist(), g_depth=gz.tolist(), map_src=qs.tolist(), map_tgt=qt.tolist(), window_layers=21)
dest = ROOT / "public" / "bench" / "depth_sharpen_9um.json"; json.dump(out, open(dest, "w"), indent=1); print("wrote", dest, "g_depth", np.round(gz, 3).tolist())
