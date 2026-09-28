"""ARM FLINJ figure panels: glyph mask, P(no injection), P(injected) and D side by side, same pixels."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def panels(path, binary, P0, P_inj, title, margin=96, max_w_px=1800, labels=("P(no injection)", "P(injected)")):
    ys, xs = np.nonzero(binary)
    y0, y1 = max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin + 1, binary.shape[0])
    x0, x1 = max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin + 1, binary.shape[1])
    sl = (slice(y0, y1), slice(x0, x1))
    D = P_inj[sl].astype(np.float64) - P0[sl].astype(np.float64)
    lim = max(float(np.abs(D).max()), 1.0)
    h, w = y1 - y0, x1 - x0; horiz = w >= h
    fig, ax = plt.subplots(4 if horiz else 1, 1 if horiz else 4, figsize=(16, 4 * 16 * h / w + 1.2) if horiz else (16, 4 * h / w + 1.2))
    for a, img, name, kw in zip(ax, [binary[sl], P0[sl], P_inj[sl], D],
                                ["glyph mask", labels[0], labels[1], f"D = {labels[1]} - {labels[0]}, +/-{lim:.0f}"],
                                [dict(cmap="gray", vmin=0, vmax=1), dict(cmap="gray", vmin=0, vmax=255), dict(cmap="gray", vmin=0, vmax=255), dict(cmap="RdBu_r", vmin=-lim, vmax=lim)]):
        a.imshow(img, interpolation="nearest", **kw); a.set_title(name, fontsize=9); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{title}  (rows {y0}:{y1}, cols {x0}:{x1})", fontsize=10)
    fig.tight_layout()
    dpi = min(100, max_w_px / fig.get_size_inches()[0])
    fig.savefig(path, dpi=dpi); plt.close(fig)
    return [y0, y1, x0, x1]
