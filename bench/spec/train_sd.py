"""ARM SPEC fine-tune, results/spec/PREREGISTRATION_SPEC.md (sha 4df26a95...).

    python ops/spec/train_sd.py --arm K|T --out results/spec/K [--steps 1500]

Self-distillation of ink_9um seed42 step-075000 on the inputs of the four FLINJ hosts (labels are never read). Each
step draws 20 patches from one seeded sequence shared by both arms: 16 normal patches with the frozen teacher's
probability as target, plus 4 more that arm K keeps normal (teacher target) and arm T depth-disorders (target 0).
Model in eval mode throughout (frozen BatchNorm, so step 0 equals the teacher exactly), wrapped in villa's TargetModel,
patches normalised with villa's normalize_flat_patch exactly as inference does. Resumable: state every 100 steps.
Writes <out>/ckpt.pth as {model, config, step} like the released files, and <out>/train_log.jsonl.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "villa" / "vesuvius" / "src"))
from vesuvius.ink_detection.config import InkConfig  # noqa: E402
from vesuvius.ink_detection.inference import infer as INF  # noqa: E402
from vesuvius.ink_detection.models.checkpoint import load_checkpoint, select_inference_weights  # noqa: E402
from vesuvius.ink_detection.models.model import make_model  # noqa: E402

CKPT = ROOT / "models/ink_9um/hybrid_3d2d-seed42/step-075000.pth"
CROPS = {"w00": ("data/zs/w00/w04/ct.zarr", "data/zs/w00/support.npy"),
         "ag144": ("data/zs/ag144/w04/ct.zarr", "data/zs/ag144/support.npy"),
         "ag174": ("data/zs/ag174/w04/ct.zarr", "data/zs/ag174/support.npy"),
         "p0500p2": ("data/f6/p0500p2/N0/ct.zarr", "data/f6/p0500p2/support.npy")}
P, D, NB, NX, CELL = 128, 17, 16, 4, 32


def banner():
    b = pathlib.Path(__file__).read_bytes()
    print(f"train_sd.py sha256 {hashlib.sha256(b).hexdigest()[:16]} pid {os.getpid()}", flush=True)


def build(payload, device):
    config = InkConfig.from_mapping(payload["config"])
    _, state = select_inference_weights(payload, source=str(CKPT))
    base = make_model(config)
    inc = base.load_state_dict(state, strict=True)
    assert not inc.missing_keys and not inc.unexpected_keys
    model = INF.TargetModel(base, input_pad_depth_to=config.model.input_pad_depth_to).to(device).eval()
    assert config.model.input_pad_depth_to is None and tuple(config.model.crop_size) == (D, P, P)
    return base, model, INF.flat_preprocessing_from_config(config.data.normalization)


def logits(model, x):
    out = model(x)
    return out.reshape(out.shape[0], P, P)


def disorder(patch, rng):
    """(i) one permutation of the planes for the whole patch, or (ii) an independent permutation per 32 x 32 cell."""
    if rng.random() < 0.5:
        return patch[rng.permutation(D)], "whole"
    out = patch.copy()
    for y in range(0, P, CELL):
        for x in range(0, P, CELL):
            out[:, y:y + CELL, x:x + CELL] = patch[rng.permutation(D), y:y + CELL, x:x + CELL]
    return out, "cell"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("K", "T"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    banner()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps")
    torch.manual_seed(7)
    payload = load_checkpoint(CKPT)
    t_base, teacher, prep = build(payload, device)
    s_base, student = build(payload, device)[:2]
    assert prep == "tifxyz_robust", prep
    for p in teacher.parameters():
        p.requires_grad_(False)

    arrays, supports, centres = {}, {}, {}
    for k, (ct, sup) in CROPS.items():
        arr = zarr.open_array(str(ROOT / ct), mode="r")[:]
        s = np.load(ROOT / sup)
        assert arr.shape == (21,) + s.shape and arr.dtype == np.uint8, (k, arr.shape, s.shape)
        m = np.zeros_like(s); m[P // 2:-P // 2, P // 2:-P // 2] = True
        arrays[k], supports[k], centres[k] = arr, s, np.argwhere(s & m)
        print(f"{k} {arr.shape} valid centres {len(centres[k])}", flush=True)
    keys = list(CROPS)

    opt = torch.optim.SGD(student.parameters(), lr=1e-3, momentum=0.9, nesterov=True, weight_decay=3e-5)
    rng = np.random.default_rng(7)          # patch sequence, identical for K and T
    drng = np.random.default_rng(1007)      # disorder draws, consumed by T only
    start = 0
    state_path = out / "state.pth"
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        student.load_state_dict(st["student"]); opt.load_state_dict(st["opt"])
        rng.bit_generator.state = st["rng"]; drng.bit_generator.state = st["drng"]; start = st["step"]
        print(f"resumed at step {start}", flush=True)

    log = open(out / "train_log.jsonl", "a")
    for step in range(start, a.steps):
        t0 = time.time()
        xs, ms, kinds = [], [], []
        for _ in range(NB + NX):
            k = keys[rng.integers(len(keys))]
            y, x = centres[k][rng.integers(len(centres[k]))]
            z0 = int(rng.integers(0, 5))
            xs.append(arrays[k][z0:z0 + D, y - P // 2:y + P // 2, x - P // 2:x + P // 2])
            ms.append(supports[k][y - P // 2:y + P // 2, x - P // 2:x + P // 2])
            kinds.append("normal")
        if a.arm == "T":
            for i in range(NB, NB + NX):
                xs[i], kinds[i] = disorder(xs[i], drng)
        x = torch.from_numpy(np.stack([INF.normalize_flat_patch(p, prep) for p in xs])[:, None]).to(device)
        m = torch.from_numpy(np.stack(ms)).to(device)
        neg = torch.tensor([kd != "normal" for kd in kinds], device=device)
        with torch.no_grad():
            tlog = logits(teacher, x); tprob = torch.sigmoid(tlog)
            if step == 0:
                d0 = float((logits(student, x) - tlog).abs().max())
                assert d0 == 0.0, f"student differs from teacher at step 0 by {d0}"
                print(f"step-0 check: student == teacher exactly (max |diff| {d0})", flush=True)
        target = torch.where(neg[:, None, None], torch.zeros_like(tprob), tprob)
        lr = 1e-3 * min(1.0, (step + 1) / 50)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        lg = logits(student, x)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(lg, target, reduction="none")
        per = (bce * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1)
        loss = per.mean()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
        opt.step()
        rec = dict(step=step + 1, loss=float(loss), loss_normal=float(per[~neg].mean()),
                   loss_neg=float(per[neg].mean()) if bool(neg.any()) else None, grad_norm=gn, lr=lr,
                   secs=round(time.time() - t0, 3), kinds=[kd for kd in kinds[NB:]])
        log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % 25 == 0:
            print(json.dumps(rec), flush=True)
        if (step + 1) % 100 == 0 or step + 1 == a.steps:
            tmp = out / f"state.pth.partial{os.getpid()}"
            torch.save(dict(student=student.state_dict(), opt=opt.state_dict(), rng=rng.bit_generator.state,
                            drng=drng.bit_generator.state, step=step + 1), tmp)
            os.replace(tmp, state_path)
    final = dict(model=s_base.state_dict(), config=payload["config"], step=int(payload.get("step", 0)) + a.steps)
    tmp = out / f"ckpt.pth.partial{os.getpid()}"
    torch.save(final, tmp); os.replace(tmp, out / "ckpt.pth")
    print(f"wrote {out / 'ckpt.pth'}", flush=True)


if __name__ == "__main__":
    main()
