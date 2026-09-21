Two follow-ups: a gap in my own validation that I have now closed, and a cross reference to work on the sibling inference path that a reviewer should see next to this.

**The gap. Everything measured in this PR used one model type, and it is not the default.** `optimized_inference` dispatches four (`entrypoint.py:717`): `timesformer`, `resnet3d-50`, `resnet3d-152` and `resnet3d-152-3d-decoder`, and `InferenceInputs.model_type` defaults to `timesformer`. Every number in this PR came from `resnet3d-152-3d-decoder`. A device patch can fail in exactly two ways, a missing MPS kernel or a numeric divergence, and neither depends on trained weights, so I ran each architecture with random weights on both devices with the same input. torch 2.14.0, MPS available true, 64 px input, one tile.

| `MODEL_TYPE` | frames | MPS fp32 vs CPU, max abs diff | MPS autocast vs CPU, max abs diff |
|---|---:|---:|---:|
| `timesformer` (the default) | 26 | 1e-06 | 0.000566 |
| `resnet3d-50` | 30 | 0 | 0.000143 |
| `resnet3d-152` | 30 | 1e-06 | 0.000705 |
| `resnet3d-152-3d-decoder` | 62 | 1e-06 | not finite |

**All four run on MPS with no missing kernel, in fp32 and under the `torch.autocast(device_type=device.type)` expression this PR introduces.** In fp32 every model agrees with the CPU to 1e-06 or exactly.

**The one cell that is not a pass, reported as such.** `resnet3d-152-3d-decoder` under autocast with RANDOM weights returns a non finite output, so there is no difference to quote and I have marked it undecided rather than OK. Untrained weights make activations large and float16 overflows. The control that separates "untrained weights overflow" from "this patch is broken" is to repeat it with the real checkpoint, which is the only weights anyone runs: `r152_3ddec_v2_l5_epoch13.ckpt` on MPS gives finite output on both paths with max abs diff 0.001018. So it is a float16 range artifact of the fixture, not a defect. It is also consistent with the end to end run already in this PR, where the two paths agree at Pearson 0.999999511858424 over 49 real tiles with 0.0138% of pixels crossing the 0.5 decision.

**And an honest note about the environment those earlier numbers were taken in.** villa's `optimized_inference/requirements.txt` lists `pytorch-lightning`, and my venv did not have it, because the 3d decoder module is the one model file that does not import it. So at the time I measured this PR, two of the four model modules could not even be imported in my environment and I had not noticed. That is why this check did not happen sooner.

**The cross reference.** @SurgeFok filed #1764 on 2026-09-11 and #1770 on 2026-09-12, which is the same defect class in `koine_machines.inference`: `prepare_model_for_inference` and `prepare_model` both resolve the device as CUDA or CPU and gate autocast the same way, so Apple Silicon silently takes the CPU path there too, with measured 5.97x MPS over CPU on an M5 Pro. That is earlier than this PR and I did not know about it when I opened this one.

The two do not overlap and neither replaces the other: #1770 touches `koine_machines` and zero files under `ink-detection/`, and this PR touches `ink-detection/optimized_inference` only. A maintainer looking at Apple Silicon support probably wants both, and the fix in each is the same shape, so whichever is reviewed first sets the pattern for the other.

Raw numbers and the script are in `results/mps_model_coverage.json` and `bench/mps_model_coverage.py` at https://github.com/AndreasHad04/villa-apple-silicon , MIT.
