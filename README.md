# Apple Silicon support for villa's ink detection inference

`villa/ink-detection/optimized_inference` selects its device with

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

There is no MPS branch, so every Apple Silicon Mac silently takes the CPU path.
Downstream of that, the autocast device type is chosen as

```python
amp_device = "cuda" if device.type == "cuda" else "cpu"
```

which, when the tensors are on MPS, makes autocast a no-op rather than an
error.

This repository holds a small patch that adds the MPS path, plus the
measurements behind it and a local runner that needs no container and no AWS
credentials. Everything is measured on real scroll data: PHerc. 1667, segment
`20240304141531-w013_20240304141531_flatboi`, the 2.399 um surface volume, with
`scrollprize/ink_canonical_2um` (`r152_3ddec_v2_l5_epoch13.ckpt`).

## What it buys

Same checkpoint, same tiles, `TILE_SIZE=256`, `STRIDE=128`, layers 1 to 63,
loaded through villa's own `model_resnet3d_3d_decoder.load_model`.

Nine timed repetitions per MPS configuration and three for CPU, after an
untimed warmup, reported as min / median / max per villa's `AGENTS.md` rule
1.4 rather than as a bare mean.

| path | min | **median** | max | forward memory |
|---|---|---|---|---|
| CPU, fp32, what a Mac gets today | 3.259 | **3.261** | 3.278 | **9.52 GB** |
| MPS, autocast off | 0.524 | 0.525 | 0.526 | 2.59 GB |
| MPS, villa's current `amp_device` | 0.525 | 0.526 | 0.527 | 2.59 GB |
| MPS, `amp_device` corrected | 0.426 | **0.427** | 0.428 | 7.85 GB at batch 4 |

Seconds per tile. **7.63x faster.** Note rows two and three: villa's current
autocast line times identically to autocast being switched off, which is the
same no-op shown numerically below. The memory number matters more than the speed one. The CPU
path costs about **9.9 GB per tile** (9.52 GB at batch 1, 19.89 GB at batch 2),
so on a 16 GB Mac one tile barely fits and two do not. MPS at batch 4 needs
7.85 GB. For most Macs this is the difference between running the inference and
not running it.

Each configuration was measured in a **fresh process**, because `ru_maxrss` is
a process-wide high-water mark and a single-process sweep reports the maximum
of everything that ran before it. A first attempt did exactly that and its
memory column was meaningless.

### Why the CPU path is so expensive

Sampling the CPU run shows the time going into
`at::native::slow_conv3d_forward_out_cpu` and `cpublas_gemm_impl`. PyTorch has
no optimised 3D convolution kernel for this case on arm64, so it falls back to
the reference implementation. The patch does not fix that; it is the reason the
MPS path is worth having.

### A separate finding, worth its own look: CPU autocast is catastrophic on arm64

This one is not about Apple Silicon specifically and the patch deliberately
does **not** change it, because changing CPU behaviour would alter what every
existing CPU user gets.

`inference.py` wraps the forward in `torch.autocast(device_type=amp_device,
enabled=True)`, and on any non-CUDA device `amp_device` is `"cpu"`. On arm64
that routes the 3D convolutions into a **bfloat16** reference kernel:

| CPU forward, one tile, batch 1 | seconds |
|---|---|
| fp32, autocast off | **3.47** |
| bf16, via `torch.autocast(device_type="cpu")` | **killed at 714, still running** |

That is a penalty of **at least 206x**, and it is a lower bound rather than a
measurement because the run was killed rather than completed. Attributed by
sampling, not assumed: the stack shows `BFloat16` symbols above
`at::native::slow_conv3d_forward_out_cpu` and `cpublas_gemm_impl`, so bf16 uses
the same reference convolution with no optimised GEMM behind it.

The practical effect is that a 49-tile crop that takes 21 s on MPS did not
finish in 33 minutes on CPU. Anyone running this on an arm64 CPU is likely to
read that as a hang. Whether the right fix is to disable autocast on CPU, gate
it on architecture, or leave it is a call for the maintainers, so this is
reported rather than patched.

### The autocast line is a silent no-op, not just mis-specified

Running autocast with `device_type="cpu"` while the tensors are on MPS produces
output **bit-identical to fp32**, max absolute difference exactly `0.0`. With
`device_type="mps"` the output differs by `1.24e-03`, which is fp16 magnitude,
and it is 1.23x faster. So Apple Silicon users are not getting a slightly wrong
autocast, they are getting none at all, with no error and no warning.

## Correctness

Three separate questions, kept separate because they have different answers.

### 1. Does the patch change what the model computes? No.

Same fixed input, CPU against MPS, single forward:

| comparison | max absolute difference |
|---|---|
| CPU vs MPS, same input | **9.54e-07** |
| CPU vs MPS, DIFFERENT input (negative control) | 0.165 |

The negative control is there because an agreement of ~0 with nothing to
compare it against would only prove the model ignores its input. It
discriminates by five orders of magnitude.

### 2. How much does correcting `amp_device` change the real output?

This is the question `AGENTS.md` 1.3 asks, so it is measured end to end on a
real crop rather than on a single tensor. villa's full pipeline, same 1024x1024
crop, 49 tiles, run twice on MPS with only the autocast device type differing:

| | |
|---|---|
| wall clock, current `amp_device` | 25.37 s |
| wall clock, corrected | 20.84 s (**1.217x**) |
| Pearson between the two outputs | **0.9999995** |
| mean absolute difference | 1.37e-04 |
| 99th percentile absolute difference | 1.78e-03 |
| max absolute difference | 3.17e-03 |
| **pixels whose ink/no-ink decision at 0.5 flips** | **0.014%** |

So the precision relaxation is real and it is small: 138 pixels in a million
change side of the 0.5 threshold. That is the trade, stated so it can be judged
rather than assumed.

### 3. Does the output match villa's published prediction? Yes, at r = 0.99,
### once the depth window is right. That turned out to be the real finding.

Comparing to the published map for the same segment
(`...new_canon_autoresearch_recipe-tile256-stride128.tif`), which sits on the
same grid as the level-0 surface so crops map one to one, the agreement
depends almost entirely on `START_LAYER`:

| scroll, segment | r at START_LAYER **1** (this README's suggestion) | best START_LAYER | r there |
|---|---|---|---|
| PHerc. 1667, 20240304141531 | 0.8521 | 25 | **0.9851** |
| PHerc. 0139, 20250108000000 | 0.9054 | 23 | **0.9900** |
| PHercParis4, 20230702185753 | 0.6240 | 24 | **0.9698** |
| PHerc. 0814, 20250925161630 | 0.8263 | 21 | **0.9005** |

Four independent scrolls, each with its own published canonical prediction,
each with its ink-rich window chosen automatically from the reference rather
than by hand. All four volumes are 109 layers, so the centred window is
`(109-62)//2 = 23`. **Best START_LAYER is 21, 23, 24, 25. START_LAYER 1 is
worse on every one**, by 0.07 to 0.35 in r.

Every offset on every scroll was also scored against a **shuffled** copy of
that scroll's reference. That floor never exceeds **0.0017**, so this is depth
alignment and not marginal statistics.

![agreement against START_LAYER](figures/start_layer_vs_reference.png)

**`START_LAYER=1` does not reproduce villa's own production output. About 21
to 25 does, and 23 is a reasonable single default for a 109-layer volume.**

### Adversarial checks on that finding

A peak alone is not evidence, so four things were checked that could each have
explained it away. All are in `results/stress_window.json`.

| check | question | result |
|---|---|---|
| S1 | is r merely tracking how much ink the output has? | **no.** r peaks at START_LAYER 25, mean ink at 19, separation at 28. They do not coincide |
| S2 | does it survive rank correlation? | **yes.** Spearman peaks at 22, invariant to any monotone recalibration |
| S3 | is the peak carried by only one side of the reference? | **no.** Scoring only the pixels the reference calls ink peaks at 25, the same place |
| S4 | shuffled-reference floor | 0.0013 |

S3 is the most direct of the four. On the pixels that actually contain ink,
agreement goes from **0.4993 at START_LAYER 1** to
**0.8592 at START_LAYER 25**.

Corroborated by a metric that never looks at the reference: the ink separation
of our own output (mean of pixels above 0.5 minus mean of those below) also
peaks in that region, 0.7210 at START_LAYER 28 against 0.6643 at START_LAYER 1.

### Why: villa's own documentation says these volumes are centred

`vesuvius/docs/ink_detection.md` states it twice, for these exact volumes:

> For a 2.399 um OME-Zarr, read XY pyramid level 2, **select the centered 84 Z
> planes**, and mean-pool every four planes to 21 slices

> **Labels occupy Z slice 32 of a 65-plane volume**

Slice 32 of 65 is the exact centre. So the writing surface sits at the middle
of these volumes, and a 62-plane inference window has to be centred on it.
For a 109-plane volume that is `START_LAYER = (109-62)//2 = 23`, which is
where the measurements land.

Put the other way: with `START_LAYER=1`, the surface at plane 54 falls at
position 53 of a 62-plane window, about 85% of the way through it, instead of
in the middle.

**A wrong turn worth recording, because the same mistake is easy to repeat.**
I first tried to locate the surface from the voxels directly, using in-plane
gradient energy and mean intensity, and concluded from their asymmetry that
the volume was not centred. **That test cannot work.** Carbon ink on
carbonised papyrus has almost no attenuation contrast, which is the premise of
the whole challenge, so intensity statistics locate bulk papyrus structure and
say nothing about where the writing surface is. The profile is in
`results/surface_depth.json` and it is a measurement of the wrong quantity.

### Stability

| | best START_LAYER |
|---|---|
| four scrolls, one region each | 25, 23, 24, 21 |
| PHerc. 1667, three further regions | 24, 24, 24 |

Seven independent measurements, all within 21 to 25, against a centred value
of 23.

![prediction vs reference](figures/prediction_vs_reference_big.png)

Left is the published prediction, middle is the same pixels through this patch
on MPS, right is the absolute difference. 4096x4096, 961 tiles, at the
corrected `START_LAYER=25`: **Pearson 0.9685, Spearman 0.7702**, against
**0.608 / 0.275** on the identical pixels at `START_LAYER=1`. Best negative
control 0.056, shuffled reference 0.000.

## Is it reading ink, or inventing it?

Same crop, same model, at the corrected window `START_LAYER=25`. Each null
destroys structure while preserving something stated, and `frac>0.5` is the
share of pixels called confident ink. Real input gives **0.4138**.

| input | frac>0.5 | Pearson vs the real prediction |
|---|---|---|
| **real data** | **0.4138** | 1.000 |
| `depth_shuffle` | 0.0000 | -0.069 |
| `per_column_shuffle` | 0.0000 | +0.031 |
| `single_layer_repeat` | 0.0000 | +0.008 |
| `voxel_shuffle` | 0.0000 | +0.006 |
| `phase_scramble` | 0.0112 | +0.244 |
| `depth_reverse` | 0.0131 | +0.313 |
| `depth_roll_half` | 0.1429 | +0.423 |

What each null preserves:

- `single_layer_repeat` every layer replaced by a copy of the middle one, so
  in-plane texture is perfect and **all** depth information is gone
- `depth_shuffle` layer order permuted, every voxel value kept
- `per_column_shuffle` independent depth permutation per column
- `voxel_shuffle` global histogram only
- `phase_scramble` amplitude spectrum kept, in-plane phase randomised
- `depth_reverse` everything kept except direction
- `depth_roll_half` cyclic depth shift, local adjacency mostly kept

**The decisive one is `single_layer_repeat`: perfect 2D texture, no depth
information, and the model calls no ink at all.** So it is not a 2D texture
reader, which is the simplest version of the hallucination worry and it is
ruled out. `depth_shuffle`, `per_column_shuffle` and `voxel_shuffle` are also
exactly zero.

`depth_reverse` collapsing to 0.0131 says the model is strongly
**directional**: flipping the depth axis, which preserves every other
property, nearly removes the detection.

### The window changes the hallucination rate, which makes this a diagnostic

Running the identical nulls at `START_LAYER=1` instead of 25:

| null | frac>0.5 at START_LAYER **25** | at START_LAYER **1** |
|---|---|---|
| `depth_shuffle` | **0.0000** | **0.0732**, and anti-correlated with the real prediction (-0.175) |
| `phase_scramble` | 0.0112 | 0.0084 |
| `voxel_shuffle` | 0.0000 | 0.0000 |

**Operating outside the right window does not merely lower agreement, it makes
the model assert confident ink on input whose depth ordering has been
destroyed.** That is a usable signal: if your nulls come back with confident
ink, suspect your window before you suspect the scan.

## inkdiag: which failure mode am I fighting?

The 2026 open problems put it directly: when a model shows no ink, "the right
conclusion is neither 'the scan failed' nor 'the model failed'", six
explanations stay open, and "better diagnostics matter just as much as better
models". `bench/inkdiag.py` measures two things that separate them.

**Sweep the window.** Slide the 62-plane input through depth and record the
output at each position. Then:

| what you see | what it means | what to do |
|---|---|---|
| peak at the centred window, high separation | real signal, correctly located | nothing |
| peak displaced by N planes from centred | the surface is mislocalised by about N | re-localise the surface |
| no confident ink at ANY window | no detectable signal here | scan or chemistry, not the model |
| confident ink only at wrong windows, none at the centred one | **false positives from misconfiguration** | fix the window before blaming the scan |

**Run the nulls.** Destroy structure while preserving stated properties and
see what survives. If confident ink survives `single_layer_repeat`, which
keeps in-plane texture perfectly and removes all depth information, the output
is 2D texture and not ink.

### The two signatures, measured on the same segment

PHerc. 1667, 20240304141531, two 1024 crops: one ink-rich (reference std
0.362) and one blank (reference std 0.0116).

| | ink region | blank region |
|---|---|---|
| confident ink at the centred window | **0.4138** | **0.0000** |
| confident ink at `START_LAYER=1` | 0.3323 | 0.0167 |
| separation at the centred window | 0.7144 | negative, nothing above 0.5 |
| best agreement with the published map | 0.9851 | 0.2660 |

The blank region produces **no** confident ink at the centred window and at
every window from 10 to 103. It produces 1.67% at `START_LAYER=1`.

**So the wrong window manufactures false positives twice over**: on blank
papyrus, and on structurally destroyed input (the `depth_shuffle` null goes
from 0.0000 at the centred window to 0.0732 at `START_LAYER=1`, anti-correlated
with the real prediction). Both point the same way, and neither would be
visible to someone looking only at their prediction image.

    python bench/inkdiag.py --y0 20480 --x0 16384 --size 1024 \
      --offsets=-1,0,9,18,23,25,28,34,46

## Also here: running it locally

`entrypoint.py` assumes `/workspace` and a credentialed `boto3` client, so it
cannot be run outside the container. `bench/run_local.py` reads a crop straight
from `s3://vesuvius-challenge-open-data/` anonymously and hands it to villa's
own `run_inference` unmodified, with no container and no credentials:

```bash
python bench/run_local.py --y0 20480 --x0 16384 --size 1024
```

Two notes for anyone reading zarr from that bucket:

- zarr 3 needs an **async** fsspec filesystem. Use
  `zarr.storage.FsspecStore.from_url("s3://...", storage_options={"anon": True})`;
  passing a plain `S3FileSystem` raises `Filesystem needs to support async
  operations`.
- `s3fs` must be recent. Installing without a version bound can resolve
  `s3fs==0.4.2`, which fails with
  `Session.__init__() got an unexpected keyword argument 'asynchronous'`.
  villa's own `requirements.txt` already pins `>=2024.3.0`; this is only a
  warning for anyone installing by hand.

## The patch

`device_utils.py` plus six changed lines, in `villa-apple-silicon.patch`.

- `select_device()`: cuda, then mps, then cpu, with an `INK_DEVICE` override.
- `amp_device_type(device)`: follows the device the tensors are actually on,
  and falls back to `cpu` for anything it does not recognise.
- `sync(device)` and `empty_cache(device)` so timing and cache clearing are not
  cuda-only.

## Controls

`bench/verify_port.py`, 12 checks, 0 failures. The check that matters is not
"does MPS work", it is **"is the CUDA path unchanged"**, since that is the path
every existing user is on.

**Stated plainly: there is no CUDA hardware here, so the CUDA path is verified
by unit test and by construction, never by execution.** `amp_device_type` is
asserted on every branch and `select_device` is tested with
`torch.cuda.is_available` monkeypatched to `True`. A positive control watches
villa's original expression returning `"cpu"` for an MPS device, so if that is
ever fixed upstream the control fails loudly instead of passing vacuously.

## Honest negatives

- **Batching buys almost nothing on MPS.** Throughput is flat from batch 1 to
  16 (2.33 to 2.39 tiles/s), so the model already saturates the GPU at batch 1.
  Batch 16 needs 27.08 GB of driver-allocated memory for a 1.25x gain. Batch 4
  is the sensible default.
- `torch.compile` does work on MPS in both `reduce-overhead` and `default`
  modes, warmup 14.1 s and 12.5 s, output matching eager to `1.36e-03`. It
  emits CUDA-specific warnings ("Not enough SMs", "skipping cudagraphs due to
  multiple devices") which are cosmetic.
- MPS refuses `float64`. Nothing in this path hits it, but anything that does
  will need casting.

## Everything here is machine-checked

`bench/verify_claims.py` re-derives every number quoted in this README from the
raw JSON in `results/` and fails if any of them drifts. **44 claims, 0
disagreements.** It also refuses em dashes and refuses any claim this work has
since withdrawn, so a superseded figure cannot survive an edit.

It has been watched failing: planting a stale speedup into this README makes it
exit 1 and name the two checks that catch it.

    python bench/verify_claims.py

## Reproducing

```bash
git clone https://github.com/ScrollPrize/villa.git
uv venv env --python 3.11
uv pip install --python ./env/bin/python torch numpy scipy tqdm \
  opencv-python-headless tifffile psutil huggingface_hub boto3 \
  "s3fs>=2025.1.0" fsspec zarr numcodecs imagecodecs albumentations matplotlib
git -C villa apply ../villa-apple-silicon.patch

./env/bin/python bench/feasibility.py      # what MPS supports, and the bucket
./env/bin/python bench/verify_port.py      # the 12 controls
./env/bin/python bench/bench_device.py     # CPU vs MPS, with a negative control
./env/bin/python bench/run_local.py        # a real crop, end to end
./env/bin/python bench/compare_reference.py --y0 20480 --x0 16384 --tag ink
```

Measured on an M1 Max, 64 GB, macOS 26.6, torch 2.14.0, python 3.11.15.

MIT licensed, same as villa.
