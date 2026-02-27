# all-in-one-mps

**GPU-accelerated music structure analysis on Apple Silicon (PyTorch/MPS).**

`all-in-one-mps` is an Apple Silicon-optimized port of the original **All-In-One Music Structure Analyzer** (upstream: [`mir-aidj/all-in-one`](https://github.com/mir-aidj/all-in-one)). It runs end-to-end inference locally using **PyTorch with Metal Performance Shaders (MPS)**, with an integrated pipeline designed for real songs (including demixing + spectrograms).

Given one or more audio tracks, it produces:

- **Tempo (BPM)**
- **Beat times**
- **Downbeat times**
- **Section boundaries**
- **Section labels** (intro / verse / chorus / bridge / outro, etc.)

---

## Why this repo exists

The upstream project is a strong reference implementation, but macOS Apple Silicon users historically lacked a first-class GPU-accelerated inference path. This repository provides that acceleration via **PyTorch MPS**, with an emphasis on:

- **High performance** on M-series GPUs via Metal Performance Shaders
- **Practical CLI defaults** for song inference (demix -> spectrogram -> model -> outputs)
- **Faithful behavior** to the upstream model + method

I'm releasing **all-in-one-mps** alongside [**all-in-one-mlx**](https://github.com/ssmall256/all-in-one-mlx) (MLX acceleration) so Apple Silicon users can choose the stack that fits their environment.

---

## Performance

Benchmark on a single file — Apple M4 Max, 128 GB RAM, macOS 26.3:

| Project | Time | vs upstream |
|---|---|---|
| [`mir-aidj/all-in-one`](https://github.com/mir-aidj/all-in-one) | 75.25s | baseline |
| [`mir-aidj/all-in-one`](https://github.com/mir-aidj/all-in-one) | 24.63s | ~3.1x faster *(patched to use MPS)* |
| **`all-in-one-mps` (this repo)** | **13.43s** | **~5.6x faster** |
| [`all-in-one-mlx`](https://github.com/ssmall256/all-in-one-mlx) | 5.96s | ~12.6x faster |

> One run, one file — results will vary by hardware and content.

---

## Related projects & attribution

| Project | Purpose |
|---|---|
| [`mir-aidj/all-in-one`](https://github.com/mir-aidj/all-in-one) | Original reference implementation and training code |
| [`all-in-one-mps`](https://github.com/ssmall256/all-in-one-mps) | This repo: PyTorch/MPS inference + packaging for Apple Silicon |
| [`all-in-one-mlx`](https://github.com/ssmall256/all-in-one-mlx) | Companion repo: MLX inference for Apple Silicon |

This repository began as a fork/port of the upstream project. The original method/model is described in:

- Taejun Kim & Juhan Nam, *All-In-One Metrical and Functional Structure Analysis with Neighborhood Attentions on Demixed Audio* ([arXiv:2307.16425](https://arxiv.org/abs/2307.16425))

If you use this in academic work, please cite the paper and the [upstream repository](https://github.com/mir-aidj/all-in-one).

---

## Requirements

| Component | Requirement |
|---|---|
| Hardware | Apple Silicon (M-series) |
| OS | macOS 13+ |
| Python | 3.10+ |

> Need CUDA / Linux / Windows? Use the [upstream project](https://github.com/mir-aidj/all-in-one).

---

## Installation

### pip

```bash
pip install allin1
```

### uv (recommended)

```bash
uv pip install allin1
```

---

## Quickstart

Analyze one or more tracks:

```bash
allin1 path/to/song.wav
# or multiple:
allin1 path/to/a.wav path/to/b.wav
```

By default, results are written under:

- `./struct` (set with `--out-dir`)

### Common options

- Choose output directory:

```bash
allin1 song.wav --out-dir ./struct
```

- Save visualizations / sonifications:

```bash
allin1 song.wav --visualize --viz-dir ./viz
allin1 song.wav --sonify --sonif-dir ./sonif
```

- Keep intermediate byproducts (demixed audio + spectrograms):

```bash
allin1 song.wav --keep-byproducts
# demix files: ./demix (override with --demix-dir)
# specs:       ./spec  (override with --spec-dir)
```

- Overwrite specific stages (demix,spec,json,viz,sonify) or everything:

```bash
allin1 song.wav --overwrite all
allin1 song.wav --overwrite demix,spec,json
```

- Timing / performance instrumentation:

```bash
allin1 song.wav --timings-path timings.jsonl
allin1 song.wav --timings-path timings.jsonl --timings-viz-path timings.png
```

---

## MPS inference controls

- Select model (pretrained name):

```bash
allin1 song.wav --model harmonix-all
```

- Select device:

```bash
allin1 song.wav --device mps   # default when available
allin1 song.wav --device cpu   # fallback
```

- Disable multiprocessing (debug / determinism / constrained envs):

```bash
allin1 song.wav --no-multiprocess
```

- Prefer standard `natten` backend over `natten-mps`:

```bash
NATTEN_MPS=0 allin1 song.wav
```

If `natten` is not installed, this build can still fall back to `natten-mps`.

---

## Outputs

The CLI writes analysis artifacts under `--out-dir` (default `./struct`). Each track produces a JSON file containing tempo, beats, downbeats, beat positions, and segment boundaries/labels.

Optional outputs:

| Artifact | Enable with |
|---|---|
| Visualizations | `--visualize` and `--viz-dir` |
| Sonifications | `--sonify` and `--sonif-dir` |
| Frame-level activations | `--activ` |
| Frame-level embeddings | `--embed` |
| JSONL timings | `--timings-path` |

## Known limitations

- Artifact naming uses input basename/stem for intermediate and output files.
- If multiple inputs share the same basename (for example `a/song.mp3` and `b/song.wav`), artifacts may overwrite each other or be reused unexpectedly.
- Workaround: process those files separately or rename files so basenames are unique.

---

## Python API

```python
import allin1

result = allin1.analyze('song.wav')
# result.bpm, result.beats, result.downbeats, result.segments

results = allin1.analyze(['a.wav', 'b.wav'])
```

---

## License

This project retains the upstream license (MIT). See `LICENSE`.

---

## Issues

Please include:

| Include | Example |
|---|---|
| macOS version + Apple Silicon model | `macOS 26.3, M4 Max` |
| Python + PyTorch versions | `Python 3.12.7, torch 2.x` |
| Exact command and logs/traceback | Full `allin1 ...` command + stack trace |
