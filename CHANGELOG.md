# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-02-27

### Added

- PyTorch/MPS GPU-accelerated inference for Apple Silicon (M1/M2/M3/M4).
- `natten-mps` Metal kernels for neighborhood attention (1D and 2D).
- In-process demucs source separation with deterministic output (seeded random shift + 16-bit PCM saving via soundfile).
- CLI entry point `allin1`.
- Selective stage overwrite (`--overwrite`).
- JSONL timing output (`--timings-path`).
- Visualization and sonification outputs.
- 2-decimal JSON float formatting for clean output.

### Changed

- Renamed package from upstream `allin1` to MPS-focused distribution.
- Requires Python 3.10+ and macOS on Apple Silicon.
- Default device is `mps` when available.
- Spectrogram backend uses torch by default.

### Removed

- CUDA and CPU-only inference paths.
- Training code and instructions (refer to upstream).
- madmom as a required dependency.

[unreleased]: https://github.com/ssmall256/all-in-one-mps/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ssmall256/all-in-one-mps/releases/tag/v1.0.0
