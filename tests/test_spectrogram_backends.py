from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from scipy.io import wavfile

from allin1.spectrogram import extract_spectrograms

_has_madmom = False
try:
  import madmom  # noqa: F401
  _has_madmom = True
except ImportError:
  pass

def _write_stem(path: Path, sr: int, freq: float) -> None:
  t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
  signal = 0.2 * np.sin(2 * np.pi * freq * t)
  data = (signal * np.iinfo(np.int16).max).astype(np.int16)
  wavfile.write(str(path), sr, data)


def _make_demix_dir(root: Path) -> Path:
  track_dir = root / "htdemucs" / "track1"
  track_dir.mkdir(parents=True, exist_ok=True)
  sr = 44100
  _write_stem(track_dir / "bass.wav", sr, 110.0)
  _write_stem(track_dir / "drums.wav", sr, 220.0)
  _write_stem(track_dir / "other.wav", sr, 330.0)
  _write_stem(track_dir / "vocals.wav", sr, 440.0)
  return track_dir


def _load_spec(path: Path) -> np.ndarray:
  return np.load(str(path))


def _compare_specs(spec_a: np.ndarray, spec_b: np.ndarray, tol: float = 1e-4) -> None:
  assert spec_a.shape == spec_b.shape
  diff = np.abs(spec_a - spec_b)
  assert float(diff.max()) <= tol


@pytest.mark.skipif(not _has_madmom, reason='madmom not installed')
def test_spectrogram_torch_parity():
  with TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    demix_dir = tmp_path / "demix"
    track_dir = _make_demix_dir(demix_dir)

    madmom_dir = tmp_path / "spec_madmom"
    torch_dir = tmp_path / "spec_torch"

    madmom_paths = extract_spectrograms(
      [track_dir],
      madmom_dir,
      multiprocess=False,
      backend="madmom",
    )
    torch_paths = extract_spectrograms(
      [track_dir],
      torch_dir,
      multiprocess=False,
      backend="torch",
      torch_device="cpu",
      torch_dtype="float32",
    )

    spec_madmom = _load_spec(madmom_paths[0])
    spec_torch = _load_spec(torch_paths[0])
    _compare_specs(spec_madmom, spec_torch)


def test_spectrogram_mlx_backend_not_supported():
  with TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    demix_dir = tmp_path / "demix"
    track_dir = _make_demix_dir(demix_dir)

    mlx_dir = tmp_path / "spec_mlx"

    with pytest.raises(ValueError, match="torch spectrogram backend"):
      extract_spectrograms(
        [track_dir],
        mlx_dir,
        multiprocess=False,
        backend="mlx",
      )
