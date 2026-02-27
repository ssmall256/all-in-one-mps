import random
import shutil
from pathlib import Path
from typing import List, Union

import torch


def _run_demucs_inprocess(
  path: Path,
  out_dir: Path,
  device: Union[str, torch.device],
):
  """Run demucs source separation in-process with deterministic output.

  Uses soundfile for 16-bit PCM saving to avoid torchaudio/torchcodec
  non-determinism, and seeds the random module to make the shift trick
  reproducible.
  """
  import soundfile as sf
  from demucs.apply import apply_model
  from demucs.audio import AudioFile, prevent_clip
  from demucs.pretrained import get_model

  model = get_model('htdemucs')
  model.cpu()
  model.eval()

  wav = AudioFile(path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)

  ref = wav.mean(0)
  wav -= ref.mean()
  wav /= ref.std()

  # Seed random for deterministic shift trick.
  random.seed(0)

  sources = apply_model(
    model,
    wav[None],
    device=str(device),
    shifts=1,
    split=True,
    overlap=0.25,
    progress=True,
  )[0]
  sources *= ref.std()
  sources += ref.mean()

  # Clamp to prevent clipping (same as demucs default 'rescale' mode).
  sources = prevent_clip(sources, mode='rescale')

  out_dir.mkdir(parents=True, exist_ok=True)
  for source, name in zip(sources, model.sources):
    # Save as 16-bit PCM WAV using soundfile for deterministic output.
    audio = source.cpu().numpy().T  # (channels, samples) -> (samples, channels)
    sf.write(out_dir / f'{name}.wav', audio, model.samplerate, subtype='PCM_16')


def demix(
  paths: List[Path],
  demix_dir: Path,
  demucs_device: Union[str, torch.device],
  overwrite: bool = False,
):
  """Demixes the audio file into its sources (torch-only)."""
  todos = []
  demix_paths = []
  for path in paths:
    out_dir = demix_dir / 'htdemucs' / path.stem
    demix_paths.append(out_dir)
    if out_dir.is_dir():
      if overwrite:
        shutil.rmtree(out_dir, ignore_errors=True)
        todos.append(path)
        continue
      if (
        (out_dir / 'bass.wav').is_file() and
        (out_dir / 'drums.wav').is_file() and
        (out_dir / 'other.wav').is_file() and
        (out_dir / 'vocals.wav').is_file()
      ):
        continue
    todos.append(path)

  if overwrite:
    existing = 0
  else:
    existing = len(paths) - len(todos)
  print(f'=> Found {existing} tracks already demixed, {len(todos)} to demix.')

  if todos:
    if demucs_device == 'mps' and not torch.backends.mps.is_available():
      demucs_device = 'cpu'

    for path in todos:
      out_dir = demix_dir / 'htdemucs' / path.stem
      _run_demucs_inprocess(path, out_dir, demucs_device)

  return demix_paths
