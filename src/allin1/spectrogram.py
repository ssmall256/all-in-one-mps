from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

# Madmom backend is deprecated - kept only for backwards compatibility
# Torch backend is the default for the MPS build

_SPEC_FRAME_SIZE = 2048
_SPEC_FPS = int(44100 / 441)
_SPEC_NUM_BANDS = 12
_SPEC_FMIN = 30.0
_SPEC_FMAX = 17000.0
_SPEC_FREF = 440.0

# Cache for window and filterbank matrices (computed once per sample rate)
_WINDOW_CACHE = {}
_FILTERBANK_CACHE = {}


def _madmom_processor():
  from madmom.audio.signal import FramedSignalProcessor
  from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
  from madmom.audio.stft import ShortTimeFourierTransformProcessor
  from madmom.processors import SequentialProcessor

  frames = FramedSignalProcessor(
    frame_size=_SPEC_FRAME_SIZE,
    fps=_SPEC_FPS,
  )
  stft = ShortTimeFourierTransformProcessor()
  filt = FilteredSpectrogramProcessor(
    num_bands=_SPEC_NUM_BANDS,
    fmin=_SPEC_FMIN,
    fmax=_SPEC_FMAX,
    norm_filters=True,
  )
  spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
  return SequentialProcessor([frames, stft, filt, spec])


def _load_wave_mono(path: Path) -> Tuple[np.ndarray, int]:
  from scipy.io import wavfile

  sr, data = wavfile.read(str(path), mmap=True)
  if data.ndim == 1:
    return data, sr
  if data.shape[1] == 1:
    return data[:, 0], sr
  if np.issubdtype(data.dtype, np.integer):
    mono = np.mean(data.astype(np.float64), axis=1)
    return mono.astype(data.dtype), sr
  return np.mean(data, axis=1).astype(data.dtype), sr


def _signal_frame(signal: np.ndarray, index: int, frame_size: int, hop_size: float, origin: int = 0) -> np.ndarray:
  frame_size = int(frame_size)
  num_samples = len(signal)
  ref_sample = int(index * hop_size)
  start = ref_sample - frame_size // 2 - int(origin)
  stop = start + frame_size

  if start >= 0 and stop <= num_samples:
    return signal[start:stop]

  frame = np.repeat(signal[:1], frame_size, axis=0)
  left, right = 0, 0
  if start < 0:
    left = min(stop, 0) - start
    frame[:left] = 0
    start = 0
  if stop > num_samples:
    right = stop - max(start, num_samples)
    frame[-right:] = 0
    stop = num_samples
  frame[left:frame_size - right] = signal[min(start, num_samples):max(stop, 0)]
  return frame


def _frame_signal(signal: np.ndarray, frame_size: int, hop_size: float) -> np.ndarray:
  num_frames = int(np.ceil(len(signal) / float(hop_size)))
  hop_round = int(round(hop_size))
  if abs(hop_size - hop_round) < 1e-6 and hop_round > 0:
    left_pad = frame_size // 2
    right_pad = frame_size // 2 + hop_round
    padded = np.zeros(len(signal) + left_pad + right_pad, dtype=signal.dtype)
    padded[left_pad:left_pad + len(signal)] = signal
    strides = (padded.strides[0] * hop_round, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(
      padded,
      shape=(num_frames, frame_size),
      strides=strides,
      writeable=False,
    )
    return frames.copy()

  frames = np.empty((num_frames, frame_size), dtype=signal.dtype)
  for idx in range(num_frames):
    frames[idx] = _signal_frame(signal, idx, frame_size, hop_size, origin=0)
  return frames


def _fft_bin_frequencies(num_bins: int, sample_rate: float) -> np.ndarray:
  return np.fft.fftfreq(num_bins * 2, 1.0 / sample_rate)[:num_bins]


def _log_frequencies(bands_per_octave: int, fmin: float, fmax: float, fref: float) -> np.ndarray:
  left = np.floor(np.log2(float(fmin) / fref) * bands_per_octave)
  right = np.ceil(np.log2(float(fmax) / fref) * bands_per_octave)
  frequencies = fref * 2.0 ** (np.arange(left, right) / float(bands_per_octave))
  frequencies = frequencies[np.searchsorted(frequencies, fmin):]
  frequencies = frequencies[:np.searchsorted(frequencies, fmax, 'right')]
  return frequencies


def _frequencies_to_bins(
  frequencies: np.ndarray,
  bin_frequencies: np.ndarray,
  unique_bins: bool,
) -> np.ndarray:
  frequencies = np.asarray(frequencies)
  bin_frequencies = np.asarray(bin_frequencies)
  indices = bin_frequencies.searchsorted(frequencies)
  indices = np.clip(indices, 1, len(bin_frequencies) - 1)
  left = bin_frequencies[indices - 1]
  right = bin_frequencies[indices]
  indices -= frequencies - left < right - frequencies
  if unique_bins:
    indices = np.unique(indices)
  return indices


def _triangular_filters(bins: np.ndarray, norm: bool) -> List[Tuple[np.ndarray, int]]:
  if len(bins) < 3:
    raise ValueError("not enough bins to create a triangular filterbank")
  filters: List[Tuple[np.ndarray, int]] = []
  for idx in range(len(bins) - 2):
    start, center, stop = bins[idx:idx + 3]
    if stop - start < 2:
      center = start
      stop = start + 1
    center = int(center)
    start = int(start)
    stop = int(stop)
    rel_center = center - start
    rel_stop = stop - start
    data = np.zeros(rel_stop, dtype=np.float32)
    if rel_center > 0:
      data[:rel_center] = np.linspace(0, 1, rel_center, endpoint=False)
    data[rel_center:] = np.linspace(1, 0, rel_stop - rel_center, endpoint=False)
    if norm:
      data = data / np.sum(data)
    filters.append((data, start))
  return filters


def _log_filterbank_matrix(
  num_fft_bins: int,
  sample_rate: int,
  num_bands: int,
  fmin: float,
  fmax: float,
  fref: float,
  norm_filters: bool,
  unique_filters: bool,
) -> np.ndarray:
  bin_freqs = _fft_bin_frequencies(num_fft_bins, sample_rate)
  frequencies = _log_frequencies(num_bands, fmin, fmax, fref)
  bins = _frequencies_to_bins(frequencies, bin_freqs, unique_bins=unique_filters)
  filters = _triangular_filters(bins, norm=norm_filters)
  fb = np.zeros((len(bin_freqs), len(filters)), dtype=np.float32)
  for band_idx, (filt, start) in enumerate(filters):
    stop = start + len(filt)
    if start < 0:
      filt = filt[-start:]
      start = 0
    if stop > len(bin_freqs):
      filt = filt[:-(stop - len(bin_freqs))]
      stop = len(bin_freqs)
    if start >= stop:
      continue
    band = fb[start:stop, band_idx]
    np.maximum(filt, band, out=band)
  return fb


def _torch_log_spectrogram(
  signal: np.ndarray,
  sample_rate: int,
  device: str,
  dtype: str,
) -> np.ndarray:
  import torch

  frame_size = _SPEC_FRAME_SIZE
  hop_size = sample_rate / float(_SPEC_FPS)
  frames = _frame_signal(signal, frame_size, hop_size)

  window = _get_cached_window(sample_rate, signal.dtype)
  fb = _get_cached_filterbank(sample_rate)

  frames_t = torch.tensor(frames, device=device, dtype=getattr(torch, dtype))
  window_t = torch.tensor(window, device=device, dtype=frames_t.dtype)
  fft_in = frames_t * window_t
  stft = torch.fft.fft(fft_in, n=frame_size, dim=1)[:, :frame_size >> 1]
  mag = stft.abs()
  fb_t = torch.tensor(fb, device=device, dtype=mag.dtype)
  filtered = mag @ fb_t
  logged = torch.log10(filtered + 1.0)
  return logged.detach().cpu().numpy()


def _get_cached_window(sample_rate: int, signal_dtype) -> np.ndarray:
  key = (sample_rate, signal_dtype)
  if key not in _WINDOW_CACHE:
    window = np.hanning(_SPEC_FRAME_SIZE).astype(np.float32)
    if np.issubdtype(signal_dtype, np.integer):
      window = window / float(np.iinfo(signal_dtype).max)
    _WINDOW_CACHE[key] = window
  return _WINDOW_CACHE[key]


def _get_cached_filterbank(sample_rate: int) -> np.ndarray:
  if sample_rate not in _FILTERBANK_CACHE:
    num_fft_bins = _SPEC_FRAME_SIZE >> 1
    fb = _log_filterbank_matrix(
      num_fft_bins=num_fft_bins,
      sample_rate=sample_rate,
      num_bands=_SPEC_NUM_BANDS,
      fmin=_SPEC_FMIN,
      fmax=_SPEC_FMAX,
      fref=_SPEC_FREF,
      norm_filters=True,
      unique_filters=True,
    )
    _FILTERBANK_CACHE[sample_rate] = fb
  return _FILTERBANK_CACHE[sample_rate]


def _to_mono_signal(signal: np.ndarray) -> np.ndarray:
  if signal.ndim == 1:
    return signal
  if signal.ndim == 2:
    if signal.shape[0] in (1, 2) and signal.shape[1] > signal.shape[0]:
      return signal.mean(axis=0)
    return signal.mean(axis=1)
  raise ValueError(f"Expected 1D or 2D audio array, got shape {signal.shape}.")


def spectrogram_from_stems(
  stems: dict,
  sample_rate: int,
  backend: str = "torch",
  torch_device: str = "cpu",
  torch_dtype: str = "float32",
  check: bool = False,
) -> np.ndarray:
  if check:
    raise ValueError("check is not supported for the torch-only spectrogram backend.")

  stem_order = ['bass', 'drums', 'other', 'vocals']
  signals = []
  for stem in stem_order:
    if stem not in stems:
      raise KeyError(f"Missing stem '{stem}' in stems dict.")
    signals.append(_to_mono_signal(stems[stem]))

  if backend != "torch":
    raise ValueError("This build supports only the torch spectrogram backend.")

  specs = [
    _torch_log_spectrogram(signal, sample_rate, torch_device, torch_dtype)
    for signal in signals
  ]
  return np.stack(specs)


def extract_spectrograms(
  demix_paths: List[Path],
  spec_dir: Path,
  multiprocess: bool = True,
  overwrite: bool = False,
  backend: str = "torch",
  torch_device: str = "cpu",
  torch_dtype: str = "float32",
  check: bool = False,
):
  if check:
    raise ValueError("check is not supported for the torch-only spectrogram backend.")

  todos = []
  spec_paths = []
  for src in demix_paths:
    dst = spec_dir / f'{src.name}.npy'
    spec_paths.append(dst)
    if dst.is_file():
      if overwrite:
        dst.unlink(missing_ok=True)
      else:
        continue
    todos.append((src, dst))

  if overwrite:
    existing = 0
  else:
    existing = len(spec_paths) - len(todos)
  print(f'=> Found {existing} spectrograms already extracted, {len(todos)} to extract.')

  if todos:
    if backend == "madmom":
      processor = _madmom_processor()
      if multiprocess:
        pool = Pool()
        map_fn = pool.imap
      else:
        pool = None
        map_fn = map

      iterator = map_fn(_extract_spectrogram_madmom, [
        (src, dst, processor)
        for src, dst in todos
      ])
      for _ in tqdm(iterator, total=len(todos), desc='Extracting spectrograms'):
        pass

      if pool:
        pool.close()
        pool.join()
    else:
      if backend != "torch":
        raise ValueError("This build supports only the torch spectrogram backend.")
      if multiprocess:
        print("=> Multiprocessing disabled for torch spectrogram backend.")
      iterator = map(_extract_spectrogram_torch, [
        (src, dst, torch_device, torch_dtype)
        for src, dst in todos
      ])
      for _ in tqdm(iterator, total=len(todos), desc='Extracting spectrograms'):
        pass

  return spec_paths


def _extract_spectrogram_madmom(args):
  from madmom.audio.signal import Signal

  src, dst, processor = args

  dst.parent.mkdir(parents=True, exist_ok=True)

  sig_bass = Signal(src / 'bass.wav', num_channels=1)
  sig_drums = Signal(src / 'drums.wav', num_channels=1)
  sig_other = Signal(src / 'other.wav', num_channels=1)
  sig_vocals = Signal(src / 'vocals.wav', num_channels=1)

  spec_bass = processor(sig_bass)
  spec_drums = processor(sig_drums)
  spec_others = processor(sig_other)
  spec_vocals = processor(sig_vocals)

  spec = np.stack([spec_bass, spec_drums, spec_others, spec_vocals])

  np.save(str(dst), spec)


def _extract_spectrogram_torch(args: Tuple[Path, Path, str, str]):
  src, dst, torch_device, torch_dtype = args

  dst.parent.mkdir(parents=True, exist_ok=True)

  stems = []
  stem_names = ['bass', 'drums', 'other', 'vocals']
  for stem in stem_names:
    signal, sr = _load_wave_mono(src / f'{stem}.wav')
    stems.append((signal, sr))

  specs = [
    _torch_log_spectrogram(signal, sr, torch_device, torch_dtype)
    for signal, sr in stems
  ]
  spec = np.stack(specs)
  np.save(str(dst), spec)
