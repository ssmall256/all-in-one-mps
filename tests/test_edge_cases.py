from pathlib import Path
from types import SimpleNamespace

import torch

from allin1.postprocessing.functional import postprocess_functional_structure
from allin1.sonify import _sonify_metronome
from allin1.typings import AllInOneOutput, AnalysisResult, Segment


def test_postprocess_functional_structure_handles_no_boundaries():
  cfg = SimpleNamespace(
    min_hops_per_beat=24,
    fps=100,
    hop_size=441,
    sample_rate=44100,
  )
  logits = AllInOneOutput(
    logits_section=torch.full((1, 100), -100.0),
    logits_function=torch.zeros((1, 10, 100)),
  )

  segments = postprocess_functional_structure(logits, cfg)
  assert len(segments) >= 1
  assert segments[0].start == 0.0
  assert segments[-1].end > 0


def test_sonify_metronome_handles_empty_downbeats():
  result = AnalysisResult(
    path=Path('/tmp/test.wav'),
    bpm=120,
    beats=[1.0, 2.0, 3.0],
    downbeats=[],
    beat_positions=[1, 2, 3],
    segments=[Segment(start=0.0, end=4.0, label='intro')],
  )

  y = _sonify_metronome(result, length=44100)
  assert y.shape == (44100,)
