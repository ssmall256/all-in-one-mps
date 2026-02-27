from pathlib import Path

import pytest

import allin1

CWD = Path(__file__).resolve().parent
TEST_MP3 = CWD / 'test.mp3'


@pytest.mark.skipif(not TEST_MP3.is_file(), reason='test.mp3 fixture not found')
def test_analyze():
  result = allin1.analyze(TEST_MP3)
  assert isinstance(result, allin1.AnalysisResult)
  assert isinstance(result.bpm, int)
  assert len(result.beats) > 0
  assert len(result.segments) > 0
