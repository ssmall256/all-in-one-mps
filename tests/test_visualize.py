from pathlib import Path

import pytest

import allin1
from allin1.utils import mkpath

CWD = Path(__file__).resolve().parent
TEST_MP3 = CWD / 'test.mp3'


@pytest.mark.skipif(not TEST_MP3.is_file(), reason='test.mp3 fixture not found')
def test_visualize():
  result = allin1.analyze(
    paths=TEST_MP3,
    keep_byproducts=True,
  )
  allin1.visualize(result)


@pytest.mark.skipif(not TEST_MP3.is_file(), reason='test.mp3 fixture not found')
def test_visualize_save():
  result = allin1.analyze(
    paths=TEST_MP3,
    keep_byproducts=True,
  )
  allin1.visualize(
    result,
    out_dir='./viz',
  )
  assert mkpath('./viz/test.pdf').is_file()
