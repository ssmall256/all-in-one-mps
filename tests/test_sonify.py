from pathlib import Path

import pytest

import allin1

CWD = Path(__file__).resolve().parent
TEST_JSON = CWD / 'test.json'


@pytest.mark.skipif(not TEST_JSON.is_file(), reason='test.json fixture not found')
def test_sonify():
  result = allin1.AnalysisResult.from_json(TEST_JSON)
  allin1.sonify(result)


@pytest.mark.skipif(not TEST_JSON.is_file(), reason='test.json fixture not found')
def test_sonify_save():
  result = allin1.AnalysisResult.from_json(TEST_JSON)
  allin1.sonify(
    result,
    out_dir='./sonif'
  )
