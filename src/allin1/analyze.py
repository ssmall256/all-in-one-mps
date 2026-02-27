import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import torch
from tqdm import tqdm

from .demix import demix
from .helpers import (
  check_paths,
  expand_paths,
  rmdir_if_empty,
  run_inference,
  save_results,
)
from .models import load_pretrained_model
from .sonify import sonify as _sonify
from .spectrogram import extract_spectrograms
from .timings_viz import visualize_timings
from .typings import AnalysisResult, PathLike
from .utils import load_result, mkpath
from .visualize import visualize as _visualize


def _run_torch_inference(
  todo_paths: List[Path],
  spec_paths_for_infer: List[Path],
  model_container: Dict[str, Any],
  model_loader_thread: Optional[threading.Thread],
  model_name: str,
  device: str,
  include_activations: bool,
  include_embeddings: bool,
  timings_embed: bool,
  out_dir: Optional[Path],
  _emit_timing,
) -> List[AnalysisResult]:
  """Run torch-based inference on audio tracks."""
  results = []

  if model_loader_thread is not None:
    model_loader_thread.join()
    if model_container['error'] is not None:
      raise model_container['error']
    if model_container['model'] is not None:
      model = model_container['model']
      t0, t1 = model_container['load_time']
      _emit_timing("model_load", None, t0, t1)
    else:
      t0 = time.perf_counter()
      model = load_pretrained_model(
        model_name=model_name,
        device=device,
      )
      t1 = time.perf_counter()
      _emit_timing("model_load", None, t0, t1)
  else:
    t0 = time.perf_counter()
    model = load_pretrained_model(
      model_name=model_name,
      device=device,
    )
    t1 = time.perf_counter()
    _emit_timing("model_load", None, t0, t1)

  with torch.no_grad():
    pbar = tqdm(zip(todo_paths, spec_paths_for_infer), total=len(todo_paths))
    for path, spec_path in pbar:
      pbar.set_description(f'Analyzing {path.name}')

      timings = {}
      result = run_inference(
        path=path,
        spec_path=spec_path,
        model=model,
        device=device,
        include_activations=include_activations,
        include_embeddings=include_embeddings,
        timings=timings,
      )
      if "nn" in timings:
        _emit_timing("nn", path, *timings["nn"])
      if "spec_load" in timings:
        _emit_timing("spec_load", path, *timings["spec_load"])
      if "postprocess" in timings:
        _emit_timing("postprocess", path, *timings["postprocess"])
      for key in (
        "metrical_prep",
        "metrical_dbn",
        "functional_probs",
        "functional_local_maxima",
        "functional_boundaries",
      ):
        if key in timings:
          _emit_timing(key, path, *timings[key])
      if timings_embed:
        result.activations = result.activations or {}
        result.activations["timings"] = {
          "nn": (timings["nn"][1] - timings["nn"][0]) if "nn" in timings else None,
          "postprocess": (timings["postprocess"][1] - timings["postprocess"][0]) if "postprocess" in timings else None,
        }

      if out_dir is not None:
        t0 = time.perf_counter()
        save_results(result, out_dir)
        t1 = time.perf_counter()
        _emit_timing("save", path, t0, t1)

      results.append(result)

  return results


def _parse_overwrite(overwrite: Union[bool, str, None]) -> Set[str]:
  if overwrite is True:
    return {"demix", "spec", "json", "viz", "sonify"}
  if not overwrite:
    return set()
  if isinstance(overwrite, str):
    value = overwrite.strip().lower()
    if value == "all":
      return {"demix", "spec", "json", "viz", "sonify"}
    stages = {part.strip() for part in value.split(",") if part.strip()}
    valid = {"demix", "spec", "json", "viz", "sonify"}
    unknown = stages - valid
    if unknown:
      raise ValueError(f"Unknown overwrite stage(s): {sorted(unknown)}")
    return stages
  return set()


def analyze(
  paths: Union[PathLike, List[PathLike]],
  out_dir: PathLike = None,
  visualize: Union[bool, PathLike] = False,
  sonify: Union[bool, PathLike] = False,
  model: str = 'harmonix-all',
  device: str = 'mps' if torch.backends.mps.is_available() else 'cpu',
  demucs_device: Optional[str] = None,
  include_activations: bool = False,
  include_embeddings: bool = False,
  demix_dir: PathLike = './demix',
  spec_dir: PathLike = './spec',
  spec_backend: Optional[str] = None,
  spec_torch_device: str = "cpu",
  spec_torch_dtype: str = "float32",
  keep_byproducts: bool = False,
  overwrite: Union[bool, str, None] = None,
  multiprocess: bool = True,
  timings_path: PathLike = None,
  timings_embed: bool = False,
  timings_viz_path: PathLike = None,
) -> Union[AnalysisResult, List[AnalysisResult]]:
  """
  Analyzes the provided audio files and returns the analysis results.
  """

  model_name = model
  if demucs_device is None:
    demucs_device = "mps" if torch.backends.mps.is_available() else "cpu"
  if spec_backend is None:
    spec_backend = "torch"
  if spec_backend != "torch":
    raise ValueError("spec_backend must be 'torch' for this MPS-only build.")
  overwrite_set = _parse_overwrite(overwrite)
  overwrite_demix = "demix" in overwrite_set
  overwrite_spec = "spec" in overwrite_set
  overwrite_json = "json" in overwrite_set
  overwrite_viz = "viz" in overwrite_set
  overwrite_sonify = "sonify" in overwrite_set
  demix_stage = "demix: demucs"
  return_list = True
  if not isinstance(paths, list):
    return_list = False
    paths = [paths]
  if not paths:
    raise ValueError('At least one path must be specified.')
  paths = [mkpath(p) for p in paths]
  paths = expand_paths(paths)
  check_paths(paths)

  temp_dir = None
  if not keep_byproducts:
    temp_dir = tempfile.mkdtemp(prefix='allin1_')
    demix_dir_actual = Path(temp_dir) / 'demix'
    spec_dir_actual = Path(temp_dir) / 'spec'
    demix_dir_actual.mkdir(parents=True, exist_ok=True)
    spec_dir_actual.mkdir(parents=True, exist_ok=True)
  else:
    demix_dir_actual = mkpath(demix_dir)
    spec_dir_actual = mkpath(spec_dir)

  timings_path = Path(timings_path) if timings_path is not None else None
  timings_handle = None
  if timings_path is not None:
    timings_path.parent.mkdir(parents=True, exist_ok=True)
    timings_handle = timings_path.open('a')

  def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
      return value
    if isinstance(value, Path):
      return str(value)
    try:
      return float(value)
    except Exception:
      return str(value)

  def _emit_timing(stage: str, track: Optional[Path], start: float, end: float, extra: Optional[Dict[str, Any]] = None):
    if timings_handle is None:
      return
    record: Dict[str, Any] = {
      "stage": stage,
      "start": float(start),
      "end": float(end),
      "duration": float(end - start),
      "device": _json_safe(device),
      "model": _json_safe(model_name),
    }
    if track is not None:
      record["track"] = str(track)
    if extra:
      record.update({k: _json_safe(v) for k, v in extra.items()})
    timings_handle.write(json.dumps(record) + "\n")
    timings_handle.flush()

  if out_dir is None or overwrite_json:
    todo_paths = paths
    exist_paths = []
  else:
    out_paths = [mkpath(out_dir) / path.with_suffix('.json').name for path in paths]
    todo_paths = [path for path, out_path in zip(paths, out_paths) if not out_path.exists()]
    exist_paths = [out_path for path, out_path in zip(paths, out_paths) if out_path.exists()]
  process_paths = paths if (overwrite_demix or overwrite_spec) else todo_paths

  print(f'=> Found {len(exist_paths)} tracks already analyzed and {len(todo_paths)} tracks to analyze.')
  if exist_paths:
    print('=> To re-analyze, please use --overwrite option.')

  results = []
  if exist_paths:
    results += [
      load_result(
        exist_path,
        load_activations=include_activations,
        load_embeddings=include_embeddings,
      )
      for exist_path in tqdm(exist_paths, desc='Loading existing results')
    ]

  demix_paths = []
  spec_paths = []
  spec_map = {}

  model_loader_thread = None
  model_container = {'model': None, 'error': None, 'load_time': None}

  if todo_paths:
    def load_model_background():
      try:
        t_load_start = time.perf_counter()
        model_container['model'] = load_pretrained_model(
          model_name=model,
          device=device,
        )
        t_load_end = time.perf_counter()
        model_container['load_time'] = (t_load_start, t_load_end)
      except Exception as e:
        model_container['error'] = e

    model_loader_thread = threading.Thread(target=load_model_background, daemon=True)
    model_loader_thread.start()

  if process_paths:
    t0 = time.perf_counter()
    demix_paths = demix(process_paths, demix_dir_actual, demucs_device, overwrite=overwrite_demix)
    t1 = time.perf_counter()
    _emit_timing(demix_stage, None, t0, t1, {"count": len(process_paths)})

    t0 = time.perf_counter()
    spec_paths = extract_spectrograms(
      demix_paths,
      spec_dir_actual,
      multiprocess,
      overwrite=overwrite_spec,
      backend=spec_backend,
      torch_device=spec_torch_device,
      torch_dtype=spec_torch_dtype,
      check=False,
    )
    t1 = time.perf_counter()
    _emit_timing("spectrogram", None, t0, t1, {"count": len(spec_paths)})
    spec_map = {demix_path.name: spec_path for demix_path, spec_path in zip(demix_paths, spec_paths)}

  if todo_paths:
    spec_paths_for_infer = [spec_map[path.stem] for path in todo_paths]
    new_results = _run_torch_inference(
      todo_paths=todo_paths,
      spec_paths_for_infer=spec_paths_for_infer,
      model_container=model_container,
      model_loader_thread=model_loader_thread,
      model_name=model_name,
      device=device,
      include_activations=include_activations,
      include_embeddings=include_embeddings,
      timings_embed=timings_embed,
      out_dir=out_dir,
      _emit_timing=_emit_timing,
    )
    results.extend(new_results)

  results = sorted(results, key=lambda result: paths.index(result.path))

  if visualize:
    t0 = time.perf_counter()
    if visualize is True:
      visualize = './viz'
    viz_results = results
    t_prep0 = time.perf_counter()
    if not overwrite_viz and visualize is not None:
      viz_dir = mkpath(visualize)
      viz_results = [
        result for result in results
        if not (viz_dir / f"{result.path.stem}.pdf").is_file()
      ]
    t_prep1 = time.perf_counter()
    _emit_timing("visualize_prep", None, t_prep0, t_prep1)
    _visualize(viz_results, out_dir=visualize, multiprocess=multiprocess)
    t1 = time.perf_counter()
    _emit_timing("visualize", None, t0, t1)
    print(f'=> Plots are successfully saved to {visualize}')

  if sonify:
    t0 = time.perf_counter()
    if sonify is True:
      sonify = './sonif'
    sonif_results = results
    if not overwrite_sonify and sonify is not None:
      sonif_dir = mkpath(sonify)
      sonif_results = [
        result for result in results
        if not (sonif_dir / f"{result.path.stem}.sonif{result.path.suffix}").is_file()
      ]
    _sonify(sonif_results, out_dir=sonify, multiprocess=multiprocess)
    t1 = time.perf_counter()
    _emit_timing("sonify", None, t0, t1)
    print(f'=> Sonified tracks are successfully saved to {sonify}')

  if not keep_byproducts and temp_dir is not None:
    shutil.rmtree(temp_dir, ignore_errors=True)
  elif not keep_byproducts:
    if overwrite_demix or overwrite_spec:
      for path in demix_paths:
        for stem in ['bass', 'drums', 'other', 'vocals']:
          (path / f'{stem}.wav').unlink(missing_ok=True)
        rmdir_if_empty(path)
      rmdir_if_empty(demix_dir_actual / 'htdemucs')
      rmdir_if_empty(demix_dir_actual)

      for path in spec_paths:
        path.unlink(missing_ok=True)
      rmdir_if_empty(spec_dir_actual)

  if timings_handle is not None:
    timings_handle.close()
  if timings_viz_path is not None:
    if timings_path is None:
      raise ValueError("timings_viz_path requires timings_path to be set.")
    visualize_timings(Path(timings_path), Path(timings_viz_path))
  if not return_list:
    return results[0]
  return results
