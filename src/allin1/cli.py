import argparse
from pathlib import Path

import torch

from .analyze import analyze


def make_parser():
  cwd = Path.cwd()
  parser = argparse.ArgumentParser()
  parser.add_argument('paths', nargs='+', type=Path, default=[], help='Path to tracks')
  parser.add_argument('-o', '--out-dir', type=Path, default=cwd / './struct',
                      help='Path to a directory to store analysis results (default: ./struct). '
                           'Note: output names are based on input basename/stem.')
  parser.add_argument('-v', '--visualize', action='store_true', default=False,
                      help='Save visualizations (default: False)')
  parser.add_argument('--viz-dir', type=str, default=cwd / 'viz',
                      help='Directory to save visualizations if -v is provided (default: ./viz)')
  parser.add_argument('-s', '--sonify', action='store_true', default=False,
                      help='Save sonifications (default: False)')
  parser.add_argument('--sonif-dir', type=str, default=cwd / 'sonif',
                      help='Directory to save sonifications if -s is provided (default: ./sonif)')
  parser.add_argument('-a', '--activ', action='store_true',
                      help='Save frame-level raw activations from sigmoid and softmax (default: False)')
  parser.add_argument('-e', '--embed', action='store_true',
                      help='Save frame-level embeddings (default: False)')
  parser.add_argument('-m', '--model', type=str, default='harmonix-all',
                      help='Name of the pretrained model to use (default: harmonix-all)')
  default_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
  parser.add_argument('-d', '--device', type=str, default=default_device, choices=['mps', 'cpu'],
                      help='Device to use (default: mps if available else cpu)')
  parser.add_argument('--demucs-device', type=str, choices=['mps', 'cpu'],
                      default=None,
                      help='Demucs device/engine (default: mps if available else cpu)')
  parser.add_argument('--timings-path', type=Path, default=None,
                      help='Write JSONL timings to this path (default: None)')
  parser.add_argument('--timings-viz-path', type=Path, default=None,
                      help='Write timing visualization to this path (requires --timings-path)')
  parser.add_argument('--timings-embed', action='store_true',
                      help='Embed timing info in activations output (default: False)')
  parser.add_argument('-k', '--keep-byproducts', action='store_true',
                      help='Keep demixed audio files and spectrograms (default: False)')
  parser.add_argument('--demix-dir', type=Path, default=cwd / 'demix',
                      help='Path to a directory to store demixed tracks (default: ./demix). '
                           'Uses input basename/stem for folder names.')
  parser.add_argument('--spec-dir', type=Path, default=cwd / 'spec',
                      help='Path to a directory to store spectrograms (default: ./spec). '
                           'Uses input basename/stem for filenames.')
  parser.add_argument('--spec-backend', type=str, choices=['torch'], default=None,
                      help='Spectrogram backend to use (default: torch)')
  parser.add_argument('--spec-torch-device', type=str, default='cpu',
                      help='Torch device for spectrogram backend (default: cpu)')
  parser.add_argument('--spec-torch-dtype', type=str, default='float32',
                      help='Torch dtype for spectrogram backend (default: float32)')
  parser.add_argument('--overwrite', nargs='?', const='all', default=None,
                      help='Overwrite stages: all or csv (demix,spec,json,viz,sonify)')
  parser.add_argument('--no-multiprocess', action='store_true', default=False,
                      help='Disable multiprocessing (default: False)')

  return parser


def main():
  parser = make_parser()
  args = parser.parse_args()

  if not args.paths:
    raise ValueError('At least one path must be specified.')

  assert args.out_dir is not None, 'Output directory must be specified with --out-dir'

  if args.demucs_device is None:
    args.demucs_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
  if args.spec_backend is None:
    args.spec_backend = 'torch'

  analyze(
    paths=args.paths,
    out_dir=args.out_dir,
    visualize=args.viz_dir if args.visualize else False,
    sonify=args.sonif_dir if args.sonify else False,
    model=args.model,
    device=args.device,
    demucs_device=args.demucs_device,
    include_activations=args.activ,
    include_embeddings=args.embed,
    demix_dir=args.demix_dir,
    spec_dir=args.spec_dir,
    spec_backend=args.spec_backend,
    spec_torch_device=args.spec_torch_device,
    spec_torch_dtype=args.spec_torch_dtype,
    keep_byproducts=args.keep_byproducts,
    overwrite=args.overwrite,
    multiprocess=not args.no_multiprocess,
    timings_path=args.timings_path,
    timings_embed=args.timings_embed,
    timings_viz_path=args.timings_viz_path,
  )

  print(f'=> Analysis results are successfully saved to {args.out_dir}')


if __name__ == '__main__':
  main()
