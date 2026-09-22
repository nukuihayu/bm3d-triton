"""Lossless PNG output for 8-bit grayscale/RGB/RGBA images."""

import argparse
import math
from pathlib import Path

import torch

from . import __version__
from .api import denoise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="8-bit input image")
    parser.add_argument("output", type=Path, help="output .png file")
    parser.add_argument(
        "--sigma", type=float, default=25, help="noise sigma in 8-bit units (default: 25)"
    )
    parser.add_argument("--single-stage", action="store_true", help="skip Wiener refinement")
    parser.add_argument("--device", default="cuda:0", help="NVIDIA CUDA device (default: cuda:0)")
    parser.add_argument(
        "--overwrite", action="store_true", help="allow replacing an existing output"
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    if not math.isfinite(args.sigma) or args.sigma < 0:
        parser.error("--sigma must be finite and nonnegative")
    if args.output.suffix.lower() != ".png":
        parser.error("output must use .png to avoid lossy re-encoding")
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    if args.output.exists() and not args.overwrite:
        parser.error("output already exists; use --overwrite to replace it")
    try:
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError:
        parser.error('image I/O requires the cli extra: pip install ".[cli]"')
    try:
        with Image.open(args.input) as source:
            source = ImageOps.exif_transpose(source)
            if source.mode not in ("L", "RGB", "RGBA"):
                raise ValueError("supported input modes are 8-bit L, RGB and RGBA")
            array = np.array(source)
        alpha = array[..., 3].copy() if array.ndim == 3 and array.shape[2] == 4 else None
        pixels = array[..., :3] if alpha is not None else array
        if pixels.ndim == 2:
            pixels = pixels[..., None]
        tensor = (
            torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].to(args.device, torch.float32)
            / 255
        )
        result = denoise(tensor, args.sigma / 255, two_step=not args.single_stage)
        output = (result[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        if alpha is not None:
            output = np.dstack((output, alpha))
        elif output.shape[2] == 1:
            output = output[..., 0]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids overwriting a file created during GPU processing.
        with args.output.open("wb" if args.overwrite else "xb") as handle:
            Image.fromarray(output).save(handle, format="PNG")
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Saved {args.output}")
    return 0
