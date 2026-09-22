"""Quality evaluation on local images; no dataset download or CUDA reference required."""

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from bm3d_triton import __version__, denoise


def metrics(clean, output):
    mse = np.mean((clean.astype(np.float64) - output.astype(np.float64)) ** 2)
    return {
        "psnr": -10 * math.log10(max(float(mse), 1e-20)),
        "ssim": float(
            structural_similarity(
                clean,
                output,
                data_range=1,
                channel_axis=2 if clean.ndim == 3 else None,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        ),
    }


def array_from(tensor, gray):
    array = tensor[0].permute(1, 2, 0).cpu().numpy()
    return array[..., 0] if gray else array


def save_panel(destination, arrays):
    sheet = Image.new("RGB", (256 * len(arrays), 280), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (label, array) in enumerate(arrays.items()):
        h, w = array.shape[:2]
        y, x = max(0, (h - 256) // 2), max(0, (w - 256) // 2)
        crop = array[y : y + 256, x : x + 256]
        sheet.paste(
            Image.fromarray(np.round(crop * 255).astype(np.uint8)).convert("RGB"),
            (column * 256, 24),
        )
        draw.text((column * 256 + 8, 5), label, fill="black")
    sheet.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--sigma", nargs="+", type=float, default=[10, 25, 50], help="8-bit units")
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789])
    parser.add_argument("--output", type=Path, default=Path("quality-output"))
    parser.add_argument(
        "--reference-module",
        help="optional external extension exposing forward(HW, variance, two_step)",
    )
    parser.add_argument("--reference-path", type=Path)
    args = parser.parse_args()
    if args.reference_path:
        sys.path.insert(0, str(args.reference_path.resolve()))
    reference = importlib.import_module(args.reference_module) if args.reference_module else None
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "version": __version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "protocol": "center crop <=512, clipped AWGN, clipped outputs, PSNR/SSIM; sigma in 8-bit units",
        "cases": [],
    }
    with torch.inference_mode():
        for index, path in enumerate(args.images):
            with Image.open(path) as source:
                if source.mode not in ("L", "RGB"):
                    parser.error(f"{path}: expected 8-bit L or RGB image")
                clean = np.asarray(source, dtype=np.float32) / 255
            h, w = clean.shape[:2]
            if min(h, w) < 11:
                parser.error("SSIM evaluation requires images at least 11x11")
            y, x = max(0, (h - 512) // 2), max(0, (w - 512) // 2)
            clean = clean[y : y + 512, x : x + 512].copy()
            gray = clean.ndim == 2
            tensor = (
                torch.from_numpy(clean[..., None] if gray else clean).permute(2, 0, 1)[None].cuda()
            )
            for sigma in args.sigma:
                for seed in args.seeds:
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    noisy = (
                        tensor
                        + torch.randn(tensor.shape, device="cuda", generator=generator)
                        * (sigma / 255)
                    ).clamp(0, 1)
                    for step in (False, True):
                        outputs = {"Noisy": array_from(noisy, gray)}
                        if reference is not None:
                            result = torch.empty_like(noisy)
                            for src, dst in zip(noisy.flatten(0, 1), result.flatten(0, 1)):
                                dst.copy_(
                                    reference.forward(src.contiguous(), (sigma / 255) ** 2, step)
                                )
                            outputs["Reference"] = array_from(result.clamp(0, 1), gray)
                        outputs["Triton"] = array_from(
                            denoise(noisy, sigma / 255, two_step=step), gray
                        )
                        if not all(np.isfinite(array).all() for array in outputs.values()):
                            raise RuntimeError("Nonfinite evaluation output")
                        case = {
                            "image": path.name,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "sigma": sigma,
                            "seed": seed,
                            "two_step": step,
                            "metrics": {
                                name: metrics(clean, array) for name, array in outputs.items()
                            },
                        }
                        report["cases"].append(case)
                        if seed == args.seeds[0] and step:
                            save_panel(
                                args.output / f"{index}_{path.stem}_sigma{sigma:g}.png",
                                {"Clean": clean, **outputs},
                            )
            print(path.name, "complete", flush=True)
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
