"""Warm, synchronized timing for installed bm3d_triton; optional external legacy comparator."""

import argparse
import importlib
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch
import triton

import bm3d_triton


def measure(fn):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin = time.perf_counter()
    start.record()
    output = fn()
    end.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - begin) * 1000
    event = start.elapsed_time(end)
    if not bool(output.isfinite().all()):
        raise RuntimeError("Nonfinite benchmark output")
    return wall, event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", default=["256x256", "512x512", "1024x1024", "1080x1920"], help="HxW"
    )
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=25, help="8-bit units")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("benchmark-output/results.json"))
    parser.add_argument(
        "--reference-module", help="optional extension exposing forward(HW, variance, two_step)"
    )
    parser.add_argument(
        "--reference-path", type=Path, help="directory containing that external extension"
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1 or args.channels < 1:
        parser.error("repeats, warmup and channels must be positive")
    if args.reference_path:
        sys.path.insert(0, str(args.reference_path.resolve()))
    reference = importlib.import_module(args.reference_module) if args.reference_module else None
    variance = (args.sigma / 255) ** 2

    def legacy(image, two_step):
        image = image.float().contiguous().clamp_min(0)
        output = torch.empty_like(image)
        for source, target in zip(image.flatten(0, 1), output.flatten(0, 1)):
            target.copy_(reference.forward(source.contiguous(), variance, two_step))
        return output

    report = {
        "version": bm3d_triton.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "sigma_255": args.sigma,
        "channels": args.channels,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": 123,
        "protocol": "GPU-resident input/output; wall clock with synchronization; no JIT or transfers; clamp=False",
        "cases": [],
    }
    with torch.inference_mode():
        for shape in args.sizes:
            h, w = map(int, shape.split("x"))
            generator = torch.Generator(device="cuda").manual_seed(123)
            yy = torch.linspace(0, 1, h, device="cuda")[:, None]
            xx = torch.linspace(0, 1, w, device="cuda")[None, :]
            clean = torch.stack(
                [
                    0.45
                    + 0.15 * torch.sin(xx * (16 + c * 3)) * torch.cos(yy * (12 + c * 2))
                    + 0.15 * (xx > 0.5)
                    for c in range(args.channels)
                ]
            )[None]
            image = (
                clean
                + torch.randn(clean.shape, device="cuda", generator=generator) * (args.sigma / 255)
            ).clamp(0, 1)
            for step in (False, True):
                functions = {
                    "triton": partial(
                        bm3d_triton.denoise, image, args.sigma / 255, two_step=step, clamp=False
                    )
                }
                if reference is not None:
                    functions["reference"] = partial(legacy, image, step)
                for fn in functions.values():
                    for _ in range(args.warmup):
                        fn()
                    torch.cuda.synchronize()
                samples = {name: {"wall_ms": [], "event_ms": []} for name in functions}
                for repeat in range(args.repeats):
                    order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
                    for name in order:
                        wall, event = measure(functions[name])
                        samples[name]["wall_ms"].append(wall)
                        samples[name]["event_ms"].append(event)
                case = {
                    "height": h,
                    "width": w,
                    "two_step": step,
                    "samples": samples,
                    "median_wall_ms": {
                        name: statistics.median(values["wall_ms"])
                        for name, values in samples.items()
                    },
                }
                report["cases"].append(case)
                print(json.dumps(case["median_wall_ms"]), shape, "two_step=", step, flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
