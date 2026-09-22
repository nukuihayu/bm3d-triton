"""Public API, importable without Triton or an initialized CUDA runtime."""

import importlib
import math

import torch
from torch import nn


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, (bool, torch.Tensor)):
        raise TypeError(f"{name} must be a real scalar, not a boolean or tensor.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite nonnegative scalar.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def denoise(
    image: torch.Tensor,
    sigma: float = 25.0 / 255.0,
    *,
    two_step: bool = True,
    clamp: bool = True,
) -> torch.Tensor:
    """Denoise a normalized NCHW CUDA tensor using BM3D.

    Args:
        image: Nonempty floating-point tensor [N, C, H, W], with finite values.
            Each channel is processed independently; values are clipped to [0, 1].
        sigma: Noise standard deviation in normalized units, e.g. 25 / 255.
        two_step: Apply the Wiener refinement after hard thresholding.
        clamp: Clip output to [0, 1]; disable to inspect reconstruction overshoot.

    Returns:
        Contiguous float32 tensor on the input CUDA device, of the same shape.
        Inference only: no autograd graph is constructed.
    """
    sigma = _nonnegative(sigma, "sigma")
    return denoise_variance(image, sigma * sigma, two_step=two_step, clamp=clamp)


def denoise_variance(
    image: torch.Tensor,
    variance: float,
    *,
    two_step: bool = True,
    clamp: bool = True,
) -> torch.Tensor:
    """BM3D with normalized noise variance; equivalent to denoise(sigma=sqrt(variance))."""
    variance = _nonnegative(variance, "variance")
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or any(dim == 0 for dim in image.shape):
        raise ValueError("image must have nonempty NCHW dimensions [N, C, H, W].")
    if not image.is_floating_point():
        raise TypeError("image must be floating point, normalized to [0, 1].")
    if not isinstance(two_step, bool) or not isinstance(clamp, bool):
        raise TypeError("two_step and clamp must be booleans.")
    if image.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("BM3D Triton currently requires an NVIDIA CUDA device.")
    # Avoid device execution and JIT imports for the identity case.
    if variance == 0:
        with torch.inference_mode():
            return image.to(dtype=torch.float32).contiguous().clamp(0, 1)
    # Scalar parameters are cast to float32 inside the kernels.
    if variance < torch.finfo(torch.float32).tiny or variance > torch.finfo(torch.float32).max:
        raise ValueError("variance must be representable as a positive normal float32 value.")
    try:
        backend = importlib.import_module("bm3d_triton._backend")
    except ImportError as exc:
        raise RuntimeError(
            "BM3D needs a Triton build compatible with PyTorch: install triton on Linux "
            "or triton-windows on Windows. See README.md for installation."
        ) from exc
    with torch.inference_mode():
        result = backend.denoise(image, variance, two_step)
        return result.clamp(0, 1) if clamp else result


class BM3D(nn.Module):
    """Parameter-free, inference-only nn.Module storing BM3D noise settings.

    No parameters or buffers are created. Module settings are ordinary Python
    attributes, not part of state_dict(); the input determines the CUDA device.
    """

    def __init__(
        self, sigma: float = 25.0 / 255.0, *, two_step: bool = True, clamp: bool = True
    ) -> None:
        super().__init__()
        self.sigma = _nonnegative(sigma, "sigma")
        if not isinstance(two_step, bool) or not isinstance(clamp, bool):
            raise TypeError("two_step and clamp must be booleans.")
        self.two_step = two_step
        self.clamp = clamp

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return denoise(image, self.sigma, two_step=self.two_step, clamp=self.clamp)

    def extra_repr(self) -> str:
        return f"sigma={self.sigma}, two_step={self.two_step}, clamp={self.clamp}"
