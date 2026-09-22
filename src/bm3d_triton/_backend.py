"""BM3D for normalized NCHW images, without a compiled CUDA extension.

8x8 orthonormal DCT, unnormalized Walsh-Hadamard group transform, and
Kaiser aggregation. Matching uses float distances (the old CUDA backend
truncated distances to integers), so results are not bitwise identical.
See NOTICE and LICENSE for attribution.
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _match(
    Image,
    Locations,
    Counts,
    H: tl.constexpr,
    W: tl.constexpr,
    NX: tl.constexpr,
    START,
    K: tl.constexpr,
    LIMIT: tl.constexpr,
):
    group = tl.program_id(0)
    ref = START + group
    ry = tl.minimum(ref // NX * 3, H - 8)
    rx = tl.minimum(ref % NX * 3, W - 8)
    c = tl.arange(0, 2048)
    y = ry + c // 39 - 19
    x = rx + c % 39 - 19
    valid = (c < 1521) & (y >= 0) & (y <= H - 8) & (x >= 0) & (x <= W - 8)
    distance = tl.full((2048,), 0, tl.float32)
    for iy in range(8):
        for ix in range(8):
            a = tl.load(Image + (ry + iy) * W + rx + ix)
            b = tl.load(Image + (y + iy) * W + x + ix, valid, 0)
            distance += (a - b) * (a - b)
    distance = tl.where(valid & (distance < LIMIT) & (c != 760), distance, float("inf"))
    tl.store(Locations + group * K, ry * W + rx)
    count = tl.full((), 1, tl.int32)
    for slot in range(1, K):
        best = tl.min(distance, 0)
        index = tl.min(tl.where(distance == best, c, 2147483647), 0)
        found = best < float("inf")
        loc = (ry + index // 39 - 19) * W + rx + index % 39 - 19
        tl.store(Locations + group * K + slot, tl.where(found, loc, ry * W + rx))
        count += found.to(tl.int32)
        distance = tl.where(c == index, float("inf"), distance)
    power = tl.full((), 1, tl.int32)
    for shift in tl.static_range(1, 6):
        power = tl.where(count >= (1 << shift), 1 << shift, power)
    tl.store(Counts + group, power)


@triton.jit
def _dct(values, K: tl.constexpr, INVERSE: tl.constexpr):
    p = tl.arange(0, 64)[None, :]
    # Separate x and y transforms, retaining float32 throughout.
    out = tl.full((K, 64), 0, tl.float32)
    for i in tl.static_range(8):
        indices = tl.broadcast_to(p // 8 * 8 + i, (K, 64))
        v = tl.gather(values, indices, 1)
        if INVERSE:
            coefficient = tl.cos((p % 8 + 0.5) * i * (math.pi / 8))
            coefficient *= 0.3535533905932738 if i == 0 else 0.5
        else:
            coefficient = tl.cos((i + 0.5) * (p % 8) * (math.pi / 8))
            coefficient *= tl.where(p % 8 == 0, 0.3535533905932738, 0.5)
        out += v * coefficient
    result = tl.full((K, 64), 0, tl.float32)
    for i in tl.static_range(8):
        indices = tl.broadcast_to(i * 8 + p % 8, (K, 64))
        v = tl.gather(out, indices, 1)
        if INVERSE:
            coefficient = tl.cos((p // 8 + 0.5) * i * (math.pi / 8))
            coefficient *= 0.3535533905932738 if i == 0 else 0.5
        else:
            coefficient = tl.cos((i + 0.5) * (p // 8) * (math.pi / 8))
            coefficient *= tl.where(p // 8 == 0, 0.3535533905932738, 0.5)
        result += v * coefficient
    return result


@triton.jit
def _hadamard(values, count, K: tl.constexpr):
    z = tl.arange(0, K)[:, None]
    for shift in tl.static_range(0, triton.next_power_of_2(K).bit_length() - 1):
        step = 1 << shift
        other = tl.gather(values, tl.broadcast_to(z ^ step, (K, 64)), 0)
        transformed = tl.where((z & step) == 0, values + other, other - values)
        values = tl.where(step < count, transformed, values)
    return values


@triton.jit
def _filter(
    Noisy,
    Basic,
    Locations,
    Counts,
    Window,
    Numerator,
    Denominator,
    W: tl.constexpr,
    K: tl.constexpr,
    VAR,
    WIENER: tl.constexpr,
):
    group = tl.program_id(0)
    z = tl.arange(0, K)[:, None]
    p = tl.arange(0, 64)[None, :]
    count = tl.load(Counts + group)
    locations = tl.load(Locations + group * K + z)
    offsets = locations + p // 8 * W + p % 8
    values = tl.load(Noisy + offsets, z < count, 0)
    values = _hadamard(_dct(values, K, False), count, K)
    if WIENER:
        basic = tl.load(Basic + offsets, z < count, 0)
        basic = _hadamard(_dct(basic, K, False), count, K)
        power = basic * basic / count
        gain = power / (power + VAR)
        gain = tl.where(z < count, gain, 0)
        energy = tl.sum(tl.sum(gain * gain, 1), 0)
        weight = tl.where(energy > 0, 1.0 / energy, 1.0)
        values *= gain
    else:
        keep = (tl.abs(values) >= 2.7 * tl.sqrt(count * VAR)) & (z < count)
        weight = 1.0 / tl.maximum(tl.sum(tl.sum(keep.to(tl.int32), 1), 0), 1)
        values = tl.where(keep, values, 0)
    values = _dct(_hadamard(values, count, K) / count, K, True)
    weights = weight * tl.load(Window + p)
    tl.atomic_add(Numerator + offsets, values * weights, z < count, sem="relaxed")
    tl.atomic_add(
        Denominator + offsets, tl.broadcast_to(weights, (K, 64)), z < count, sem="relaxed"
    )


@triton.jit
def _normalize(Numerator, Denominator, Original, Output, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    num = tl.load(Numerator + i, i < SIZE, 0)
    den = tl.load(Denominator + i, i < SIZE, 0)
    original = tl.load(Original + i, i < SIZE, 0)
    tl.store(Output + i, tl.where(den > 0, num / den, original), i < SIZE)


def _stage(noisy, basic, variance, wiener, window):
    h, w = noisy.shape
    nx = triton.cdiv(w - 8, 3) + 1
    ny = triton.cdiv(h - 8, 3) + 1
    k = 32 if wiener else 16
    numerator = torch.zeros_like(noisy)
    denominator = torch.zeros_like(noisy)
    # Bound scratch storage independently of image resolution.
    chunk = min(nx * ny, 4096)
    locations = torch.empty((chunk, k), device=noisy.device, dtype=torch.int32)
    counts = torch.empty(chunk, device=noisy.device, dtype=torch.int32)
    for start in range(0, nx * ny, chunk):
        size = min(chunk, nx * ny - start)
        _match[(size,)](
            basic, locations, counts, h, w, nx, start, k, (400 if wiener else 2500) * 64 / 255**2
        )
        _filter[(size,)](
            noisy, basic, locations, counts, window, numerator, denominator, w, k, variance, wiener
        )
    output = torch.empty_like(noisy)
    _normalize[(triton.cdiv(h * w, 256),)](numerator, denominator, noisy, output, h * w, 256)
    return output


def denoise(image_nchw: torch.Tensor, variance: float, two_step: bool = True) -> torch.Tensor:
    """Denoise each batch/channel independently; variance is in [0,1] units.

    Inputs smaller than 8 pixels are replicate-padded and cropped afterwards.
    Overlap accumulation uses float32 atomics, with normal rounding variation.
    """
    if image_nchw.ndim != 4:
        raise ValueError("BM3D expects an NCHW tensor.")
    if image_nchw.device.type != "cuda":
        raise RuntimeError("BM3D Triton requires a CUDA device.")
    variance = float(variance)
    if not math.isfinite(variance) or variance < 0:
        raise ValueError("BM3D variance must be finite and nonnegative.")
    if any(dim == 0 for dim in image_nchw.shape):
        raise ValueError("BM3D does not accept empty dimensions.")
    with torch.cuda.device(image_nchw.device), torch.inference_mode():
        image = image_nchw.to(torch.float32).contiguous().clamp(0, 1)
        if variance == 0:
            return image.clone()
        h, w = image.shape[-2:]
        image = F.pad(image, (0, max(0, 8 - w), 0, max(0, 8 - h)), mode="replicate")
        window = torch.kaiser_window(8, periodic=False, beta=2.0, device=image.device)
        window = (window[:, None] * window[None, :]).contiguous()
        output = torch.empty_like(image)
        for source, target in zip(image.flatten(0, 1), output.flatten(0, 1)):
            basic = _stage(source, source, variance, False, window)
            target.copy_(_stage(source, basic, variance, True, window) if two_step else basic)
        return output[..., :h, :w].contiguous()
