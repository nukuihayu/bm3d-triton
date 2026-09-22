"""GPU BM3D numerical checks; also runnable directly without pytest/ComfyUI."""

import importlib.util
import math
import unittest

import pytest
import torch
import torch.nn.functional as F


def _load_backend():
    from bm3d_triton import _backend

    return _backend


def _reference_stage(noisy, basic, variance, wiener):
    """Independent CPU reference using dense transform matrices and unfold."""
    h, w = noisy.shape
    patches = F.unfold(noisy[None, None], 8)[0].T.reshape(-1, 8, 8)
    guides = F.unfold(basic[None, None], 8)[0].T.reshape(-1, 8, 8)
    freq = torch.arange(8)[:, None]
    dct = torch.cos(math.pi / 8 * freq * (torch.arange(8)[None, :] + 0.5)) * 0.5
    dct[0] *= 1 / math.sqrt(2)
    win = torch.kaiser_window(8, periodic=False, beta=2)
    window = win[:, None] * win[None, :]
    numerator, denominator = torch.zeros_like(noisy), torch.zeros_like(noisy)
    rows = list(range(0, h - 7, 3))
    cols = list(range(0, w - 7, 3))
    if rows[-1] != h - 8:
        rows.append(h - 8)
    if cols[-1] != w - 8:
        cols.append(w - 8)
    for y in rows:
        for x in cols:
            ref = y * (w - 7) + x
            candidates = [
                yy * (w - 7) + xx
                for yy in range(max(0, y - 19), min(h - 8, y + 19) + 1)
                for xx in range(max(0, x - 19), min(w - 8, x + 19) + 1)
                if (yy, xx) != (y, x)
            ]
            distance = ((guides[candidates] - guides[ref]) ** 2).sum((1, 2))
            order = torch.argsort(distance, stable=True)
            selected = [
                candidates[i]
                for i in order
                if distance[i] < (400 if wiener else 2500) * 64 / 255**2
            ]
            indices = [ref] + selected[: 31 if wiener else 15]
            count = 2 ** int(math.log2(len(indices)))
            indices = indices[:count]
            hadamard = torch.ones(1, 1)
            while hadamard.shape[0] < count:
                hadamard = torch.cat(
                    (torch.cat((hadamard, hadamard), 1), torch.cat((hadamard, -hadamard), 1)), 0
                )
            coeff = hadamard @ (dct @ patches[indices] @ dct.T).reshape(count, 64)
            if wiener:
                guide = hadamard @ (dct @ guides[indices] @ dct.T).reshape(count, 64)
                power = guide.square() / count
                gain = power / (power + variance)
                energy = gain.square().sum()
                weight = 1 / energy if energy > 0 else 1.0
                coeff *= gain
            else:
                keep = coeff.abs() >= 2.7 * math.sqrt(count * variance)
                weight = 1 / max(keep.sum().item(), 1)
                coeff *= keep
            restored = dct.T @ (hadamard @ coeff / count).reshape(count, 8, 8) @ dct
            for idx, patch in zip(indices, restored):
                yy, xx = divmod(idx, w - 7)
                numerator[yy : yy + 8, xx : xx + 8] += patch * window * weight
                denominator[yy : yy + 8, xx : xx + 8] += window * weight
    return numerator / denominator


@pytest.mark.cuda
@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("triton"), "CUDA and Triton required"
)
class TestBM3DTriton(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = _load_backend()

    def test_reference(self):
        generator = torch.Generator().manual_seed(43)
        for shape in [(1, 1), (8, 8), (9, 11), (12, 13)]:
            original = (0.5 + 0.07 * torch.randn(shape, generator=generator)).clamp(0, 1)
            padded = F.pad(
                original[None, None],
                (0, max(0, 8 - shape[1]), 0, max(0, 8 - shape[0])),
                mode="replicate",
            )[0, 0]
            basic = _reference_stage(padded, padded, 0.01, False)
            for two_step in [False, True]:
                with self.subTest(shape=shape, two_step=two_step):
                    expected = _reference_stage(padded, basic, 0.01, True) if two_step else basic
                    actual = self.backend.denoise(original[None, None].cuda(), 0.01, two_step)
                    torch.testing.assert_close(
                        actual[0, 0].cpu(), expected[: shape[0], : shape[1]], atol=2e-5, rtol=2e-5
                    )

    def test_batch_channels_noncontiguous_and_zero(self):
        image = torch.rand(2, 3, 11, 9, device="cuda").transpose(-1, -2)
        torch.testing.assert_close(self.backend.denoise(image, 0), image)
        actual = self.backend.denoise(image, 0.01)
        for n in range(2):
            for c in range(3):
                expected = self.backend.denoise(image[n : n + 1, c : c + 1], 0.01)
                torch.testing.assert_close(actual[n, c], expected[0, 0], atol=2e-5, rtol=2e-5)

    def test_denoising_and_constant(self):
        generator = torch.Generator(device="cuda").manual_seed(123)
        # More than 4096 reference patches exercises scratch-buffer reuse.
        clean = torch.full((1, 1, 200, 201), 0.5, device="cuda")
        noisy = clean + torch.randn(clean.shape, device="cuda", generator=generator) * 0.08
        for two_step in [False, True]:
            output = self.backend.denoise(noisy, 0.08**2, two_step)
            self.assertLess((output - clean).square().mean(), (noisy - clean).square().mean() * 0.3)
            constant = self.backend.denoise(clean, 0.08**2, two_step)
            torch.testing.assert_close(constant, clean, atol=2e-4, rtol=2e-4)

    def test_validation(self):
        image = torch.ones(1, 1, 8, 8, device="cuda")
        for variance in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                self.backend.denoise(image, variance)
        with self.assertRaises(RuntimeError):
            self.backend.denoise(image.cpu(), 0.01)
        with self.assertRaises(ValueError):
            self.backend.denoise(image[0], 0.01)


if __name__ == "__main__":
    unittest.main()
