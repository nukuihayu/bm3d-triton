import math
from unittest.mock import patch

import pytest
import torch

from bm3d_triton import BM3D, denoise, denoise_variance


@pytest.mark.parametrize("sigma", [-1, math.nan, math.inf])
def test_invalid_sigma(sigma):
    with pytest.raises(ValueError, match="sigma"):
        denoise(torch.ones(1, 1, 8, 8), sigma)


@pytest.mark.parametrize("sigma", [True, torch.tensor(0.1), object()])
def test_non_scalar_sigma(sigma):
    with pytest.raises(TypeError, match="sigma"):
        BM3D(sigma)


def test_cpu_validation_does_not_load_triton():
    with patch("bm3d_triton.api.importlib.import_module") as loader:
        with pytest.raises(RuntimeError, match="CUDA"):
            denoise(torch.ones(1, 1, 8, 8))
        loader.assert_not_called()


@pytest.mark.parametrize("shape", [(8, 8), (1, 1, 0, 8)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError, match="NCHW"):
        denoise(torch.empty(shape))


def test_input_type_and_options():
    with pytest.raises(TypeError, match="Tensor"):
        denoise([])
    with pytest.raises(TypeError, match="floating"):
        denoise(torch.ones(1, 1, 8, 8, dtype=torch.int32))
    with pytest.raises(TypeError, match="booleans"):
        BM3D(two_step=1)
    with pytest.raises(ValueError, match="variance"):
        denoise_variance(torch.ones(1, 1, 8, 8), -1)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_public_api_dtype_variance_and_autograd():
    image = torch.full((1, 1, 9, 11), 0.5, device="cuda", dtype=torch.float16, requires_grad=True)
    a = denoise(image, 0.1)
    b = denoise_variance(image, 0.01)
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    assert a.dtype == torch.float32 and a.device == image.device
    assert not a.requires_grad and a.is_contiguous()
    assert a.min() >= 0 and a.max() <= 1
    with patch("bm3d_triton.api.importlib.import_module", side_effect=ImportError("no Triton")):
        torch.testing.assert_close(denoise(image, 0), image.float())
        with pytest.raises(RuntimeError, match="compatible"):
            denoise(image, 0.1)
