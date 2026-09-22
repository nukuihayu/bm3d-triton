from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from bm3d_triton.cli import main


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "8-bit" in capsys.readouterr().out


def test_refuses_overwrite(tmp_path):
    output = tmp_path / "out.png"
    output.write_bytes(b"keep")
    with pytest.raises(SystemExit):
        main([str(tmp_path / "input.png"), str(output)])
    assert output.read_bytes() == b"keep"


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_io_preserves_pixels_and_alpha(tmp_path, mode):
    rng = np.random.default_rng(123)
    shape = (9, 11) if mode == "L" else (9, 11, len(mode))
    data = rng.integers(0, 256, shape, dtype=np.uint8)
    source, target = tmp_path / "in.png", tmp_path / "out.png"
    Image.fromarray(data).save(source)
    # Exercise file I/O independently of the GPU; kernel execution is tested separately.
    with patch("bm3d_triton.cli.denoise", side_effect=lambda x, *a, **kw: x) as call:
        assert main([str(source), str(target), "--device", "cpu", "--sigma", "25"]) == 0
        assert call.call_args.args[1] == 25 / 255
    np.testing.assert_array_equal(np.array(Image.open(target)), data)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cli_gpu(tmp_path):
    source, target = tmp_path / "in.png", tmp_path / "out.png"
    Image.fromarray(np.full((9, 11, 3), 128, dtype=np.uint8)).save(source)
    assert main([str(source), str(target)]) == 0
    np.testing.assert_allclose(np.array(Image.open(target)), 128, atol=1)
