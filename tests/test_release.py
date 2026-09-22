"""Release guards must reject mismatched tags before publishing artifacts."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_release", Path(__file__).resolve().parents[1] / "tools/check_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.fixture
def source(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    package = tmp_path / "src/bm3d_triton"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "1.2.3"\n')
    return tmp_path


@pytest.mark.parametrize("tag", ["", "v1.2.3"])
def test_matching_versions(source, tag):
    assert release.check_release(source, tag) == "1.2.3"


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2.4", "v1.2.3-rc1", "unrelated"])
def test_wrong_tag(source, tag):
    with pytest.raises(ValueError, match="Release tag"):
        release.check_release(source, tag)


def test_source_mismatch(source):
    (source / "src/bm3d_triton/__init__.py").write_text('__version__ = "1.2.4"\n')
    with pytest.raises(ValueError, match="Source version"):
        release.check_release(source, "v1.2.3")
