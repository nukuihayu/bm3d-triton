"""Check source versions and require release tags to be exactly v<package version>."""

import argparse
import ast
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 development environments
    import tomli as tomllib


def check_release(root: Path, tag: str = "") -> str:
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = metadata["project"]["version"]
    tree = ast.parse((root / "src/bm3d_triton/__init__.py").read_text(encoding="utf-8"))
    versions = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        )
    ]
    if versions != [version]:
        raise ValueError(
            f"Source version {versions!r} does not match pyproject version {version!r}"
        )
    if tag and tag != f"v{version}":
        raise ValueError(f"Release tag must be v{version}, got {tag!r}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="", help="release tag, e.g. v0.1.0")
    args = parser.parse_args()
    print(check_release(Path(__file__).resolve().parents[1], args.tag))
