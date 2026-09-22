# Contributing

Contributions should preserve a small, auditable denoising implementation.
Discuss changes to matching, transforms, noise modeling or public API behavior
in an issue before investing in a large implementation.

## Setup

Install compatible CUDA PyTorch/Triton for kernel work, then from the checkout:

```bash
python -m pip install -e ".[dev,cli,eval]"
ruff check .
ruff format --check .
pytest
```

CPU-only contributors can run `pytest -m "not cuda"`. Never describe skipped GPU
tests as GPU validation. CPU CI tests supported Python versions; GPU execution
depends on an explicitly configured self-hosted runner or local machine.

## Pull requests

- Describe the observable problem and resulting behavior.
- Keep changes focused; avoid unrelated formatting or dependency upgrades.
- Preserve input/output units, clipping behavior and device semantics, or document
  and test an intentional API change.
- For kernels, run the independent CPU reference comparisons and the image-quality
  suite. Report cases, sigma values, seed, GPU, software versions and tolerances.
- For performance claims, synchronize, warm up, alternate comparator order and
  attach raw timings. Report compilation separately; do not mix it into warm latency.
- Add meaningful regression coverage and update the relevant documentation.

Use BSD-2-Clause-compatible code and retain source/license attribution. Do not
submit images or model artifacts you cannot redistribute. The benchmark tools
accept local images specifically to keep data ownership separate from code.

By contributing, you agree that your contribution is available under the
repository's BSD-2-Clause license.

## Tests and numerical behavior

CPU tests cover input validation, lazy imports, CLI parsing, overwrite protection
and image/alpha preservation. GPU tests execute real kernels and compare small
images with an independent CPU matrix reference. They also cover odd and tiny
dimensions, batch/channel isolation, noncontiguous inputs, zero noise, chunked
scratch storage, dtype conversion and a CLI round trip.

The small-image reference tolerance is `atol=2e-5, rtol=2e-5`, not a universal
repeatability bound. Atomic rounding can perturb nearly tied second-stage
matches. Investigate discrepancies rather than widening tolerances without
analysis. A skipped GPU test is not GPU validation. Benchmark methods are in
[README.md](README.md#run-your-own-benchmarks); define acceptance criteria in advance.

## Release checks

Keep versions in `pyproject.toml` and `src/bm3d_triton/__init__.py` synchronized.
Run CPU/GPU tests, lint and representative quality checks. Build with
`python -m build` and validate with `python -m twine check dist/*`. Test the
installed wheel outside the checkout and inspect archives for retained licenses,
documented image provenance, and absence of private paths or benchmark outputs.

Before public hosting, configure real project URLs, maintainer contacts and
private vulnerability reporting. Publish only owner-approved artifacts and
release notes. The workflows build/test artifacts; they do not publish to PyPI.

## Community conduct

Be respectful, keep feedback technical and actionable, and protect others'
privacy. Harassment, discriminatory conduct and threats are not acceptable.
Maintainers may remove abusive content or restrict participation. Report
concerns privately through an available maintainer contact or the hosting
platform's moderation route. Security reports follow [SECURITY.md](SECURITY.md).
