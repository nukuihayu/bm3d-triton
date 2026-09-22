# Documentation assets

| File | Purpose |
|---|---|
| `denoising.png`, `.pdf` | Figure 1: reference / noisy / BM3D comparison; rectangles mark the detail region |
| `detail.png`, `.pdf` | Figure 2: identical crops, rendered with nearest-neighbor sampling |
| `pipeline.png`, `.svg`, `.pdf` | Figure 3: two-stage computation and basic-estimate guidance |

The PNG panels come from the actual `bm3d_triton.denoise` implementation.
They illustrate its behavior, without performance scores or benchmark rankings.
No sharpening, retouching or learned enhancement is applied.

All figures are drawn by `tools/plot_figures.py` using Matplotlib's noninteractive
Agg backend. Style: white background, serif text, numbered subpanels, thin lines
and restrained color. PNG exports use 200 dpi; PDFs preserve vector text and
diagram elements (photographs remain raster). The diagram also has a vector SVG.

## Image source and attribution

Source: NASA portrait of astronaut Eileen Collins, distributed as
`astronaut.png` in scikit-image. The scikit-image documentation describes this
NASA photograph as public domain, with no known copyright restrictions.

- [Original / NASA Great Images collection](https://flic.kr/p/r9qvLn)
- [scikit-image documentation](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)

NASA and the subject do not endorse this software. The source photograph's
public-domain status is separate from the repository's BSD-2-Clause code license.
Repository-authored figure layouts and generation code use that license.
The panels contain added synthetic noise and filtered derivatives.

## Reproduce

Use the original 512×512 RGB source and a compatible CUDA/Triton installation.
Supply a local path; the script does not download images or require scikit-image.

```bash
python -m pip install -e ".[figures]"
python tools/plot_figures.py /path/to/astronaut.png --output assets
```

Noise: NumPy `default_rng(123)`, Gaussian sigma `25/255`, clipped to `[0,1]`.
Processing: float32, two-stage BM3D, default output clipping. The detail crop is
`(left=160, top=48, right=352, bottom=240)` in source coordinates. Float32 atomic
aggregation can cause small differences when regenerating the output.
