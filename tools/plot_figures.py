"""Generate the documentation's academic-style figures with Matplotlib."""

import argparse
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

from bm3d_triton import denoise

ROI = (160, 48, 352, 240)


def save(fig, output, name, *, vector=False):
    fig.savefig(output / f"{name}.png", dpi=200, facecolor="white")
    fig.savefig(output / f"{name}.pdf", facecolor="white")
    if vector:
        buffer = StringIO()
        fig.savefig(buffer, format="svg", facecolor="white")
        svg = "\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n"
        (output / f"{name}.svg").write_text(svg, encoding="utf-8")
    plt.close(fig)


def comparison(arrays, output, *, detail=False):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.85))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.015, wspace=0.025)
    titles = (r"(a) Reference $x$", r"(b) Noisy input $z$", r"(c) BM3D estimate $\hat{x}$")
    for ax, array, title in zip(axes, arrays, titles):
        x0, y0, x1, y1 = ROI
        if detail:
            array = array[y0:y1, x0:x1]
        ax.imshow(array, interpolation="nearest" if detail else "antialiased", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, pad=9)
        ax.set_axis_off()
        if not detail:
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#a33232", linewidth=0.9
                )
            )
    save(fig, output, "detail" if detail else "denoising")


def pipeline(output):
    fig, ax = plt.subplots(figsize=(12.8, 5.0))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.97, bottom=0.03)
    ax.set(xlim=(0, 14.2), ylim=(0, 6))
    ax.set_axis_off()
    ink, blue, pale = "#30363d", "#355d79", "#edf2f5"

    def box(x, y, width, text, *, shaded=False):
        ax.add_patch(
            Rectangle(
                (x, y - 0.45),
                width,
                0.9,
                linewidth=0.85,
                edgecolor=ink,
                facecolor=pale if shaded else "white",
            )
        )
        ax.text(x + width / 2, y, text, ha="center", va="center", fontsize=10)

    def arrow(start, end, *, guide=False):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.9,
                color=blue if guide else ink,
                linestyle="--" if guide else "-",
                shrinkA=0,
                shrinkB=0,
            )
        )

    ax.text(0.2, 5.62, "(a) Basic-estimate stage", fontsize=12, fontweight="bold")
    ax.text(0.2, 2.67, "(b) Refinement stage", fontsize=12, fontweight="bold")
    positions = [(0.2, 1.25), (2.1, 1.8), (4.55, 1.8), (7.0, 2.0), (9.65, 2.0), (12.3, 1.65)]
    first = [
        "Noisy image\n$z$",
        "Block matching\nGroup patches",
        "2D DCT\n1D Hadamard",
        "Hard threshold\nInverse transforms",
        "Weighted\naggregation",
        "Basic estimate\n" + r"$\hat{x}_{\mathrm{basic}}$",
    ]
    second = [
        "Noisy image\n$z$",
        "Guided matching\nGroup patches",
        "2D DCT\n1D Hadamard",
        "Wiener filtering\nInverse transforms",
        "Weighted\naggregation",
        "Final estimate\n" + r"$\hat{x}$",
    ]
    for y, labels in [(4.7, first), (1.65, second)]:
        for i, ((x, width), label) in enumerate(zip(positions, labels)):
            box(x, y, width, label, shaded=i in (3, 4))
            if i:
                px, pw = positions[i - 1]
                arrow((px + pw, y), (x, y))
    ax.plot([13.125, 13.125, 3.0], [4.25, 3.43, 3.43], color=blue, linewidth=0.9, linestyle="--")
    arrow((3.0, 3.43), (3.0, 2.1), guide=True)
    arrow((8.0, 3.43), (8.0, 2.1), guide=True)
    ax.text(
        6.85,
        3.58,
        "Basic estimate: matching guide and Wiener-gain estimation",
        ha="center",
        color=blue,
        fontsize=9,
    )
    ax.text(
        7.0,
        0.58,
        "Solid arrows: signal path     Dashed arrows: basic-estimate guidance",
        ha="center",
        fontsize=9,
        color=ink,
    )
    ax.text(
        7.0,
        0.2,
        r"Patch size $8\times8$; orthonormal DCT; Kaiser-weighted overlap aggregation.",
        ha="center",
        fontsize=9,
        color=ink,
    )
    save(fig, output, "pipeline", vector=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", type=Path, help="local 512×512 astronaut.png; see assets/README.md"
    )
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    with Image.open(args.image) as source:
        if source.size != (512, 512):
            parser.error("the documented crop layout requires the 512×512 source")
        clean = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
    noisy = np.clip(
        clean + np.random.default_rng(123).normal(0, 25 / 255, clean.shape), 0, 1
    ).astype(np.float32)
    tensor = torch.from_numpy(noisy).permute(2, 0, 1)[None].cuda()
    result = denoise(tensor, sigma=25 / 255, two_step=True)
    assert result.shape == tensor.shape and bool(result.isfinite().all())
    restored = result[0].permute(1, 2, 0).cpu().numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    comparison((clean, noisy, restored), args.output)
    comparison((clean, noisy, restored), args.output, detail=True)
    pipeline(args.output)
    print(f"Saved Matplotlib PNG/PDF/SVG figures to {args.output}")


if __name__ == "__main__":
    main()
