"""House plotting style for the paper's figures (Nature/Science-quality).

A thin wrapper over `SciencePlots <https://github.com/garrettj403/SciencePlots>`_
plus a few repo-specific overrides, so the figure scripts share one look instead of
each re-deriving colours, fonts, and sizes. Currently wired into the three
single-accelerator experiment figures (paper Sec. 3.6):

    scripts/plot_exchange_band.py      (Fig 3, Exp 1)
    scripts/plot_operand_envelope.py   (Fig 4, Exp 2)
    scripts/plot_overhead_affine.py    (Fig 5, Exp 3)

Usage (after ``matplotlib.use("Agg")``)::

    from powerladder.plotstyle import apply_house_style, C, WIDTH_WIDE, save
    apply_house_style()
    ...
    save(fig, "exchange_band")

We drive matplotlib with ``plt.style.use(['science', 'nature', 'no-latex'])``: the
``nature`` style gives the sans-serif Nature look, and ``no-latex`` renders text
through matplotlib's own mathtext so regenerating figures needs no LaTeX install
(reproducibility, CLAUDE.md). Fonts are embedded as editable TrueType
(``pdf.fonttype = 42``) as journals require.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science'/'nature' styles)
from cycler import cycler

# Okabe-Ito colourblind-safe qualitative palette (https://jfly.uni-koeln.de/color/).
OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Named handles for semantic use in the scripts.
C = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#9a9a9a",
}

# ST2 attack-family colours, so the Sec.-6 frontier and Pareto figures read as one
# system. Single source of truth for new call sites; scripts/plot_st2_frontier.py
# still carries its own identical copy (same hex values) and should import this
# instead next time that file is touched -- re-running it rewrites the frozen
# frontier_summary.json, so it is deliberately left alone here.
FAMILY_COLOR = {
    "jitter": C["orange"], "work": C["blue"], "drift": C["green"],
    "phase": C["vermillion"], "relocate": C["purple"], "harmonic": C["yellow"],
    "shape": C["skyblue"], "dilute": C["black"], "meter": C["grey"],
}

# Native figure widths [inches] matched to the single-column (6.5 in) manuscript so
# the \includegraphics widths render at ~1:1 (no text rescaling):
#   WIDTH_WIDE   ~ 0.78 * 6.5 in  (Figs 3, 4)
#   WIDTH_NARROW ~ 0.72 * 6.5 in  (Fig 5)
WIDTH_WIDE = 5.1
WIDTH_NARROW = 4.7

# Native width for a single-column figure in the two-column ICML layout of
# paper-beta (6.75 in text width -> \columnwidth ~ 3.25 in), so
# \includegraphics[width=\columnwidth] renders at ~1:1.
WIDTH_ICML_COL = 3.25

# Row-of-three panel geometry for the Sec.-3 explainer (notes/section3-explainer).
# Each subfigure there renders at 0.32 of the full tufte figure* width (480pt),
# i.e. 153.6pt = 2.13in. We author the panels at that *native* width so they
# display ~1:1 (fonts at their true point size, not shrunk), and give all three
# the *same* fixed height so the row shares one aspect ratio.
PANEL_W = 2.13
PANEL_H = 1.95

_FIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "figures"


def apply_house_style() -> None:
    """Apply the shared SciencePlots + repo-override rcParams. Idempotent."""
    plt.style.use(["science", "nature", "no-latex"])
    plt.rcParams.update({
        # sans-serif Nature look; math glyphs in a matching sans face.
        # DejaVu Sans first ON PURPOSE: it ships with matplotlib, so it resolves
        # identically on every machine. Helvetica/Arial resolve only where the
        # system happens to have them, and the fallback silently changes text
        # metrics -> different tight-bbox -> different figure bytes. Every figure
        # tracked in this repo was rendered with DejaVu; pinning it first is what
        # makes "regenerate and diff" a usable reproducibility check.
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "stixsans",
        # editable embedded TrueType (journal requirement)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # output quality
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # legible at our ~5-inch single-column width (Nature's native column is ~3.3 in)
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        # ticks inward, minor ticks on; open L-shaped axis (left + bottom only),
        # the Nature/Science look -- no top/right spines or ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # colourblind-safe cycle
        "axes.prop_cycle": cycler(color=OKABE_ITO),
    })


def apply_panel_style() -> None:
    """House style tuned for the small row-of-three panels (``PANEL_W`` wide).

    Extends :func:`apply_house_style` with (a) smaller type that stays legible at
    the ~2.1in native panel width, and (b) ``savefig.bbox='standard'`` so every
    panel keeps its declared ``figsize`` — a tight bounding box would crop each
    figure to its own labels and reintroduce the mismatched aspect ratios we are
    fixing. Combine with ``layout='constrained'`` to fit labels inside the fixed
    canvas. Idempotent.
    """
    apply_house_style()
    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        # keep the full figsize so all three panels share one aspect ratio
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
    })


def save(fig, stem: str, subdir: str | None = None) -> pathlib.Path:
    """Write ``figures/{stem}.pdf`` (vector, primary) and ``.png`` (600-dpi preview).

    Returns the PDF path. The figures directory is resolved from the repo root, so
    this works regardless of the current working directory. ``subdir`` redirects the
    outputs to ``figures/{subdir}/`` — used for per-paper figure variants (e.g. the
    Viterbi-only figures of ``paper-viterbi/``) without touching the shared figures.
    """
    out_dir = _FIG_DIR / subdir if subdir else _FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    # Suppress the wall-clock /CreationDate matplotlib would otherwise stamp into
    # the PDF: with it, a tracked figure is byte-different on every regeneration
    # even when the plot is identical, so "every figure is regenerable" cannot be
    # checked by regenerate-and-diff and the figures churn in git for no reason.
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(out_dir / f"{stem}.png")
    return pdf


# Detector keys shared by the type-detection figure scripts (scripts/plot_b*.py).
DETECTOR_CHOICES = ("spectral", "viterbi")


def add_variant_args(ap) -> None:
    """Register the per-paper figure-variant flags on an ``argparse`` parser.

    ``--detectors`` selects which detector curves are *drawn*; scoring and the
    summary JSONs always cover both detectors so the shared results record is
    independent of the variant being plotted. ``--outdir`` names a subdirectory of
    ``figures/`` for the outputs (default: ``figures/`` itself, the shared
    two-detector figures used by ``paper/main.tex``).
    """
    ap.add_argument("--detectors", nargs="+", choices=DETECTOR_CHOICES,
                    default=list(DETECTOR_CHOICES), metavar="DET",
                    help="detector curves to draw (default: both); summaries always cover both")
    ap.add_argument("--outdir", default=None, metavar="SUBDIR",
                    help="subdirectory of figures/ for the outputs (e.g. paper-viterbi)")


def styled(style: dict, detectors) -> dict:
    """Filter a per-detector ``_STYLE`` dict to the detectors selected for drawing."""
    return {k: v for k, v in style.items() if k in detectors}
