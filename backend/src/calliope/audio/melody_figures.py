"""Figures for the OpenScore Lieder melody report.

The contour panel is read from docs/melody-holdout.json. The interval
histogram is recomputed on the same song split the evaluation uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calliope.audio.lieder_melody import iter_parsed_songs
from calliope.audio.melody_eval import (
    HOLDOUT_FRACTION,
    SPLIT_SEED,
    _true_delta,
    steps_for_song,
)
from calliope.audio.vocal_melody_ml import holdout_indices

REPO = Path(__file__).resolve().parents[4]
FIGURES = REPO / "docs" / "figures"
RESULTS = REPO / "docs" / "melody-holdout.json"
CORPUS = REPO / "data" / "openscore-lieder"

INK = "#1c2430"
MUTED = "#5c6b7a"
RULE = "#d5dbe3"
TEACHER = "#1f4e79"
ROLLOUT = "#c47b2b"
BAR = "#3d5a80"
BAR_ZERO = "#1f4e79"

CONTOUR_MODELS = (
    ("Constant median", "Constant\ninterval"),
    ("Markov interval", "Markov\ninterval"),
    ("MLP, all features", "MLP,\nall features"),
    ("MLP, previous degrees only", "MLP, pitch\nhistory only"),
)


def reference_steps(root: Path) -> np.ndarray:
    """Clipped reference intervals on the report holdout, including the opening step."""
    songs = iter_parsed_songs(root)
    hold = holdout_indices(len(songs), np.random.default_rng(SPLIT_SEED), fraction=HOLDOUT_FRACTION)
    deltas: list[int] = []
    for i, song in enumerate(songs):
        if i not in hold:
            continue
        steps = steps_for_song(song)
        if not steps:
            continue
        prev = steps[0].pool_len // 2
        for step in steps:
            deltas.append(_true_delta(step.true_idx, prev))
            prev = step.true_idx
    if not deltas:
        raise RuntimeError(f"no holdout steps under {root}")
    return np.asarray(deltas, dtype=int)


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.6,
        }
    )


def save_pipeline(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    _style(plt)
    fig, ax = plt.subplots(figsize=(10.4, 2.55))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 2.55)
    ax.axis("off")

    boxes = [
        (0.18, 0.72, 2.05, 1.22, "#f4f6f8", "Request\nlyric, voice, genre,\narrangement, tempo"),
        (2.62, 0.72, 2.25, 1.22, "#f4f6f8", "Symbolic melody\n12→64→32→2\nscale step, duration"),
        (5.26, 0.72, 2.25, 1.22, "#e7eef6", "Note list\ndegree and duration\nper syllable"),
        (7.90, 0.72, 2.28, 1.22, "#f4f6f8", "Synthesis\nspeech mapped\nonto the melody"),
    ]
    for x, y, w, h, face, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor=face,
                edgecolor=INK,
                linewidth=0.8,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, linespacing=1.25)

    for x0, x1 in ((2.23, 2.62), (4.87, 5.26), (7.51, 7.90)):
        ax.add_patch(
            FancyArrowPatch(
                (x0, 1.33),
                (x1, 1.33),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.8,
                color=INK,
            )
        )

    ax.annotate(
        "scored in this report",
        xy=(6.38, 0.72),
        xytext=(6.38, 0.22),
        ha="center",
        va="center",
        fontsize=8,
        color=TEACHER,
        arrowprops={"arrowstyle": "-", "color": TEACHER, "linewidth": 0.7},
    )
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_histogram(deltas: np.ndarray, path: Path) -> float:
    import matplotlib.pyplot as plt

    _style(plt)
    bins = np.arange(-5, 6)
    counts = np.array([(deltas == k).sum() for k in bins], dtype=float)
    frac = counts / counts.sum()
    unison = float(frac[bins == 0][0])

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = [BAR_ZERO if k == 0 else BAR for k in bins]
    ax.bar(bins, frac, width=0.82, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xticks(bins)
    ax.set_xlabel("Reference scale step")
    ax.set_ylabel("Fraction of test notes")
    ax.set_xlim(-5.7, 5.7)
    ax.set_ylim(0, max(frac) * 1.18)
    ax.yaxis.grid(True, color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate(
        f"{unison:.3f} unison",
        xy=(0, unison),
        xytext=(1.35, unison + 0.012),
        fontsize=8.5,
        color=TEACHER,
        arrowprops={"arrowstyle": "-", "color": TEACHER, "linewidth": 0.6},
    )
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return unison


def save_contour(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    _style(plt)
    by_name = {row["name"]: row for row in result["models"]}
    labels = [label for _name, label in CONTOUR_MODELS]
    teacher = []
    rollout = []
    teacher_err = []
    rollout_err = []
    for name, _label in CONTOUR_MODELS:
        row = by_name[name]
        for store, err, protocol in (
            (teacher, teacher_err, "teacher_forced"),
            (rollout, rollout_err, "rollout"),
        ):
            metric = row[protocol]
            y = float(metric["contour_acc"])
            lo, hi = metric["contour_acc_ci95"]
            store.append(y)
            err.append((y - float(lo), float(hi) - y))

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    t_err = np.array(teacher_err).T
    r_err = np.array(rollout_err).T
    ax.bar(
        x - width / 2,
        teacher,
        width,
        yerr=t_err,
        label="Teacher forcing",
        color=TEACHER,
        ecolor=INK,
        capsize=3,
        error_kw={"linewidth": 0.7},
    )
    ax.bar(
        x + width / 2,
        rollout,
        width,
        yerr=r_err,
        label="Autoregressive rollout",
        color=ROLLOUT,
        ecolor=INK,
        capsize=3,
        error_kw={"linewidth": 0.7},
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Contour accuracy")
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not CORPUS.is_dir():
        raise SystemExit(f"missing corpus at {CORPUS}")
    if not RESULTS.is_file():
        raise SystemExit(f"missing results at {RESULTS}")

    save_pipeline(FIGURES / "pipeline.png")
    deltas = reference_steps(CORPUS)
    unison = save_histogram(deltas, FIGURES / "interval-histogram.png")
    save_contour(json.loads(RESULTS.read_text()), FIGURES / "contour-accuracy.png")
    print(f"holdout notes={len(deltas)} unison={unison:.4f}")
    print(f"wrote {FIGURES}")


if __name__ == "__main__":
    main()
