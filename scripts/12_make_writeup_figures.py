from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

DATA = Path("012_writeup/per_seed_250k.csv")
FIGURES = Path("012_writeup/figures")
Row = dict[str, Any]


def read_rows() -> list[dict[str, float | int | str]]:
    with DATA.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "variant": row["variant"],
                    "clear_rate": float(row["clear_rate"]),
                    "eval_cap_rate": float(row["eval_cap_rate"]),
                    "penalized_completion_cap": float(row["penalized_completion_cap"]),
                    "completion_time_given_clear": float(row["completion_time_given_clear"] or 0),
                    "stuck_fraction": float(row["stuck_fraction"]),
                    "mean_progress": float(row["mean_progress"]),
                }
            )
        return rows


def pair_rows(rows: list[dict[str, float | int | str]]) -> list[tuple[Row, Row]]:
    by_key = {(row["seed"], row["variant"]): row for row in rows}
    return [
        (by_key[(seed, "baseline")], by_key[(seed, "baseline_plus_time_tail")])
        for seed in sorted({int(row["seed"]) for row in rows})
    ]


def plot_reliability(pairs: list[tuple[Row, Row]]) -> None:
    metrics = [
        ("clear_rate", "Clear rate", "higher is better", 1),
        ("eval_cap_rate", "Cap rate", "lower is better", -1),
        ("stuck_fraction", "Stuck fraction", "lower is better", -1),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.5), sharey=True)
    seeds = [int(base["seed"]) for base, _ in pairs]

    for ax, (field, title, subtitle, direction) in zip(axes, metrics, strict=True):
        for base, tail in pairs:
            seed = int(base["seed"])
            base_y = seed - 0.1
            tail_y = seed + 0.1
            base_value = float(base[field])
            tail_value = float(tail[field])
            delta = (tail_value - base_value) * direction
            color = "#1f9d55" if delta >= 0 else "#c2410c"
            ax.plot(
                [base_value, tail_value],
                [base_y, tail_y],
                color=color,
                linewidth=2.2,
                alpha=0.75,
            )
            ax.scatter(base_value, base_y, s=48, color="#606060", zorder=3)
            ax.scatter(tail_value, tail_y, s=58, color="#2563eb", zorder=4)

        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.set_xlim(-0.04, 1.04)
        ax.set_xlabel("episode fraction")
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("seed")
    axes[0].set_yticks(seeds)
    axes[0].scatter([], [], s=48, color="#606060", label="baseline")
    axes[0].scatter([], [], s=58, color="#2563eb", label="time-tail")
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle("250k fine-tuning reliability by paired seed", fontsize=14, y=1.03)
    fig.text(
        0.5,
        -0.02,
        "Green connector means time-tail is better or tied; orange means worse.",
        ha="center",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "011_reliability_by_seed.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_penalized_completion(pairs: list[tuple[Row, Row]]) -> None:
    seeds = [int(base["seed"]) for base, _ in pairs]
    base_values = [float(base["penalized_completion_cap"]) for base, _ in pairs]
    tail_values = [float(tail["penalized_completion_cap"]) for _, tail in pairs]
    x = list(range(len(seeds)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    ax.bar([i - width / 2 for i in x], base_values, width, label="baseline", color="#777777")
    ax.bar([i + width / 2 for i in x], tail_values, width, label="time-tail", color="#2563eb")
    ax.axhline(
        sum(base_values) / len(base_values),
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        label="baseline mean",
    )
    ax.axhline(
        sum(tail_values) / len(tail_values),
        color="#2563eb",
        linestyle="--",
        linewidth=1.2,
        label="time-tail mean",
    )
    ax.set_title("Penalized completion by seed\nnon-clear episodes count as the 5000-step cap")
    ax.set_xlabel("seed")
    ax.set_ylabel("steps; lower is better")
    ax.set_xticks(x, seeds)
    ax.set_ylim(0, 5300)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "011_penalized_completion_by_seed.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pairs = pair_rows(read_rows())
    plot_reliability(pairs)
    plot_penalized_completion(pairs)


if __name__ == "__main__":
    main()
