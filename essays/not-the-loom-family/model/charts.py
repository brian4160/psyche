#!/usr/bin/env python3
"""
charts.py — render the two-economy model's outputs as small-multiple line charts.

Reads ./output/trajectories.csv and ./output/uncertainty.csv (written by two_economy.py)
and writes PNGs to ./output:
  macro.png       the two economies by terminal state (six measures)
  households.png  household archetypes by terminal state (six measures)
  mechanism.png   capability, deployment, human jobs and the wage in the modal scenario
  fan.png         all-in expectation with 10th to 90th percentile bands from the Monte Carlo
  drawdown.png    a retiree's savings under 4% real withdrawals, by scenario and strategy

Palette: the validated reference categorical palette (slots 1–5, fixed order) for the five
modeled terminal states; the loom comparator is a dashed neutral gray so it reads as a
reference line rather than a sixth series. Lines are 2px, grid is a recessive hairline,
every panel carries a legend and sparing direct labels at the line ends.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"

SERIES = {  # fixed order; validated palette slots 1-5
    "managed_dependency": ("Managed dependency", "MD", "#2a78d6"),
    "durable_fragmentation": ("Durable fragmentation", "DF", "#eb6834"),
    "compensated_restructuring": ("Compensated restructuring", "CR", "#1baf7a"),
    "capability_diffusion": ("Capability diffusion", "CD", "#eda100"),
    "prolonged_disorder": ("Prolonged disorder", "PD", "#e87ba4"),
}
COMPARATOR = ("loom", "Comparator: the loom", "loom", "#8a8985")
FOOTER = ("Source: two_economy.py, an illustrative model (not a forecast). Indices: 2026 = 100, real terms, "
          "trend productivity growth included. Parameters and caveats in MODEL.md.")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style(ax, title: str, pct: bool = False, xmax: float = 2042.5) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.grid(axis="x", visible=False)
    ax.tick_params(colors=TEXT2, length=0)
    ax.set_title(title, loc="left", color=TEXT, fontweight="bold")
    if pct:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(2026, xmax)
    ax.set_xticks([2026, 2030, 2034, 2038])
    ax.margins(y=0.08)


def end_labels(ax, items, min_gap_frac: float = 0.075) -> None:
    """Direct labels at the right edge, nudged apart so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda x: x[0])
    ys = [y for y, _, _ in items]
    for k in range(1, len(ys)):
        if ys[k] - ys[k - 1] < gap:
            ys[k] = ys[k - 1] + gap
    over = ys[-1] - hi
    if over > 0:
        ys = [y - over for y in ys]
    for (y0, label, color), y in zip(items, ys):
        ax.plot([2040, 2040.35], [y0, y], color=color, lw=0.8, alpha=0.9)
        ax.text(2040.5, y, label, fontsize=7.5, color=TEXT2, va="center", ha="left")


def panel(ax, df: pd.DataFrame, col: str, title: str, scale: float = 100.0, pct: bool = False) -> None:
    ends = []
    for key, (name, short, color) in SERIES.items():
        d = df[df.scenario == key]
        y = d[col] * (1.0 if pct else scale)
        ax.plot(d.year, y, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", label=name)
        ax.plot(d.year.iloc[-1], y.iloc[-1], "o", ms=6, color=color, mec=SURFACE, mew=1.5)
        ends.append((float(y.iloc[-1]), short, color))
    d = df[df.scenario == COMPARATOR[0]]
    if len(d):
        y = d[col] * (1.0 if pct else scale)
        ax.plot(d.year, y, color=COMPARATOR[3], lw=1.5, ls=(0, (4, 3)), label=COMPARATOR[1])
        ends.append((float(y.iloc[-1]), COMPARATOR[2], COMPARATOR[3]))
    style(ax, title, pct=pct)
    end_labels(ax, ends)


def figure(df: pd.DataFrame, panels, suptitle: str, subtitle: str, fname: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8))
    for ax, (col, title, scale, pct) in zip(axes.flat, panels):
        panel(ax, df, col, title, scale=scale, pct=pct)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=6, frameon=False,
               fontsize=8, handlelength=2.2, columnspacing=1.6)
    fig.suptitle(suptitle, x=0.035, ha="left", fontsize=13, fontweight="bold", color=TEXT, y=0.985)
    fig.text(0.035, 0.945, subtitle, fontsize=8.5, color=TEXT2, ha="left")
    fig.text(0.035, 0.012, FOOTER, fontsize=7.5, color=TEXT2, ha="left")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.9), w_pad=2.5)
    fig.savefig(os.path.join(OUT, fname), dpi=160)
    plt.close(fig)


def mechanism_figure(df: pd.DataFrame, fname: str) -> None:
    d = df[df.scenario == "managed_dependency"]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    series = [
        ("cap_cog", "AI can do (share of cognitive tasks)", "#2a78d6"),
        ("deployed_cog", "Actually deployed in firms", "#eb6834"),
        ("jobs_cog_demand", "Human cognitive jobs demanded (2026 = 100%)", "#1baf7a"),
        ("real_wage_cog", "Real cognitive wage (2026 = 100%)", "#8a8985"),
    ]
    ends = []
    for col, label, color in series:
        ls = (0, (4, 3)) if color == "#8a8985" else "-"
        ax.plot(d.year, d[col], color=color, lw=2 if ls == "-" else 1.5, ls=ls, label=label)
        ax.plot(d.year.iloc[-1], d[col].iloc[-1], "o", ms=6, color=color, mec=SURFACE, mew=1.5)
        ends.append((float(d[col].iloc[-1]), label.split(" (")[0], color))
    style(ax, "The mechanism, in the modal scenario (managed dependency)", pct=True, xmax=2046)
    end_labels(ax, ends, min_gap_frac=0.07)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8)
    fig.text(0.02, 0.012, "Capability runs ahead of deployment; deployment removes human tasks; the wage follows the excess of\n"
             "workers over the tasks that remain human. Source: two_economy.py (an illustrative model, not a forecast).",
             fontsize=7.5, color=TEXT2, va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, fname), dpi=160)
    plt.close(fig)


def fan_figure(u: pd.DataFrame, fname: str, p_wrong: float) -> None:
    """All-in expectation (premise-wrong probability included) with Monte Carlo bands."""
    panels = [
        ("real_wage_cog", "Real pay, office and professional work", 100, False),
        ("real_household_income", "Real household income, all households", 100, False),
        ("entrant_index", "New graduate, cognitive work: openings × pay", 100, False),
        ("portfolio_60_40", "60/40 portfolio, total return, real", 100, False),
    ]
    blue = "#2a78d6"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    a = u[u.kind == "conditional"]
    loom = u[u.kind == "loom"]
    for ax, (col, title, scale, pct) in zip(axes.flat, panels):
        d = a[a.measure == col].sort_values("year")
        k = 1.0 if pct else scale
        ax.fill_between(d.year, d.p10 * k, d.p90 * k, color=blue, alpha=0.12, linewidth=0)
        ax.plot(d.year, d.p50 * k, color=blue, lw=2, solid_capstyle="round")
        ax.plot(d.year.iloc[-1], d.p50.iloc[-1] * k, "o", ms=6, color=blue, mec=SURFACE, mew=1.5)
        lm = loom[loom.measure == col].sort_values("year")
        ax.plot(lm.year, lm.p50 * k, color=COMPARATOR[3], lw=1.5, ls=(0, (4, 3)))
        style(ax, title, pct=pct)
        end_labels(ax, [(float(d.p50.iloc[-1] * k), "median", blue),
                        (float(d.p90.iloc[-1] * k), "90th", blue),
                        (float(d.p10.iloc[-1] * k), "10th", blue),
                        (float(lm.p50.iloc[-1] * k), "old pattern", COMPARATOR[3])], min_gap_frac=0.07)
    handles = [Line2D([], [], color=blue, lw=2), Patch(facecolor=blue, alpha=0.12),
               Line2D([], [], color=COMPARATOR[3], lw=1.5, ls=(0, (4, 3)))]
    labels = ["If the premise holds: median", "If the premise holds: 10th to 90th percentile across parameter draws",
              "If the premise is wrong: comparator median"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("How sure is any of this? Two worlds and the spread inside each", x=0.035, ha="left",
                 fontsize=13, fontweight="bold", color=TEXT, y=0.985)
    fig.text(0.035, 0.945, f"Roughly {100 * p_wrong:.0f}% on the dashed line (the old pattern holds) and {100 - 100 * p_wrong:.0f}% on the band "
             "(the five modeled endings, weighted by the essay's odds).\nBands: 400 draws of the shared parameters. Indices: 2026 = 100.",
             fontsize=8.5, color=TEXT2, ha="left", va="top")
    fig.text(0.035, 0.012, FOOTER, fontsize=7.5, color=TEXT2, ha="left")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.88), w_pad=2.5)
    fig.savefig(os.path.join(OUT, fname), dpi=160)
    plt.close(fig)


def drawdown_figure(df: pd.DataFrame, fname: str) -> None:
    """A retiree's savings under 4% real withdrawals: one panel per scenario, three strategies."""
    strategies = [
        ("draw_all_equity", "All equity", "#2a78d6"),
        ("draw_60_40", "60/40, rebalanced", "#eb6834"),
        ("draw_bucket", "Equity + 3-year cash bucket", "#1baf7a"),
    ]
    order = list(SERIES.keys()) + [COMPARATOR[0]]
    titles = {k: v[0] for k, v in SERIES.items()}
    titles[COMPARATOR[0]] = "If the premise is wrong (comparator)"
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), sharey=True)
    for ax, key in zip(axes.flat, order):
        d = df[df.scenario == key]
        ends = []
        for col, label, color in strategies:
            ax.plot(d.year, d[col], color=color, lw=2, solid_capstyle="round", label=label)
            ax.plot(d.year.iloc[-1], d[col].iloc[-1], "o", ms=6, color=color, mec=SURFACE, mew=1.5)
            ends.append((float(d[col].iloc[-1]), label.split(",")[0], color))
        style(ax, titles[key])
        ax.axhline(100, color=GRID, lw=1)
        end_labels(ax, ends, min_gap_frac=0.07)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("A retiree's savings under 4% real withdrawals, by ending and strategy", x=0.035, ha="left",
                 fontsize=13, fontweight="bold", color=TEXT, y=0.985)
    fig.text(0.035, 0.945, "Start at 100 in 2026, withdraw 4 a year in real terms. The bucket strategy spends cash while "
             "equities are below their previous peak and refills it at new peaks.", fontsize=8.5, color=TEXT2, ha="left")
    fig.text(0.035, 0.012, FOOTER, fontsize=7.5, color=TEXT2, ha="left")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.9), w_pad=2.0)
    fig.savefig(os.path.join(OUT, fname), dpi=160)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(os.path.join(OUT, "trajectories.csv"))
    figure(
        df,
        [
            ("real_wage_cog", "Real wage, cognitive work", 100, False),
            ("real_wage_phys", "Real wage, presence work", 100, False),
            ("not_employed", "Not employed, share of workforce", 1, True),
            ("dependency_ratio", "Dependency ratio (transfers' share of income)", 1, True),
            ("real_household_income", "Real household income, all households", 100, False),
            ("broad_equity", "Broad equity, real", 100, False),
        ],
        "The two economies, 2026–2040, by terminal state",
        "Five modeled terminal states from 'Not the Loom' plus a comparator in which AI behaves like earlier technologies. Indices: 2026 = 100.",
        "macro.png",
    )
    figure(
        df,
        [
            ("retiree_income", "Retired couple: Social Security + portfolio draw", 100, False),
            ("portfolio_60_40", "60/40 portfolio, total return, real", 100, False),
            ("incumbent_wage", "Mid-career professional, keeps the job: real pay", 100, False),
            ("entrant_index", "New graduate, cognitive work: openings × pay", 100, False),
            ("licensed_index", "Licensed presence profession: real pay", 100, False),
            ("infra_index", "Infrastructure engineer: real pay", 100, False),
        ],
        "What each life stage faces, by terminal state",
        "Household archetypes from the family essay. 2026 = 100 in real terms; trend growth of 1.5%/yr is included, so a flat line means falling behind.",
        "households.png",
    )
    mechanism_figure(df, "mechanism.png")
    drawdown_figure(df, "drawdown.png")
    written = ["macro.png", "households.png", "mechanism.png", "drawdown.png"]
    upath = os.path.join(OUT, "uncertainty.csv")
    if os.path.exists(upath):
        import json
        with open(os.path.join(OUT, "params.json")) as fh:
            p_wrong = json.load(fh)["P_PREMISE_WRONG"]
        fan_figure(pd.read_csv(upath), "fan.png", p_wrong)
        written.append("fan.png")
    print("wrote", ", ".join(written))


if __name__ == "__main__":
    main()
