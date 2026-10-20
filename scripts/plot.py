"""Learning curves: validation cost vs training time, with optional baseline reference lines.

    uv run python scripts/plot.py --runs runs/A runs/B --labels "critic" "shared" \
        --hline "OR-Tools GLS=16.12" --hline "ACO=16.30" --out n20_baselines
Writes results/plots/<out>.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def read_log(run):
    with open(Path(run) / "log.csv") as fh:
        rows = list(csv.DictReader(fh))
    return [float(r["minutes"]) for r in rows], [float(r["val_cost"]) for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="*")
    p.add_argument("--hline", action="append", default=[], help="label=value")
    p.add_argument("--title", default="Validation cost during training (lower is better)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    labels = a.labels or [Path(r).name for r in a.runs]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (run, lab) in enumerate(zip(a.runs, labels)):
        x, y = read_log(run)
        ax.plot(x, y, color=SERIES[i], lw=2, label=lab, solid_capstyle="round")
        best = min(range(len(y)), key=y.__getitem__)
        ax.plot(x[best], y[best], "o", ms=6, color=SERIES[i], mec=SURFACE, mew=2)
        ax.annotate(f"{lab}  {y[best]:.3f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    for i, h in enumerate(sorted(a.hline, key=lambda s: -float(s.rsplit("=", 1)[1]))):
        name, val = h.rsplit("=", 1)
        ax.axhline(float(val), color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.annotate(f"{name}  {float(val):.3f}", (0, float(val)), xytext=(4, 3 if i % 2 == 0 else -11),
                    textcoords="offset points", fontsize=8, color=INK2, xycoords=("axes fraction", "data"))
    ax.set_xlabel("training time (minutes)", color=INK2, fontsize=9)
    ax.set_ylabel("mean cost on validation days", color=INK2, fontsize=9)
    ax.set_title(a.title, color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    if len(a.runs) >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    out = ROOT / "results" / "plots" / f"{a.out}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
