"""Figures for the BrainBlock report.

Produces:
    * Learning-curve panels (averaged over seeds, +/- std band) comparing the
      two reward functions: total reward, covered area, episode length, and
      invalid-action rate vs. training progress.
    * A gallery of distinct solutions found by the agent (colored tilings).
    * A step-trace figure showing one rollout placing pieces one at a time.

Usage:
    python visualize.py --runs-dir runs --results-dir results --out figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from brainblock.pieces import PIECE_TYPES
from brainblock.env import H, W

# Fixed color per piece type (index 0..4), plus light gray for empty (-1).
_PIECE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
_EMPTY_COLOR = "#EEEEEE"


# --------------------------------------------------------------- curves
def load_histories(runs_dir, mode):
    hs = []
    for ck in sorted(glob.glob(os.path.join(runs_dir, f"{mode}_seed*.pt"))):
        data = torch.load(ck, map_location="cpu", weights_only=False)
        hs.append(data["history"])
    return hs


def _series(histories, key):
    """Stack a metric across seeds into (n_seeds, T); truncate to min length."""
    seqs = [[rec.get(key, np.nan) for rec in h] for h in histories]
    T = min(len(s) for s in seqs)
    arr = np.array([s[:T] for s in seqs], dtype=float)
    return arr


def _episode_axis(histories):
    """Cumulative episode count (x-axis) from the shortest run."""
    T = min(len(h) for h in histories)
    n_ep = np.array([h[t].get("n_episodes", 0) for t in range(T)
                     for h in [histories[0]]])
    return np.cumsum(n_ep)


def _curriculum_zero_x(histories):
    """Episode-axis position where the curriculum floor (prefill_low) first
    reaches 0, i.e. when empty-board episodes begin. Returns None if absent."""
    h = histories[0]
    x = _episode_axis(histories)
    for t in range(min(len(h), len(x))):
        if h[t].get("prefill_low", 0) == 0:
            return x[t]
    return None


def _plot_curve_grid(histories_by_mode, panels, suptitle, path, mark_curriculum=True):
    colors = {"dense": "#1f77b4", "sparse": "#d62728"}
    n = len(panels)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, title) in zip(axes, panels):
        for mode, hs in histories_by_mode.items():
            arr = _series(hs, key)
            x = _episode_axis(hs)
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            c = colors.get(mode)
            ax.plot(x, mean, label=f"{mode} (n={len(hs)})", color=c)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=c)
            if mark_curriculum:
                zx = _curriculum_zero_x(hs)
                if zx is not None:
                    ax.axvline(zx, color=c, ls=":", lw=1, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Episode #")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(len(panels), len(axes)):
        axes[j].axis("off")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)


def plot_learning_curves(runs_dir, modes, out_dir):
    histories_by_mode = {}
    for mode in modes:
        h = load_histories(runs_dir, mode)
        if h:
            histories_by_mode[mode] = h
    if not histories_by_mode:
        print("no run histories found in", runs_dir)
        return

    # (1) Real empty-board task (pf0_* metrics). Flat until the curriculum floor
    # reaches 0 (dotted line) because no empty-board episodes exist before then;
    # the agent has already acquired the skills, so success rises quickly after.
    pf0_panels = [
        ("pf0_return_mean", "Total reward (empty board)"),
        ("pf0_covered_mean", "Total covered area"),
        ("pf0_ep_len_mean", "Episode length"),
        ("pf0_invalid_rate", "Invalid-action rate"),
        ("pf0_success_rate", "Success rate"),
        ("entropy", "Policy entropy"),
    ]
    _plot_curve_grid(
        histories_by_mode, pf0_panels,
        "BrainBlock PPO learning curves -- real empty-board task "
        "(mean +/- std over seeds; dotted = curriculum floor reaches 0)",
        os.path.join(out_dir, "learning_curves.png"))

    # (2) Training progress over the full curriculum mixture (all difficulties).
    # Shows the gradual rise in reward/length and fall in invalid rate as the
    # agent improves and the curriculum hardens.
    mixed_panels = [
        ("ep_return_mean", "Total reward (training mixture)"),
        ("covered_mean", "Total covered area (training mixture)"),
        ("ep_len_mean", "Episode length (training mixture)"),
        ("invalid_rate", "Invalid-action rate (training mixture)"),
        ("success_rate", "Success rate (training mixture)"),
        ("entropy", "Policy entropy"),
    ]
    _plot_curve_grid(
        histories_by_mode, mixed_panels,
        "BrainBlock PPO training progress -- curriculum mixture "
        "(mean +/- std over seeds)",
        os.path.join(out_dir, "learning_curves_training.png"))


# --------------------------------------------------------------- boards
def _draw_board(ax, label_board, title=None):
    label_board = np.asarray(label_board)
    cmap = ListedColormap([_EMPTY_COLOR] + _PIECE_COLORS)
    # Map -1 (empty) -> 0, type t -> t+1.
    grid = label_board + 1
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(_PIECE_COLORS),
              origin="upper", aspect="equal")
    for r in range(H):
        for c in range(W):
            v = label_board[r, c]
            if v >= 0:
                ax.text(c, r, PIECE_TYPES[v], ha="center", va="center",
                        fontsize=8, color="white", weight="bold")
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def plot_solution_gallery(solutions, out_dir, tag, max_show=6):
    sols = solutions[:max_show]
    if not sols:
        print("no solutions to plot for", tag)
        return
    n = len(sols)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, sol in enumerate(sols):
        _draw_board(axes[i], sol["label_board"],
                    title=f"Solution {i+1} (seed {sol.get('seed','?')})")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Distinct solutions found by the agent ({tag})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"solutions_{tag}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)


def plot_step_trace(solution, out_dir, tag):
    """Render the incremental board after each placement of one solution."""
    from brainblock.pieces import get_cells
    trace = solution["trace"]
    n = len(trace)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 1.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    label = np.full((H, W), -1, dtype=int)
    for i, (t, o, x, y) in enumerate(trace):
        for dy, dx in get_cells(t, o):
            label[y + dy, x + dx] = t
        _draw_board(axes[i], label.copy(),
                    title=f"step {i+1}: place {PIECE_TYPES[t]}")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Qualitative rollout: step-by-step solution ({tag})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"step_trace_{tag}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)


def parse_args():
    p = argparse.ArgumentParser(description="Make BrainBlock report figures")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--modes", nargs="+", default=["dense", "sparse"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    plot_learning_curves(args.runs_dir, args.modes, args.out)

    for mode in args.modes:
        sol_path = os.path.join(args.results_dir, f"solutions_{mode}.json")
        if os.path.exists(sol_path):
            with open(sol_path) as f:
                sols = json.load(f)
            plot_solution_gallery(sols, args.out, mode)
            if sols:
                plot_step_trace(sols[0], args.out, mode)


if __name__ == "__main__":
    main()
