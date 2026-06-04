"""Sampling-temperature sweep: success rate vs. solution diversity.

For a trained agent, the softmax temperature at evaluation trades off reliability
against the number of *distinct* solutions found. This script sweeps temperature,
runs many episodes per setting, and plots both curves on a dual axis.

Usage:
    python temperature_sweep.py                      # dense + sparse if present
    python temperature_sweep.py --runs 1000 --out figures
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brainblock.env import BrainBlockEnv
from brainblock.model import obs_to_tensor
from brainblock.ppo import PPOTrainer


@torch.no_grad()
def sweep_model(path, reward_mode, temps, n_runs, base_seed=200_000):
    net, _ = PPOTrainer.load_model(path)
    env = BrainBlockEnv(reward_mode=reward_mode)
    succ_rates, distinct_counts = [], []
    for temp in temps:
        solved, sols = 0, set()
        for ep in range(n_runs):
            obs, _ = env.reset(seed=base_seed + ep)  # same queues across temps
            done = False
            while not done:
                logits, _ = net(obs_to_tensor(obs))
                a = int(torch.distributions.Categorical(logits=logits / temp).sample().item())
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc
            if info["is_success"]:
                solved += 1
                sols.add(tuple(env.label_board.flatten().tolist()))
        succ_rates.append(solved / n_runs)
        distinct_counts.append(len(sols))
        print(f"  [{os.path.basename(path)}] temp {temp:.2f}: "
              f"success {solved/n_runs:.3f}, distinct {len(sols)}")
    return np.array(succ_rates), np.array(distinct_counts)


def _panel(ax, temps, succ, distinct, title):
    c_succ, c_div = "#1f77b4", "#d62728"
    ax.plot(temps, succ, "-o", color=c_succ, label="success rate")
    ax.set_xlabel("Sampling temperature")
    ax.set_ylabel("Success rate", color=c_succ)
    ax.tick_params(axis="y", labelcolor=c_succ)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(temps, distinct, "-s", color=c_div, label="distinct solutions")
    ax2.set_ylabel("Distinct solutions", color=c_div)
    ax2.tick_params(axis="y", labelcolor=c_div)
    ax2.set_ylim(0, max(distinct.max() * 1.15, 5))

    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_title(title)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=800)
    p.add_argument("--out", default="figures")
    p.add_argument("--temps", type=float, nargs="+",
                   default=[0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.2, 2.6])
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    temps = np.array(args.temps)

    # (label, checkpoint, reward_mode) — skip any that are missing.
    candidates = [
        ("R_dense agent", "models/dense_seed0.pt", "dense"),
        ("R_sparse agent", "runs/sparse_seed0.pt", "sparse"),
    ]
    panels = [(lbl, ck, rm) for (lbl, ck, rm) in candidates if os.path.exists(ck)]
    if not panels:
        raise SystemExit("No checkpoints found.")

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5.2),
                             squeeze=False)
    for ax, (lbl, ck, rm) in zip(axes[0], panels):
        print(f"sweeping {lbl} ({ck})")
        succ, distinct = sweep_model(ck, rm, temps, args.runs)
        _panel(ax, temps, succ, distinct, f"{lbl}  ({args.runs} runs/point)")

    fig.suptitle("Sampling temperature: reliability vs. solution diversity",
                 y=0.99, fontsize=13)
    # Shared legend below the title, above the panels.
    handles = [plt.Line2D([], [], color="#1f77b4", marker="o", label="success rate"),
               plt.Line2D([], [], color="#d62728", marker="s", label="distinct solutions"),
               plt.Line2D([], [], color="gray", ls=":", label="temperature = 1.0 (default)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.945), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path = os.path.join(args.out, "temperature_tradeoff.png")
    fig.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    main()
