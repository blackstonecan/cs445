"""Exhaustively find and visualize all distinct solutions a trained agent can
produce, across *every* distinct piece ordering.

Unlike ``evaluate.py`` (which caps distinct solutions at 20 and samples additive
seeds), this script:

    * enumerates all 113,400 distinct queue orderings (10 pieces = 2x each of
      I,O,L,Z,T  ->  10! / 2!^5 = 113,400),
    * runs each ordering through the policy with batched (vectorized) rollouts,
    * collects the *full, uncapped* set of distinct final tilings,
    * saves them to JSON and renders gallery figure(s).

Determinism vs. diversity is controlled by ``--temperature``:
    * ``--deterministic`` (argmax): one solution per ordering; the agent's
      intrinsic, reproducible behavior (few distinct solutions, ~peaky policy).
    * ``--temperature T`` with ``--samples K``: sample K rollouts per ordering at
      softmax temperature T; higher T / more samples -> more distinct solutions,
      at a small cost in per-rollout success rate.

Usage:
    python enumerate_solutions.py --ckpt models/dense_seed0.pt --deterministic
    python enumerate_solutions.py --ckpt runs/sparse_seed1.pt --temperature 1.0 --samples 3
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from brainblock.pieces import PIECE_CELLS, NUM_TYPES
from brainblock.env import H, W, N_CELLS
from brainblock.ppo import PPOTrainer
from visualize import _draw_board  # reuse the report's board styling

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Piece cell offsets as a dense array: CELL[type, orient] = 4 x (dy, dx).
CELL = np.zeros((NUM_TYPES, 8, 4, 2), dtype=np.int64)
for _ti in range(NUM_TYPES):
    for _oi in range(8):
        CELL[_ti, _oi] = np.array(PIECE_CELLS[_ti][_oi], dtype=np.int64)


def all_orderings():
    """All distinct queue orderings of [2x each of the 5 piece types]."""
    from sympy.utilities.iterables import multiset_permutations
    base = [t for t in range(NUM_TYPES) for _ in range(2)]  # [0,0,1,1,2,2,3,3,4,4]
    return np.array(list(multiset_permutations(base)), dtype=np.int64)


@torch.no_grad()
def rollout_batch(net, orders, temperature, deterministic, device="cpu"):
    """Play one batched rollout (one action choice per ordering).

    Returns (label_boards (B,H,W) int8, solved_mask (B,) bool, traces list).
    A trace is the list of (type, orient, x, y) placements for that ordering.
    """
    B = len(orders)
    board = np.zeros((B, H, W), np.float32)
    label = np.full((B, H, W), -1, np.int8)
    ptr = np.zeros(B, np.int64)
    dead = np.zeros(B, bool)
    traces = [[] for _ in range(B)]

    for _ in range(10):  # at most 10 pieces
        if dead.all():
            break
        cur = orders[np.arange(B), np.clip(ptr, 0, 9)]
        piece = np.zeros((B, NUM_TYPES), np.float32)
        piece[np.arange(B), cur] = 1.0
        inv = np.zeros((B, NUM_TYPES), np.float32)
        for ti in range(NUM_TYPES):
            inv[:, ti] = [(orders[i, ptr[i]:] == ti).sum() for i in range(B)]

        obs = {
            "board": torch.from_numpy(board[:, None]).to(device),
            "piece": torch.from_numpy(piece).to(device),
            "inventory": torch.from_numpy(inv).to(device),
        }
        logits, _ = net(obs)
        if deterministic:
            a = logits.argmax(-1).cpu().numpy()
        else:
            a = torch.distributions.Categorical(
                logits=logits / temperature).sample().cpu().numpy()

        orient = a // (W * H)
        rem = a % (W * H)
        x = rem // H
        y = rem % H
        for i in range(B):
            if dead[i] or ptr[i] >= 10:
                continue
            ti = cur[i]
            cells = CELL[ti, orient[i]]
            ok = True
            for dy, dx in cells:
                ny, nx = y[i] + dy, x[i] + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W or board[i, ny, nx]:
                    ok = False
                    break
            if not ok:
                dead[i] = True
                continue
            for dy, dx in cells:
                board[i, y[i] + dy, x[i] + dx] = 1
                label[i, y[i] + dy, x[i] + dx] = ti
            traces[i].append((int(ti), int(orient[i]), int(x[i]), int(y[i])))
            ptr[i] += 1

    solved = ptr == 10
    return label, solved, traces


def find_all_solutions(net, orders, temperature, deterministic, samples,
                       chunk, seed, device="cpu"):
    """Collect every distinct solved tiling across all orderings.

    Returns (solutions list, n_solved_rollouts, n_total_rollouts).
    Each solution dict: {label_board, trace, ordering}.
    """
    torch.manual_seed(seed)
    distinct = {}  # signature (bytes) -> solution dict
    n_solved = 0
    n_total = 0
    reps = 1 if deterministic else max(1, samples)
    for rep in range(reps):
        for s in range(0, len(orders), chunk):
            q = orders[s:s + chunk]
            label, solved, traces = rollout_batch(
                net, q, temperature, deterministic, device)
            n_total += len(q)
            n_solved += int(solved.sum())
            for i in np.where(solved)[0]:
                sig = label[i].tobytes()
                if sig not in distinct:
                    distinct[sig] = {
                        "label_board": label[i].tolist(),
                        "trace": traces[i],
                        "ordering": q[i].tolist(),
                    }
    # Stable order: sort by the flattened tiling signature.
    sols = [distinct[k] for k in sorted(distinct)]
    return sols, n_solved, n_total


def plot_all_solutions(solutions, out_dir, tag, max_per_fig, title_prefix):
    """Render every distinct solution into one or more grid figures."""
    if not solutions:
        print("no solutions to plot for", tag)
        return []
    paths = []
    n = len(solutions)
    n_pages = (n + max_per_fig - 1) // max_per_fig
    for page in range(n_pages):
        chunk = solutions[page * max_per_fig:(page + 1) * max_per_fig]
        m = len(chunk)
        cols = min(8, m)
        rows = (m + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 1.5 * rows))
        axes = np.atleast_1d(axes).ravel()
        for i, sol in enumerate(chunk):
            idx = page * max_per_fig + i + 1
            _draw_board(axes[i], sol["label_board"], title=f"#{idx}")
        for j in range(m, len(axes)):
            axes[j].axis("off")
        suffix = "" if n_pages == 1 else f" (page {page + 1}/{n_pages})"
        fig.suptitle(f"{title_prefix}: {n} distinct solutions{suffix}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        name = f"all_solutions_{tag}.png" if n_pages == 1 else \
               f"all_solutions_{tag}_p{page + 1}.png"
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
        print("saved", path)
    return paths


def parse_args():
    p = argparse.ArgumentParser(
        description="Exhaustively find & visualize all distinct solutions")
    p.add_argument("--ckpt", required=True, help="checkpoint .pt to evaluate")
    p.add_argument("--deterministic", action="store_true",
                   help="argmax (one reproducible solution per ordering)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="softmax temperature when sampling (ignored if --deterministic)")
    p.add_argument("--samples", type=int, default=1,
                   help="sampled rollouts per ordering (more -> more solutions)")
    p.add_argument("--chunk", type=int, default=10000, help="batch size")
    p.add_argument("--seed", type=int, default=0, help="sampling RNG seed")
    p.add_argument("--max-per-fig", type=int, default=64,
                   help="distinct solutions per gallery figure")
    p.add_argument("--out", default="figures", help="figure output dir")
    p.add_argument("--results", default="results", help="json output dir")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    net, ckpt = PPOTrainer.load_model(args.ckpt, device="cpu")
    mode = ckpt["cfg"]["reward_mode"]
    base = os.path.splitext(os.path.basename(args.ckpt))[0]
    if args.deterministic:
        tag = f"{base}_argmax"
        title = f"{base} (argmax)"
    else:
        tag = f"{base}_t{args.temperature:g}_s{args.samples}"
        title = f"{base} (temp={args.temperature:g}, {args.samples} sample/order)"

    orders = all_orderings()
    print(f"orderings: {len(orders):,}  | checkpoint: {args.ckpt} (reward={mode})")
    print(f"mode: {'argmax' if args.deterministic else f'sample temp={args.temperature:g} x{args.samples}'}")

    t0 = time.time()
    sols, n_solved, n_total = find_all_solutions(
        net, orders, args.temperature, args.deterministic,
        args.samples, args.chunk, args.seed)
    dt = time.time() - t0

    print(f"\nrollouts: {n_solved:,}/{n_total:,} solved "
          f"({100 * n_solved / n_total:.1f}%)")
    print(f"DISTINCT solutions found: {len(sols):,}   ({dt:.0f}s)")

    sol_path = os.path.join(args.results, f"all_solutions_{tag}.json")
    with open(sol_path, "w") as f:
        json.dump({"checkpoint": os.path.basename(args.ckpt),
                   "reward_mode": mode,
                   "deterministic": args.deterministic,
                   "temperature": None if args.deterministic else args.temperature,
                   "samples": 1 if args.deterministic else args.samples,
                   "n_orderings": int(len(orders)),
                   "rollout_success_rate": n_solved / n_total,
                   "n_distinct_solutions": len(sols),
                   "solutions": sols}, f)
    print("saved", sol_path)

    plot_all_solutions(sols, args.out, tag, args.max_per_fig, title)


if __name__ == "__main__":
    main()
