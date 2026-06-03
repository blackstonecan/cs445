# BrainBlock Packing — Deep RL (CS 445/545)

A from-scratch Deep RL solution to the BrainBlock tetromino packing puzzle:
place 10 tetrominoes (2× each of I, O, L, Z, T) from a shuffled queue onto an
**8×5** board so it is fully tiled. Solved with **PPO** (actor-critic, CNN) in a
custom **Gymnasium** environment.

## Approach in one paragraph

The piece geometry `(type, orientation) → cells` is a *deterministic* function,
so it is computed analytically (a lookup table), not learned — that table drives
legality checks, board updates, and rendering. All learning capacity goes to the
**packing policy**: a CNN actor-critic trained with PPO. The action space is the
required `8 × 8 × 5 = 320` discrete `(orientation, x, y)`. We deliberately use
**no action masking**: the agent must *learn* legality, and an invalid placement
**hard-terminates** the episode — which makes the invalid-action rate a genuine
learning signal. Solution **diversity** (≥5 distinct solutions) comes from the
shuffled queue (different seeds → different piece orders) plus an entropy bonus,
with **stochastic sampling** at evaluation.

## Layout

```
brainblock/
  pieces.py     # deterministic 5×8 tetromino geometry table (4×4 box)
  env.py        # Gymnasium env: 320 actions, no mask, hard-terminate, 2 rewards
  model.py      # CNN actor-critic (conv board trunk + piece/inventory vectors)
  ppo.py        # PPO from scratch: GAE, clipped loss, entropy bonus, logging
  solver.py     # exact-cover backtracking solver (verification/oracle only)
train.py        # train PPO over (reward_mode, seed) grid -> runs/*.pt
evaluate.py     # deterministic/stochastic rollouts, metrics, distinct solutions
visualize.py    # learning curves, solution gallery, step-trace figures
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install gymnasium numpy matplotlib
```

## Reproduce the experiments

The `--curriculum` flag (reverse curriculum) is what lets the agent learn to
solve the full board; without it, plain PPO plateaus around 60% coverage and
never completes (see report §5). A moderate entropy coefficient (`0.03`) keeps
the policy from collapsing onto a single tiling, so it finds many distinct
solutions.

```bash
# 1. Train: 2 reward functions × 5 seeds (CPU; ~17 min per 8M-step run)
.venv/bin/python train.py \
    --reward-modes dense sparse --seeds 0 1 2 3 4 \
    --timesteps 8000000 --n-envs 32 --rollout-steps 128 \
    --ent-coef 0.03 --no-anneal-lr --curriculum --out-dir runs

# 2. Evaluate each reward mode, aggregated over its 5 seeds
#    (always starts from an empty board; samples for solution diversity)
.venv/bin/python evaluate.py --reward-mode dense  --episodes 300 --runs-dir runs
.venv/bin/python evaluate.py --reward-mode sparse --episodes 300 --runs-dir runs

# 3. Figures: learning curves + solution gallery + step trace
.venv/bin/python visualize.py --runs-dir runs

# (single quick run, one agent)
.venv/bin/python train.py --reward-modes dense --seeds 0 \
    --timesteps 8000000 --n-envs 32 --rollout-steps 128 \
    --ent-coef 0.03 --no-anneal-lr --curriculum --out-dir runs
```

Checkpoints land in `runs/`, metrics/solutions JSON in `results/`, figures in
`figures/`.

## MDP summary

| Component | Definition |
|-----------|------------|
| **State** | 1-channel 8×5 board (filled/empty), current-piece one-hot (5), remaining-inventory counts (5) |
| **Action** | Discrete 320 = orientation(8) × x(8) × y(5), decoded `o·40 + x·5 + y` |
| **Transition** | Place current queue head if legal (in-bounds, non-overlapping); else terminate |
| **Reward** | Two variants — `dense` (+1/placement, +10 completion, −1 invalid) and `sparse` (+1 completion only) |
| **Termination** | Full tiling (success) or any invalid placement (hard-terminate) |

## Self-tests

```bash
.venv/bin/python -m brainblock.pieces   # geometry: 4×4 fit, unique-orient counts
.venv/bin/python -m brainblock.env      # random-legal rollout sanity
.venv/bin/python -m brainblock.model    # network shapes / param count
```

## Live demo

```bash
# Watch a pretrained agent solve, step by step, in the terminal.
# A ready pretrained model ships in models/ (no training needed):
.venv/bin/python -m brainblock.demo --ckpt models/dense_seed0.pt
```
