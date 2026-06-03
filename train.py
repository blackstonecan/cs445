"""Train PPO agents on BrainBlock.

Trains one run per (reward_mode, seed) combination and saves a checkpoint
(model weights + full metric history) per run under ``--out-dir``.

Examples:
    python train.py                      # both reward modes, seeds 0..4
    python train.py --reward-modes dense --seeds 0 --timesteps 1500000
"""

from __future__ import annotations

import argparse
import os

from brainblock.ppo import PPOTrainer, PPOConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on BrainBlock")
    p.add_argument("--reward-modes", nargs="+", default=["dense", "sparse"],
                   choices=["dense", "sparse"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--no-anneal-lr", action="store_true",
                   help="keep LR constant (sustains exploration)")
    p.add_argument("--curriculum", action="store_true",
                   help="reverse curriculum: start with pieces pre-placed, "
                        "gradually start from emptier boards as success rises")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for mode in args.reward_modes:
        for seed in args.seeds:
            tag = f"{mode}_seed{seed}"
            print(f"\n=== Training {tag} ({args.timesteps:,} steps) ===")
            cfg = PPOConfig(
                reward_mode=mode,
                total_timesteps=args.timesteps,
                n_envs=args.n_envs,
                rollout_steps=args.rollout_steps,
                lr=args.lr,
                ent_coef=args.ent_coef,
                anneal_lr=not args.no_anneal_lr,
                curriculum=args.curriculum,
                seed=seed,
                log_interval=max(1, (args.timesteps // (args.n_envs * args.rollout_steps)) // 20),
            )
            trainer = PPOTrainer(cfg)
            path = os.path.join(args.out_dir, f"{tag}.pt")
            # Periodic checkpointing so long runs are durable / inspectable.
            trainer.train(verbose=not args.quiet, save_path=path,
                          save_every=max(1, trainer.num_updates // 10))
            final = trainer.history[-1]
            print(f"saved {path} | final success_rate="
                  f"{final.get('success_rate', 0):.3f} "
                  f"covered={final.get('covered_mean', 0):.3f}")


if __name__ == "__main__":
    main()
