"""Evaluate trained BrainBlock PPO agents.

Reports the metrics required by the assignment's evaluation protocol:
    * success rate (fraction of episodes solved),
    * mean and std of episodic return,
    * mean episode length,
    * invalid-action rate,
and collects >= 5 *distinct* solutions (deduplicated by final tiling) with full
step traces, saved for visualization.

Distinct solutions come from the shuffled queue (different seeds -> different
piece orders) plus the policy's residual stochasticity (entropy). We therefore
*sample* from the policy at eval by default rather than taking the argmax.

Examples:
    python evaluate.py --ckpt runs/dense_seed0.pt --episodes 300
    python evaluate.py --reward-mode dense --runs-dir runs   # aggregate seeds
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from brainblock.env import BrainBlockEnv
from brainblock.model import obs_to_tensor
from brainblock.ppo import PPOTrainer


@torch.no_grad()
def run_episodes(net, reward_mode, n_episodes, base_seed=10_000,
                 deterministic=False, temperature=1.0, device="cpu",
                 collect_solutions=True, max_solutions=20):
    env = BrainBlockEnv(reward_mode=reward_mode)
    returns, lengths, successes, invalid_steps, total_steps, covered = \
        [], [], [], 0, 0, []
    solutions = {}  # signature -> dict(trace, label_board)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        done = False
        ep_ret, ep_len = 0.0, 0
        trace = []  # (piece_type, orient, x, y) per accepted step
        while not done:
            t = obs_to_tensor(obs, device=device)
            logits, _ = net(t)
            if deterministic:
                action = int(logits.argmax(dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=logits / temperature)
                action = int(dist.sample().item())
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_ret += reward
            ep_len += 1
            total_steps += 1
            if info["invalid"]:
                invalid_steps += 1
            if info["valid"] and not info["invalid"] and env.placements:
                trace.append(env.placements[-1])
        returns.append(ep_ret)
        lengths.append(ep_len)
        successes.append(1.0 if info["is_success"] else 0.0)
        covered.append(info["covered_fraction"])
        if collect_solutions and info["is_success"]:
            sig = tuple(env.label_board.flatten().tolist())
            if sig not in solutions and len(solutions) < max_solutions:
                solutions[sig] = {
                    "trace": [list(map(int, p)) for p in trace],
                    "label_board": env.label_board.tolist(),
                    "seed": base_seed + ep,
                }

    r = np.asarray(returns)
    metrics = {
        "n_episodes": n_episodes,
        "success_rate": float(np.mean(successes)),
        "return_mean": float(r.mean()),
        "return_std": float(r.std()),
        "ep_len_mean": float(np.mean(lengths)),
        "covered_mean": float(np.mean(covered)),
        "invalid_action_rate": invalid_steps / max(1, total_steps),
        "n_distinct_solutions": len(solutions),
    }
    return metrics, list(solutions.values())


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate BrainBlock PPO agents")
    p.add_argument("--ckpt", help="single checkpoint to evaluate")
    p.add_argument("--reward-mode", default="dense", choices=["dense", "sparse"])
    p.add_argument("--runs-dir", default="runs",
                   help="aggregate all {reward_mode}_seed*.pt in this dir")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="softmax temperature for sampling (>1 = more diverse)")
    p.add_argument("--out", default="results", help="output dir for json/solutions")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.ckpt:
        ckpts = [args.ckpt]
    else:
        ckpts = sorted(glob.glob(os.path.join(args.runs_dir, f"{args.reward_mode}_seed*.pt")))
    if not ckpts:
        raise SystemExit("No checkpoints found to evaluate.")

    per_seed = []
    best_solutions, best_n = [], -1
    for ck in ckpts:
        net, ckpt = PPOTrainer.load_model(ck, device="cpu")
        mode = ckpt["cfg"]["reward_mode"]
        metrics, sols = run_episodes(net, mode, args.episodes,
                                     deterministic=args.deterministic,
                                     temperature=args.temperature)
        metrics["checkpoint"] = os.path.basename(ck)
        per_seed.append(metrics)
        print(f"{os.path.basename(ck):22s} | succ {metrics['success_rate']:.3f} "
              f"ret {metrics['return_mean']:6.2f}+-{metrics['return_std']:.2f} "
              f"len {metrics['ep_len_mean']:4.2f} "
              f"cov {metrics['covered_mean']:.3f} "
              f"inval {metrics['invalid_action_rate']:.3f} "
              f"distinct {metrics['n_distinct_solutions']}")
        if len(sols) > best_n:
            best_n, best_solutions = len(sols), sols

    # Aggregate across seeds (the >=5 random seeds protocol).
    def agg(key):
        vals = np.asarray([m[key] for m in per_seed])
        return float(vals.mean()), float(vals.std())

    summary = {
        "reward_mode": args.reward_mode if not args.ckpt else per_seed[0].get("checkpoint"),
        "n_seeds": len(per_seed),
        "deterministic": args.deterministic,
        "episodes_per_seed": args.episodes,
        "success_rate_mean_std": agg("success_rate"),
        "return_mean_std": agg("return_mean"),
        "ep_len_mean_std": agg("ep_len_mean"),
        "covered_mean_std": agg("covered_mean"),
        "invalid_action_rate_mean_std": agg("invalid_action_rate"),
        "per_seed": per_seed,
    }
    tag = os.path.splitext(os.path.basename(args.ckpt))[0] if args.ckpt else args.reward_mode
    with open(os.path.join(args.out, f"eval_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, f"solutions_{tag}.json"), "w") as f:
        json.dump(best_solutions, f, indent=2)

    print("\n=== Aggregate over %d seed(s) ===" % len(per_seed))
    for k in ["success_rate", "return_mean", "ep_len_mean", "covered_mean",
              "invalid_action_rate"]:
        m, s = agg(k)
        print(f"  {k:22s}: {m:.3f} +/- {s:.3f}")
    print(f"  distinct solutions collected: {len(best_solutions)}")
    print(f"saved eval_{tag}.json and solutions_{tag}.json to {args.out}/")


if __name__ == "__main__":
    main()
