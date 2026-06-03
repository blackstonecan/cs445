"""Live terminal demo: a pretrained agent solves BrainBlock step by step.

Required for the presentation (Section 7): the pretrained RL agent solves the
task visually on the spot. This renders the board after each accepted placement.

Usage:
    python -m brainblock.demo --ckpt runs/dense_seed0.pt
    python -m brainblock.demo --ckpt runs/dense_seed0.pt --seed 42 --delay 0.4
"""

from __future__ import annotations

import argparse
import time

import torch

from .env import BrainBlockEnv
from .model import obs_to_tensor
from .ppo import PPOTrainer


# ANSI colors per piece type for a nicer terminal render.
_COLORS = {
    "I": "\033[44m", "O": "\033[43m", "L": "\033[42m",
    "Z": "\033[41m", "T": "\033[45m",
}
_RESET = "\033[0m"


def render(env) -> str:
    from .pieces import PIECE_TYPES
    rows = []
    for r in env.label_board:
        cells = []
        for v in r:
            if v < 0:
                cells.append(" . ")
            else:
                name = PIECE_TYPES[v]
                cells.append(f"{_COLORS[name]} {name} {_RESET}")
        rows.append("".join(cells))
    head = PIECE_TYPES[env.queue[env.ptr]] if env.ptr < env.n_pieces else "-"
    rows.append(f"placed {env.ptr}/{env.n_pieces}   next piece: {head}")
    return "\n".join(rows)


@torch.no_grad()
def play_episode(net, env, seed, delay, deterministic):
    obs, info = env.reset(seed=seed)
    print(f"\n--- seed {seed} ---")
    print(render(env))
    done = False
    while not done:
        t = obs_to_tensor(obs)
        logits, _ = net(t)
        if deterministic:
            action = int(logits.argmax(-1).item())
        else:
            action = int(torch.distributions.Categorical(logits=logits).sample().item())
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        time.sleep(delay)
        print("\033[H\033[J", end="")  # clear screen
        print(render(env))
        if info["invalid"]:
            print("** invalid placement -> episode terminated **")
    return info["is_success"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seed", type=int, default=None,
                   help="fixed seed; default searches seeds for a solved demo")
    p.add_argument("--delay", type=float, default=0.4)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--max-tries", type=int, default=200)
    args = p.parse_args()

    net, ckpt = PPOTrainer.load_model(args.ckpt)
    env = BrainBlockEnv(reward_mode=ckpt["cfg"]["reward_mode"])

    if args.seed is not None:
        ok = play_episode(net, env, args.seed, args.delay, args.deterministic)
        print("SOLVED!" if ok else "did not solve this seed.")
        return

    # Search for a seed the agent solves, so the live demo is clean.
    for s in range(args.max_tries):
        ok = play_episode(net, env, s, args.delay, args.deterministic)
        if ok:
            print(f"\nSOLVED on seed {s}!  Board fully tiled.")
            return
    print("No solved seed found within max-tries; show a different checkpoint.")


if __name__ == "__main__":
    main()
