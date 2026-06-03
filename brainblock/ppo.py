"""PPO (clipped, with GAE) implemented from scratch in PyTorch for BrainBlock.

Single-file, dependency-light actor-critic PPO:
    * synchronous vectorized rollout over ``n_envs`` BrainBlock envs,
    * Generalized Advantage Estimation,
    * clipped surrogate policy loss + clipped value loss + entropy bonus,
    * per-update logging of all metrics the report requires (episodic return,
      success rate, episode length, invalid-action rate, covered area).

The entropy bonus is the diversity mechanism: it keeps the policy stochastic so
training explores many solutions, and at eval we sample (not argmax) across
seeds to surface distinct solutions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .env import BrainBlockEnv, N_ACTIONS
from .model import ActorCritic, stack_obs


@dataclass
class PPOConfig:
    reward_mode: str = "dense"
    total_timesteps: int = 1_000_000
    n_envs: int = 16
    rollout_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    update_epochs: int = 4
    num_minibatches: int = 4
    anneal_lr: bool = True
    norm_adv: bool = True
    clip_vloss: bool = True
    seed: int = 0
    device: str = "cpu"
    log_interval: int = 1
    # Reverse curriculum (training only; eval always starts from empty board).
    curriculum: bool = False
    curriculum_pool_size: int = 64
    curriculum_start: int = 9        # initial prefill ceiling (pieces pre-placed)
    curriculum_anneal_frac: float = 0.6  # fraction of training over which the
    #                                      prefill ceiling anneals 9 -> 0


class RunningEpisodeStats:
    """Tracks stats of episodes completed during a rollout."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.returns, self.lengths = [], []
        self.successes, self.invalids, self.covered = [], [], []
        # Real-task (prefill == 0) episodes only -> clean, unconfounded curves.
        self.pf0_ret, self.pf0_len = [], []
        self.pf0_succ, self.pf0_inv, self.pf0_cov = [], [], []

    def add(self, ret, length, info):
        self.returns.append(ret)
        self.lengths.append(length)
        self.successes.append(1.0 if info["is_success"] else 0.0)
        self.invalids.append(1.0 if info["invalid"] else 0.0)
        self.covered.append(info["covered_fraction"])
        if info.get("episode_prefill", 0) == 0:
            self.pf0_ret.append(ret)
            self.pf0_len.append(length)
            self.pf0_succ.append(1.0 if info["is_success"] else 0.0)
            self.pf0_inv.append(1.0 if info["invalid"] else 0.0)
            self.pf0_cov.append(info["covered_fraction"])

    def summary(self):
        if not self.returns:
            return None

        def m(xs):
            return float(np.mean(xs)) if xs else 0.0

        return {
            # Mixed over the curriculum difficulty distribution.
            "ep_return_mean": float(np.mean(self.returns)),
            "ep_return_std": float(np.std(self.returns)),
            "ep_len_mean": float(np.mean(self.lengths)),
            "success_rate": float(np.mean(self.successes)),
            "invalid_rate": float(np.mean(self.invalids)),
            "covered_mean": float(np.mean(self.covered)),
            "n_episodes": len(self.returns),
            # Real empty-board task only (use these for report learning curves).
            "pf0_success_rate": m(self.pf0_succ),
            "pf0_return_mean": m(self.pf0_ret),
            "pf0_ep_len_mean": m(self.pf0_len),
            "pf0_invalid_rate": m(self.pf0_inv),
            "pf0_covered_mean": m(self.pf0_cov),
            "pf0_n_episodes": len(self.pf0_succ),
        }


class PPOTrainer:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Optional reverse curriculum: build a pool of diverse completable
        # tilings and start each env with `curriculum_start` pieces pre-placed.
        self.tiling_pool = None
        self.prefill = 0           # upper bound (held at curriculum_start)
        self.prefill_low = 0       # lower bound (annealed curriculum_start -> 0)
        if cfg.curriculum:
            self.tiling_pool = self._build_tiling_pool(cfg.curriculum_pool_size)
            self.prefill = cfg.curriculum_start
            self.prefill_low = cfg.curriculum_start

        self.envs = []
        for _ in range(cfg.n_envs):
            e = BrainBlockEnv(reward_mode=cfg.reward_mode,
                              tiling_pool=self.tiling_pool,
                              prefill=self.prefill,
                              prefill_random=cfg.curriculum)
            e.set_prefill_low(self.prefill_low)
            self.envs.append(e)
        # Distinct per-env seeds for the initial reset; envs re-shuffle on reset.
        self._next_seed = cfg.seed * 100003 + 1
        self.cur_obs = [self._reset_env(e) for e in self.envs]
        self.ep_return = np.zeros(cfg.n_envs, dtype=np.float64)
        self.ep_len = np.zeros(cfg.n_envs, dtype=np.int64)

        self.net = ActorCritic().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)

        self.batch_size = cfg.n_envs * cfg.rollout_steps
        self.minibatch_size = self.batch_size // cfg.num_minibatches
        self.num_updates = cfg.total_timesteps // self.batch_size

        self.history = []  # list of per-update metric dicts
        self.global_step = 0

    def _build_tiling_pool(self, size):
        """Generate a pool of diverse completable tilings for the curriculum."""
        from .solver import find_tiling
        rng = np.random.default_rng(self.cfg.seed + 777)
        pool, seen, attempts = [], set(), 0
        while len(pool) < size and attempts < size * 30:
            attempts += 1
            t = find_tiling(rng=rng, max_nodes=3000)
            if t is None:
                continue
            sig = tuple(sorted(t))
            if sig in seen:
                continue
            seen.add(sig)
            pool.append(t)
        print(f"[curriculum] built tiling pool of {len(pool)} distinct tilings")
        return pool

    def _anneal_curriculum(self, update):
        """Hold the prefill upper bound at `curriculum_start` (easy completions
        always sampled) and anneal the lower bound 9 -> 0 over
        `curriculum_anneal_frac` of training, progressively adding harder boards
        down to the real task (level 0)."""
        if not self.cfg.curriculum:
            return
        frac = update / max(1, self.num_updates)
        low = self.cfg.curriculum_start * (1.0 - frac / self.cfg.curriculum_anneal_frac)
        low = max(0, int(round(low)))
        if low != self.prefill_low:
            self.prefill_low = low
            for e in self.envs:
                e.set_prefill_low(low)

    def _reset_env(self, env):
        seed = self._next_seed
        self._next_seed += 1
        obs, _ = env.reset(seed=seed)
        return obs

    # ----------------------------------------------------------- rollout
    def collect_rollout(self):
        cfg = self.cfg
        T, N = cfg.rollout_steps, cfg.n_envs
        dev = self.device

        # Storage.
        obs_board = torch.zeros((T, N, 1, *self.cur_obs[0]["board"].shape[1:]), device=dev)
        obs_piece = torch.zeros((T, N, self.cur_obs[0]["piece"].shape[0]), device=dev)
        obs_inv = torch.zeros((T, N, self.cur_obs[0]["inventory"].shape[0]), device=dev)
        actions = torch.zeros((T, N), dtype=torch.long, device=dev)
        logprobs = torch.zeros((T, N), device=dev)
        rewards = torch.zeros((T, N), device=dev)
        values = torch.zeros((T, N), device=dev)
        dones = torch.zeros((T, N), device=dev)

        stats = RunningEpisodeStats()

        for t in range(T):
            batch = stack_obs(self.cur_obs, device=dev)
            obs_board[t] = batch["board"]
            obs_piece[t] = batch["piece"]
            obs_inv[t] = batch["inventory"]

            with torch.no_grad():
                logits, value = self.net(batch)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
            values[t] = value
            actions[t] = action
            logprobs[t] = logprob

            a_np = action.cpu().numpy()
            for i, env in enumerate(self.envs):
                obs, reward, term, trunc, info = env.step(int(a_np[i]))
                done = term or trunc
                rewards[t, i] = reward
                dones[t, i] = 1.0 if done else 0.0
                self.ep_return[i] += reward
                self.ep_len[i] += 1
                if done:
                    stats.add(self.ep_return[i], int(self.ep_len[i]), info)
                    self.ep_return[i] = 0.0
                    self.ep_len[i] = 0
                    obs = self._reset_env(env)
                self.cur_obs[i] = obs
            self.global_step += N

        # Bootstrap value for the final next-obs of each env.
        with torch.no_grad():
            _, next_value = self.net(stack_obs(self.cur_obs, device=dev))

        # GAE.
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nextnonterminal = 1.0 - dones[t]
            nextvalues = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values

        # Flatten (T, N, ...) -> (T*N, ...).
        b = {
            "board": obs_board.reshape((-1,) + obs_board.shape[2:]),
            "piece": obs_piece.reshape((-1, obs_piece.shape[-1])),
            "inventory": obs_inv.reshape((-1, obs_inv.shape[-1])),
        }
        return (b, actions.reshape(-1), logprobs.reshape(-1),
                advantages.reshape(-1), returns.reshape(-1), values.reshape(-1),
                stats)

    # ----------------------------------------------------------- update
    def update(self, b_obs, b_actions, b_logprobs, b_adv, b_returns, b_values):
        cfg = self.cfg
        idx = np.arange(self.batch_size)
        clipfracs = []
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, self.batch_size, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                mb_obs = {k: v[mb] for k, v in b_obs.items()}
                new_logprob, entropy, new_value = self.net.evaluate_actions(
                    mb_obs, b_actions[mb])
                logratio = new_logprob - b_logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = b_adv[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped policy loss.
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # (Optionally clipped) value loss.
                if cfg.clip_vloss:
                    v_unclipped = (new_value - b_returns[mb]) ** 2
                    v_clipped = b_values[mb] + torch.clamp(
                        new_value - b_values[mb], -cfg.clip_coef, cfg.clip_coef)
                    v_clipped = (v_clipped - b_returns[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

        return {
            "pg_loss": float(pg_loss.item()),
            "v_loss": float(v_loss.item()),
            "entropy": float(entropy_loss.item()),
            "clipfrac": float(np.mean(clipfracs)),
        }

    # ----------------------------------------------------------- train loop
    def train(self, verbose: bool = True, save_path=None, save_every: int = 0):
        cfg = self.cfg
        last_summary = None
        start = time.time()
        for update in range(1, self.num_updates + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (update - 1.0) / self.num_updates
                for g in self.opt.param_groups:
                    g["lr"] = frac * cfg.lr

            (b_obs, b_actions, b_logprobs, b_adv,
             b_returns, b_values, stats) = self.collect_rollout()
            losses = self.update(b_obs, b_actions, b_logprobs,
                                 b_adv, b_returns, b_values)

            summary = stats.summary() or last_summary or {}
            last_summary = summary or last_summary
            self._anneal_curriculum(update)
            rec = {"update": update, "timestep": self.global_step,
                   "lr": self.opt.param_groups[0]["lr"],
                   "prefill_low": self.prefill_low, **summary, **losses}
            self.history.append(rec)

            if verbose and (update % cfg.log_interval == 0 or update == 1):
                sps = int(self.global_step / (time.time() - start))
                pf = f"pf[{self.prefill_low}..{self.prefill}] " if cfg.curriculum else ""
                p0 = (f"pf0succ {summary.get('pf0_success_rate', 0):5.2f} "
                      if cfg.curriculum else "")
                print(f"upd {update:4d}/{self.num_updates} "
                      f"step {self.global_step:>8d} "
                      f"{pf}{p0}"
                      f"ret {summary.get('ep_return_mean', float('nan')):7.2f} "
                      f"succ {summary.get('success_rate', 0):5.2f} "
                      f"cov {summary.get('covered_mean', 0):4.2f} "
                      f"inv {summary.get('invalid_rate', 0):4.2f} "
                      f"len {summary.get('ep_len_mean', 0):4.1f} "
                      f"ent {losses['entropy']:5.3f} "
                      f"{sps} sps")

            if save_path and save_every and update % save_every == 0:
                self.save(save_path)
        if save_path:
            self.save(save_path)
        return self.history

    # ----------------------------------------------------------- io
    def save(self, path):
        torch.save({"model": self.net.state_dict(),
                    "cfg": asdict(self.cfg),
                    "history": self.history}, path)

    @staticmethod
    def load_model(path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        net = ActorCritic().to(device)
        net.load_state_dict(ckpt["model"])
        net.eval()
        return net, ckpt
