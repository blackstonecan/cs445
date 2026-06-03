"""CNN actor-critic network for BrainBlock.

Architecture (justified in the report):
    board (B,1,5,8) --> Conv3x3(1->32) --> Conv3x3(32->64) --> flatten
                                                                  |
    piece one-hot (B,5)  ----------------------------------+      |
    inventory counts (B,5) --------------------------------+-- concat
                                                                  |
                                              shared MLP (Linear->ReLU)
                                                 |-- actor head -> 320 logits
                                                 |-- critic head -> scalar V

The board goes through convolutions (spatial inductive bias for a spatial
packing problem); the non-spatial piece/inventory vectors are concatenated after
the conv flatten rather than forced through convolutions. No action masking:
the actor produces logits over all 320 actions and must learn legality.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .env import H, W, N_ACTIONS
from .pieces import NUM_TYPES


def _orthogonal(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 256, conv_channels=(32, 64)):
        super().__init__()
        c1, c2 = conv_channels
        self.conv = nn.Sequential(
            _orthogonal(nn.Conv2d(1, c1, kernel_size=3, padding=1)),
            nn.ReLU(),
            _orthogonal(nn.Conv2d(c1, c2, kernel_size=3, padding=1)),
            nn.ReLU(),
        )
        conv_flat = c2 * H * W
        extra = NUM_TYPES * 2  # piece one-hot + inventory counts

        self.trunk = nn.Sequential(
            _orthogonal(nn.Linear(conv_flat + extra, hidden)),
            nn.ReLU(),
            _orthogonal(nn.Linear(hidden, hidden)),
            nn.ReLU(),
        )
        # Small actor gain (0.01) -> near-uniform initial policy (good for PPO).
        self.actor = _orthogonal(nn.Linear(hidden, N_ACTIONS), gain=0.01)
        self.critic = _orthogonal(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, obs):
        """obs: dict of tensors {board:(B,1,H,W), piece:(B,5), inventory:(B,5)}."""
        feat = self.conv(obs["board"])
        feat = feat.flatten(start_dim=1)
        x = torch.cat([feat, obs["piece"], obs["inventory"]], dim=1)
        x = self.trunk(x)
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    # ---- convenience wrappers used by the PPO trainer ----------------------
    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        """Sample (or argmax) an action. Returns action, logprob, value tensors."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, obs, actions):
        """For PPO update: log-probs, entropy, and values of given actions."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


def obs_to_tensor(obs, device="cpu"):
    """Convert a single env obs (numpy dict) to a batched tensor dict (B=1)."""
    return {
        "board": torch.as_tensor(obs["board"], dtype=torch.float32, device=device).unsqueeze(0),
        "piece": torch.as_tensor(obs["piece"], dtype=torch.float32, device=device).unsqueeze(0),
        "inventory": torch.as_tensor(obs["inventory"], dtype=torch.float32, device=device).unsqueeze(0),
    }


def stack_obs(obs_list, device="cpu"):
    """Stack a list of numpy obs dicts into a batched tensor dict."""
    return {
        "board": torch.as_tensor(np.stack([o["board"] for o in obs_list]),
                                 dtype=torch.float32, device=device),
        "piece": torch.as_tensor(np.stack([o["piece"] for o in obs_list]),
                                 dtype=torch.float32, device=device),
        "inventory": torch.as_tensor(np.stack([o["inventory"] for o in obs_list]),
                                     dtype=torch.float32, device=device),
    }


if __name__ == "__main__":
    from .env import BrainBlockEnv
    env = BrainBlockEnv()
    obs, _ = env.reset(seed=0)
    net = ActorCritic()
    t = obs_to_tensor(obs)
    logits, value = net(t)
    a, lp, v = net.act(t)
    print("logits", tuple(logits.shape), "value", tuple(value.shape))
    print("sampled action", int(a), "logprob", float(lp), "value", float(v))
    n_params = sum(p.numel() for p in net.parameters())
    print(f"param count: {n_params:,}")
