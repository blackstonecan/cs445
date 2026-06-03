"""Gymnasium environment for the BrainBlock 8x5 tetromino packing puzzle.

Design decisions (justified in the report):
    * Single-channel board observation (1 = filled, 0 = empty), plus the current
      piece (one-hot over 5 types) and the remaining-inventory counts.
    * Discrete action space of size 8 x W x H = 320, decoded as
      (orientation, x_anchor, y_anchor) -- equivalent to the Cartesian product
      A = {0..7} x {0..W-1} x {0..H-1} required by the assignment.
    * No action masking: the policy must *learn* legality. An invalid placement
      hard-terminates the episode. This makes the invalid-action rate a genuine
      learning signal.
    * Two selectable reward functions ("dense" and "sparse") for the required
      reward-function comparison.

Coordinate convention follows ``pieces.py``: cell = (row, col) = (y, x), board
shape (H, W) = (5, 8), anchor (y, x), piece offsets (dy, dx).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .pieces import (
    PIECE_TYPES,
    NUM_TYPES,
    NUM_ORIENTS,
    INVENTORY,
    get_cells,
)

# Board geometry: W x H = 8 x 5 (40 cells).
W, H = 8, 5
N_CELLS = W * H
N_ACTIONS = NUM_ORIENTS * W * H  # 8 * 8 * 5 = 320

# Two distinct reward functions to compare (Section 3.3 of the assignment).
#   place    : reward granted for each accepted (legal) placement
#   complete : bonus granted when the board is fully solved
#   invalid  : reward (penalty) on an illegal placement, which also terminates
#
# Note on the invalid term: because an invalid placement HARD-TERMINATES the
# episode, the agent already pays an implicit cost (it forfeits all future
# reward). An *additional* explicit penalty makes placing the next piece a
# negative-expected-value gamble whenever the agent's success probability is
# below ~0.5, which we observed drives a risk-averse local optimum (the agent
# safely stops after ~5 pieces and never completes). We therefore set invalid=0
# and let termination be the only cost. See the report's failure-case analysis.
REWARD_PRESETS = {
    # Dense shaping: per-placement progress reward + a large completion bonus.
    "dense": {"place": 1.0, "complete": 10.0, "invalid": 0.0},
    # Sparse: signal only on full completion; no shaping. Harder credit problem.
    "sparse": {"place": 0.0, "complete": 1.0, "invalid": 0.0},
}


def decode_action(action: int):
    """Map a flat action id in [0, 320) to (orientation, x, y).

    Flattening convention: action = orientation * (W*H) + x * H + y.
    """
    orient = action // (W * H)
    rem = action % (W * H)
    x = rem // H
    y = rem % H
    return orient, x, y


def encode_action(orient: int, x: int, y: int) -> int:
    """Inverse of :func:`decode_action`."""
    return orient * (W * H) + x * H + y


class BrainBlockEnv(gym.Env):
    """Finite-horizon packing MDP over a shuffled 10-piece queue."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, reward_mode: str = "dense", render_mode: str | None = None,
                 tiling_pool=None, prefill: int = 0, prefill_random: bool = False):
        super().__init__()
        if reward_mode not in REWARD_PRESETS:
            raise ValueError(
                f"reward_mode must be one of {list(REWARD_PRESETS)}, got {reward_mode!r}"
            )
        self.reward_mode = reward_mode
        self.reward_cfg = REWARD_PRESETS[reward_mode]
        self.render_mode = render_mode

        # Reverse-curriculum support (training only). When prefill > 0, reset()
        # pre-places pieces of a random known tiling, so the agent starts from a
        # board that is provably completable and only has to play the remaining
        # pieces. With prefill_random=True, each reset draws the level uniformly
        # from {0..prefill}; this keeps the real task (level 0) in the training
        # mix at all times while still supplying easy completions, and the
        # trainer anneals the ceiling `prefill` 9 -> 0 over training. Evaluation
        # uses prefill = 0 (always start from an empty board). See report Sec 3/5.
        self.tiling_pool = tiling_pool  # list of tilings, or None
        self.prefill = int(prefill)            # sampling upper bound (easy end)
        self.prefill_low = int(prefill)        # sampling lower bound
        self.prefill_random = bool(prefill_random)

        # Flat queue of piece-type indices: 2 of each type -> 10 pieces.
        self._base_queue = []
        for t, name in enumerate(PIECE_TYPES):
            self._base_queue.extend([t] * INVENTORY[name])
        self.n_pieces = len(self._base_queue)  # 10

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(0.0, 1.0, shape=(1, H, W), dtype=np.float32),
                "piece": spaces.Box(0.0, 1.0, shape=(NUM_TYPES,), dtype=np.float32),
                "inventory": spaces.Box(0.0, float(self.n_pieces),
                                        shape=(NUM_TYPES,), dtype=np.float32),
            }
        )

        # State (initialized in reset).
        self.board = None          # (H, W) int8, 1 = filled
        self.label_board = None    # (H, W) int8, piece-type per cell or -1
        self.queue = None          # list[int] of type indices
        self.ptr = 0               # index of current piece in queue
        self.placements = None     # list of (type, orient, x, y) accepted

    # ------------------------------------------------------------------ core
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((H, W), dtype=np.int8)
        self.label_board = np.full((H, W), -1, dtype=np.int8)
        self.queue = list(self._base_queue)
        self.np_random.shuffle(self.queue)
        self.ptr = 0
        self.placements = []

        prefill = self.prefill if options is None else options.get("prefill", self.prefill)
        if self.prefill_random and options is None:
            # Sample difficulty uniformly from {prefill_low..prefill}. The trainer
            # holds the upper bound at 9 (easy completions always available) and
            # anneals the lower bound 9 -> 0, progressively adding harder boards
            # (level 0 = the real, full-board task).
            lo = min(self.prefill_low, self.prefill)
            prefill = int(self.np_random.integers(lo, int(self.prefill) + 1))
        self.episode_prefill = int(prefill) if self.tiling_pool else 0
        if prefill and self.tiling_pool:
            self._apply_prefill(int(prefill))
        return self._obs(), self._info(valid=True, invalid=False)

    def _apply_prefill(self, prefill: int):
        """Pre-place the first `prefill` pieces from a random completable tiling.

        A tiling assigns each piece type its placements; we map them onto the
        current shuffled queue (by type) and auto-commit the first `prefill`.
        This guarantees a valid completion still exists from the start state.
        """
        prefill = max(0, min(prefill, self.n_pieces - 1))
        if prefill == 0:
            return
        tiling = self.tiling_pool[self.np_random.integers(len(self.tiling_pool))]
        by_type = {}
        for (t, o, x, y) in tiling:
            by_type.setdefault(t, []).append((o, x, y))
        # Stable copy so we can pop per type as we walk the queue.
        pool = {t: list(v) for t, v in by_type.items()}
        for i in range(prefill):
            t = self.queue[i]
            o, x, y = pool[t].pop()
            for (dy, dx) in get_cells(t, o):
                self.board[y + dy, x + dx] = 1
                self.label_board[y + dy, x + dx] = t
            self.placements.append((t, o, x, y))
        self.ptr = prefill

    def set_prefill(self, prefill: int):
        """Set the curriculum upper bound (easy end) for resets."""
        self.prefill = int(prefill)

    def set_prefill_low(self, low: int):
        """Set the curriculum lower bound (hard end) for sampling."""
        self.prefill_low = int(low)

    def step(self, action: int):
        orient, x, y = decode_action(int(action))
        cells = self._absolute_cells(orient, x, y)
        legal = self._is_legal(cells)
        cfg = self.reward_cfg

        if not legal:
            # Hard terminate on invalid placement.
            reward = cfg["invalid"]
            info = self._info(valid=False, invalid=True)
            return self._obs(), reward, True, False, info

        # Commit the placement.
        t = self.queue[self.ptr]
        for (ny, nx) in cells:
            self.board[ny, nx] = 1
            self.label_board[ny, nx] = t
        self.placements.append((t, orient, x, y))
        self.ptr += 1

        solved = self.ptr == self.n_pieces  # all pieces placed -> board full
        reward = cfg["place"]
        if solved:
            reward += cfg["complete"]

        terminated = solved
        info = self._info(valid=True, invalid=False)
        return self._obs(), reward, terminated, False, info

    # --------------------------------------------------------------- helpers
    def _absolute_cells(self, orient: int, x: int, y: int):
        """Absolute (row, col) cells for placing the current piece, or None."""
        t = self.queue[self.ptr]
        return [(y + dy, x + dx) for (dy, dx) in get_cells(t, orient)]

    def _is_legal(self, cells) -> bool:
        """All cells in-bounds and not overlapping filled cells."""
        for (ny, nx) in cells:
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                return False
            if self.board[ny, nx]:
                return False
        return True

    def legal_actions(self):
        """List of valid action ids for the current piece (eval/analysis only).

        Not used during stepping (the agent is unmasked); handy for measuring
        whether a failure was a forced dead-end vs. an avoidable mistake.
        """
        if self.ptr >= self.n_pieces:
            return []
        t = self.queue[self.ptr]
        valid = []
        for orient in range(NUM_ORIENTS):
            cells_off = get_cells(t, orient)
            for x in range(W):
                for yy in range(H):
                    cells = [(yy + dy, x + dx) for (dy, dx) in cells_off]
                    if self._is_legal(cells):
                        valid.append(encode_action(orient, x, yy))
        return valid

    def _inventory_counts(self):
        """Counts of remaining (not-yet-placed) pieces per type, incl. current."""
        counts = np.zeros(NUM_TYPES, dtype=np.float32)
        for t in self.queue[self.ptr:]:
            counts[t] += 1.0
        return counts

    def _obs(self):
        piece = np.zeros(NUM_TYPES, dtype=np.float32)
        if self.ptr < self.n_pieces:
            piece[self.queue[self.ptr]] = 1.0
        return {
            "board": self.board[None, :, :].astype(np.float32),
            "piece": piece,
            "inventory": self._inventory_counts(),
        }

    def _info(self, valid: bool, invalid: bool):
        filled = int(self.board.sum())
        solved = self.ptr == self.n_pieces
        return {
            "valid": valid,
            "invalid": invalid,
            "placed": self.ptr,
            "filled": filled,
            "covered_fraction": filled / N_CELLS,
            "is_success": bool(solved),
            "episode_prefill": getattr(self, "episode_prefill", 0),
        }

    # ---------------------------------------------------------------- render
    def render(self):
        if self.render_mode != "ansi":
            return None
        return self.to_ascii()

    def to_ascii(self) -> str:
        """ASCII board where each cell shows its piece type letter or '.'."""
        lines = []
        for row in self.label_board:
            cells = [("." if v < 0 else PIECE_TYPES[v]) for v in row]
            lines.append(" ".join(cells))
        head = (PIECE_TYPES[self.queue[self.ptr]]
                if self.ptr < self.n_pieces else "-")
        lines.append(f"placed={self.ptr}/{self.n_pieces} current={head}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test: random *legal* rollouts to confirm the board can be solved.
    env = BrainBlockEnv(reward_mode="dense")
    solves = 0
    episodes = 2000
    rng = np.random.default_rng(0)
    for ep in range(episodes):
        env.reset(seed=ep)
        total = 0.0
        while True:
            legal = env.legal_actions()
            if not legal:  # forced dead-end
                break
            a = int(rng.choice(legal))
            _, r, term, trunc, info = env.step(a)
            total += r
            if term or trunc:
                if info["is_success"]:
                    solves += 1
                break
    print(f"Random-legal policy solved {solves}/{episodes} episodes "
          f"({100*solves/episodes:.1f}%).")
    print("Example final board (greedy-legal):")
    print(env.to_ascii())
