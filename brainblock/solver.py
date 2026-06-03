"""Exact-cover tiling utilities for the BrainBlock board.

This is NOT the learning agent. It is a deterministic backtracking solver used
to (a) verify that the environment is solvable, (b) check that placements a
learned agent produces form a valid exact cover, and (c) optionally provide a
scripted oracle baseline. The RL agent in this project must discover solutions
on its own; this file only supports testing and verification.

Key fact exploited here: because the 10 pieces of any complete tiling are
mutually disjoint, they can be placed in *any* queue order without ever
overlapping or going out of bounds. So a fixed shuffled queue is solvable iff a
free-order tiling exists -- and we can realize a single tiling for every seed.
"""

from __future__ import annotations

import numpy as np

from .env import H, W, encode_action
from .pieces import get_cells, NUM_ORIENTS, NUM_TYPES


def find_tiling(inventory=None, rng=None, max_nodes=20000):
    """Return a list of (type, orient, x, y) placements tiling the board, or None.

    Pruned by always covering the topmost-leftmost empty cell with some
    available piece -- a correct exact-cover prune for *free* piece order.

    If ``rng`` (a numpy Generator) is given, the order in which piece types and
    orientations are tried is shuffled, so repeated calls return *diverse*
    tilings (used to build a varied curriculum / solution pool). ``max_nodes``
    bounds the search so an unlucky random order can't backtrack for a long
    time; on budget exhaustion it returns None and the caller simply retries.
    """
    if inventory is None:
        inventory = [2] * NUM_TYPES
    board = np.zeros((H, W), dtype=np.int8)
    avail = list(inventory)
    placed = []
    nodes = [0]

    def first_empty():
        for r in range(H):
            for c in range(W):
                if not board[r, c]:
                    return r, c
        return None

    def rec():
        nodes[0] += 1
        if nodes[0] > max_nodes:
            raise _BudgetExceeded
        target = first_empty()
        if target is None:
            return True
        tr, tc = target
        type_order = list(range(NUM_TYPES))
        orient_order = list(range(NUM_ORIENTS))
        if rng is not None:
            rng.shuffle(type_order)
            rng.shuffle(orient_order)
        for t in type_order:
            if avail[t] == 0:
                continue
            for o in orient_order:
                off = get_cells(t, o)
                for (dy, dx) in off:                 # land some cell on target
                    y, x = tr - dy, tc - dx
                    if x < 0 or y < 0:
                        continue
                    cells = [(y + ay, x + ax) for ay, ax in off]
                    if (tr, tc) not in cells:
                        continue
                    if all(0 <= r < H and 0 <= c < W and not board[r, c]
                           for r, c in cells):
                        for r, c in cells:
                            board[r, c] = 1
                        avail[t] -= 1
                        placed.append((t, o, x, y))
                        if rec():
                            return True
                        placed.pop()
                        avail[t] += 1
                        for r, c in cells:
                            board[r, c] = 0
        return False

    try:
        ok = rec()
    except _BudgetExceeded:
        return None
    return list(placed) if ok else None


class _BudgetExceeded(Exception):
    """Internal: raised when the backtracking node budget is exceeded."""


def realize_in_queue_order(tiling, queue):
    """Order a tiling's placements to match a fixed queue, return action ids."""
    by_type = {}
    for p in tiling:
        by_type.setdefault(p[0], []).append(p)
    actions = []
    for t in queue:
        p = by_type[t].pop()
        actions.append(encode_action(p[1], p[2], p[3]))
    return actions
