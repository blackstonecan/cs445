"""Deterministic tetromino geometry for the BrainBlock environment.

The mapping ``(piece_type, orientation) -> set of occupied cells`` is a fixed
function with only 5 x 8 = 40 input combinations. It is therefore *computed
analytically* here (not learned): this table is environment dynamics, and it is
used to (a) check placement legality, (b) write pieces onto the board, and
(c) render solutions.

Coordinate convention (kept consistent across env, action decoding, eval):
    - Cartesian/row-major hybrid: a cell is ``(row, col) = (y, x)``.
    - Board has shape ``(H, W) = (5, 8)`` -> rows y in [0, H), cols x in [0, W).
    - A placement has an anchor ``(y, x)`` and a piece contributes cell offsets
      ``(dy, dx)``; the absolute occupied cells are ``(y + dy, x + dx)``.

Orientations: index ``o in {0..7}`` encodes ``mirror = o // 4`` (0 or 1) and
``rotation = o % 4`` (number of 90 deg CCW rotations). Pieces with symmetry
produce duplicate orientations across some indices; that is intentional and
allowed by the assignment ("redundant indices may be mapped to equivalent
transforms"). Keeping a uniform 8-slot scheme makes the action space regular.
"""

from __future__ import annotations

import numpy as np

# Piece type names and their canonical index.
PIECE_TYPES = ["I", "O", "L", "Z", "T"]
TYPE_TO_IDX = {name: i for i, name in enumerate(PIECE_TYPES)}
NUM_TYPES = len(PIECE_TYPES)
NUM_ORIENTS = 8
CELLS_PER_PIECE = 4

# Inventory: 2 of each type -> 10 pieces, 40 cells (exactly tiles an 8x5 board).
INVENTORY = {name: 2 for name in PIECE_TYPES}

# Base reference orientation for each type, as (row, col) offsets.
_BASE_SHAPES = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],   # horizontal bar
    "O": [(0, 0), (0, 1), (1, 0), (1, 1)],   # 2x2 square
    "L": [(0, 0), (1, 0), (2, 0), (2, 1)],   # L
    "Z": [(0, 0), (0, 1), (1, 1), (1, 2)],   # Z
    "T": [(0, 0), (0, 1), (0, 2), (1, 1)],   # T
}


def _normalize(cells):
    """Shift cells so the minimum row and minimum col are both 0."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _rotate90(cells):
    """Rotate 90 degrees CCW: (r, c) -> (-c, r), then normalize."""
    return _normalize([(-c, r) for r, c in cells])


def _mirror(cells):
    """Reflect across the vertical axis: (r, c) -> (r, -c), then normalize."""
    return _normalize([(r, -c) for r, c in cells])


def _build_table():
    """Return PIECE_CELLS[type_idx][orient_idx] = tuple of (dy, dx) offsets."""
    table = []
    for name in PIECE_TYPES:
        base = _normalize(_BASE_SHAPES[name])
        variants = []
        for m in range(2):                       # mirror flag
            cells = _mirror(base) if m else list(base)
            for _ in range(4):                   # 4 rotations
                variants.append(tuple(_normalize(cells)))
                cells = _rotate90(cells)
        assert len(variants) == NUM_ORIENTS
        table.append(variants)
    return table


# PIECE_CELLS[type_idx][orient_idx] -> tuple of 4 (dy, dx) offsets, normalized.
PIECE_CELLS = _build_table()


def get_cells(type_idx: int, orient_idx: int):
    """Cell offsets (dy, dx) for a given piece type and orientation index."""
    return PIECE_CELLS[type_idx][orient_idx]


def num_unique_orientations(type_idx: int) -> int:
    """Count distinct shapes among the 8 orientation slots for a piece type."""
    return len(set(PIECE_CELLS[type_idx]))


def to_mask(type_idx: int, orient_idx: int) -> np.ndarray:
    """Render a piece orientation as a 4x4 binary mask (for viz/inspection)."""
    mask = np.zeros((4, 4), dtype=np.int8)
    for dy, dx in get_cells(type_idx, orient_idx):
        mask[dy, dx] = 1
    return mask


if __name__ == "__main__":
    # Quick sanity dump: unique-orientation counts and a render of each shape.
    # L is chiral -> mirror (J) adds 4 distinct shapes = 8; Z's mirror (S) adds 2 = 4.
    expected_unique = {"I": 2, "O": 1, "L": 8, "Z": 4, "T": 4}
    for ti, name in enumerate(PIECE_TYPES):
        u = num_unique_orientations(ti)
        flag = "ok" if u == expected_unique[name] else "MISMATCH"
        print(f"{name}: {u} unique orientations [{flag}]")
        for oi in range(NUM_ORIENTS):
            cells = get_cells(ti, oi)
            assert len(cells) == CELLS_PER_PIECE
            assert len(set(cells)) == CELLS_PER_PIECE, "duplicate cell!"
            h = 1 + max(r for r, _ in cells)
            w = 1 + max(c for _, c in cells)
            assert h <= 4 and w <= 4, "exceeds 4x4 bounding box!"
    print("All pieces fit in 4x4, 4 distinct cells each. OK.")
