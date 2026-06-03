# CS 445 & 545 Spring 26: Deep Reinforcement Learning
## Project Assignment: BrainBlock Packing Environment

## 1 Game Specification

This assignment is based on a tangram-style packing puzzle. You are given an 8 × 5 board and a fixed inventory of tetromino pieces. The objective is to place all pieces legally so the board is fully covered.

Each episode is a finite-horizon sequential decision process. At every step, the environment provides one current piece from a shuffled queue. The agent must choose an orientation and an anchor location for that piece. The placement is accepted only if all occupied cells are inside the board and do not overlap existing filled cells. If accepted, the piece is written to the board and removed from the queue.

Coordinate conventions for board indexing may be selected by your implementation (for example, row-major or Cartesian), but must be consistent throughout environment logic, action decoding, and evaluation.

### 1.1 Board

Use a board of size

$$W \times H = 8 \times 5,$$

so the board has 40 cells in total.

> *Figure 1: BrainBlock board geometry (8 × 5 unit-cell grid).*

### 1.2 Piece set and inventory

Use the following tetromino types:

$$P = \{I, O, L, Z, T\}.$$

Each tetromino type occupies exactly 4 unit cells. The inventory is:

$$2 \times I, \quad 2 \times O, \quad 2 \times L, \quad 2 \times Z, \quad 2 \times T.$$

Hence there are 10 pieces total, covering exactly

$$10 \cdot 4 = 40$$

cells when fully solved.

> *Figure 2: Tetromino types used in the assignment (one reference orientation each): I, O, L, Z, T.*

**There are more than 500 solutions to this problem.**

### 1.3 Orientations and placement legality

A placement is legal iff:

1. all cells of the oriented piece lie within board bounds, and
2. none of those cells overlap already-filled cells.

You may define orientation handling internally as you wish, but your action interface must match Section 3.2.

### 1.4 Episode dynamics

At episode start, construct a sequence (queue) containing the 10 pieces above, then randomize its order. At each step, only the current queue head piece may be placed.

An episode ends when one of the following occurs:

- all pieces are placed successfully (completion),
- an invalid placement is executed (if your environment uses hard termination for invalid actions, this is up to you),
- or your environment reaches a terminal dead-end condition by your own design.

## 2 Project Task and Learning Goals

In this project, you will formulate and solve the BrainBlock puzzle above as a sequential decision-making problem using deep reinforcement learning (DRL). Your RL agent should be able to find multiple solutions (rather than memorizing one). Factor this into your design (of course you can start by first discovering one solution!) and demonstrate the example solutions found by your agent (at least 5 solutions).

**Important:** No starter code is provided. You are expected to implement the environment as a gym environment, training evaluation pipeline using PyTorch.

**Learning goals.** This project is given to you with the aim of equipping you with the following skills:

- Formalize a custom DRL problem as an MDP.
- Design and compare reward functions.
- Train and evaluate DRL agents under reproducible settings.
- Analyze failure modes and justify modeling choices.

## 3 MDP Requirements

### 3.1 State space requirements

Your state/observation must include all of the following:

1. current board state,
2. current piece to be placed,
3. information about remaining pieces.

**Constraint:** We do not prescribe a specific encoding, mathematical representation trick, feature engineering approach, or architecture-side representation. You must design and justify these choices.

### 3.2 Action space

You must use a discrete action corresponding to orientation and anchor position:

$$A = \{0, 1, \dots, 7\} \times \{0, 1, \dots, W-1\} \times \{0, 1, \dots, H-1\}.$$

For this assignment, with $W = 8$ and $H = 5$:

$$|A| = 8 \cdot 8 \cdot 5 = 320.$$

Interpretation:

- first component: orientation index,
- second component: x anchor,
- third component: y anchor.

Orientation indices 0 through 7 represent the 8 transformed variants (rotations/reflections) supported by your implementation. If a piece has fewer unique geometric orientations due to symmetry, redundant indices may be mapped to equivalent transforms.

You may internally flatten/unflatten actions if needed, but your semantics must be equivalent to the Cartesian-product action space above.

### 3.3 Reward function

No reward function is provided. You must:

1. propose at least two distinct reward functions,
2. train and evaluate agents under each,
3. compare outcomes quantitatively and qualitatively,
4. justify your final choice.

## 4 What to Implement

1. Environment implementation (must be Gymnasium-compatible).
2. At least one DRL agent from scratch using PyTorch.
3. Training script with configurable hyperparameters.
4. Evaluation script with deterministic rollouts and summary metrics.
5. Report (see Section 6).

## 5 Evaluation Protocol

Run at least 5 random seeds for each major experiment. Report:

- success rate (fraction of episodes solved),
- mean and standard deviation of episodic return,
- mean episode length,
- invalid-action rate (if applicable),
- at least one learning curve figure.

Include one short qualitative rollout analysis with visualizations or step traces.

## 6 Deliverables

1. Code package with clear run instructions.
2. Report PDF

Your report must include (but not limited to) the following:

- MDP definition and design choices for state representation.
- Full reward-function definitions (at least two), motivation, and analysis.
- Algorithm details and hyperparameters.
- Experimental setup and evaluation metrics.
- Results, and failure-case discussions.
- Sample solutions found by your system (at least 5).
- Figures with captions and sufficient explanations in the report main text showing progress for those solutions during training (at minimum):
  - total reward vs. episode #,
  - total covered area vs. episode #,
  - episode length over time,
  - invalid-action rate over time (if applicable)
- Any other useful information about your solution.
- Final conclusions and future improvements.

## 7 Presentation

You must conduct a 10 minute presentation, explaining your design choices, your results and your evaluations. You must also show a live demo where your pretrained RL agent solves the task visually on the spot.

## 8 Deadlines

- **Project Group Formation — 19th of April, 23:59**
  - Project groups must ideally consist of 3 people. Cross code groups (members from both 445 and 545) are **NOT** allowed.
  - In exceptional cases where you are required to form a group larger or smaller than 3, contact your TA.
  - You must enter your project group info in the shared Google Sheet provided in the assignment.
- **Project Code + Report Deadline — 4th of June 2026, 23:59**
- **Presentations — 5th of June, Time TBA**

## 9 Academic Integrity

You may discuss high-level ideas with peers, but all submitted code, experiments, and writing must be your own work unless explicitly cited. Any external code or tools must be clearly acknowledged. If you are using Generative AI, you must explicitly state the usage conditions.
