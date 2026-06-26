# Rules & mechanics

The complete rulebook for the **3-UAV** explore + surveil challenge. This document
describes *what the simulator does*. Finding good policies is up to you. The
[`README.md`](../README.md) gives the high-level picture. This file is the detailed
reference.

## Rules

- You are provided a Python simulator and training graphs under `graphs/train/`.
- The challenge runs with **three UAVs** (`n_agents = 3` in
  `exploration_challenge/params.toml`). One centralized `Explorer` controls all
  three.
- Your UAVs start at **distinct random nodes** (one random placement per
  evaluation seed, or fix agent 0 with `--start`).
- At each decision point each UAV observes nodes within **`k` graph hops** of its
  position, along paths where every consecutive hop respects a **maximum turn
  angle** (`max_turn_deg`). This prevents vision from bending around corners.
- Each scan may **independently miss** candidate nodes with probability
  `drop_prob`. A missed node is not visible and blocks see-through to nodes
  beyond it; later scans draw fresh.
- A node counts as **observed** the moment it is **detected** in sensor range,
  including nodes you fly past without stopping.
- **Exploration** ends once the observed fraction reaches `explore_threshold`
  (from `[eval]` in `exploration_challenge/params.toml`, default **90%**). The
  team then switches to surveillance.
- At the phase transition the **surveillance counter resets**: observations made
  during exploration do not count toward surveillance coverage.
- **Surveillance** ends once the re-observed fraction reaches
  `surveil_threshold` (from `[eval]` in `exploration_challenge/params.toml`, default
  **90%**). No return to the start node is required.
- **Scoring:** **lowest makespan flight distance wins** — finish explore + surveil
  with the smallest max per-agent distance (slowest UAV sets the episode score).
  Incomplete runs (invalid moves, stalling, or hitting the step limit) score `inf`.
- At `reset()` you receive **start node ids** and an initial **observation**
  per UAV of nearby nodes, edges, and coordinates.
- On every step you also receive your **visited-node history**.
- You may only move along **edges you have already discovered**.
- Each step, return a **list of three actions** (one per UAV). Each action is an **int**
  node id: a **known neighbour** of that UAV's current position (one graph hop), or
  its current position to wait.
- **Automatic observations while moving**: each hop reveals nodes along the way as
  the UAV traverses the edge.
- **Invalid actions end the episode**: naming a non-neighbour, an unknown node,
  returning a path/list instead of a single int, returning the wrong number of
  actions, or raising from `step()` aborts the run.
- Sensor settings **`k`**, **`max_turn_deg`**, and **`drop_prob`** are configured
  in `[eval]` in `exploration_challenge/params.toml` or via `run_eval.py` CLI flags.
- **Graph guarantees**: graphs are connected.
- Edges are undirected and represent collision-free straight-line flight between
  3D waypoints. Edge cost is traversal distance (usually Euclidean).
- Results are averaged over **multiple seeds** (random starts) per graph.
- **Deliverable**: implement a Python `Explorer` class in a single submission
  file with `reset(starts, observations, seed)` and
  `step(observations, phase) -> list[action]` (three actions per tick). See
  [Submitting](#submitting) below.

## Three UAVs

- One centralized `Explorer` sees every UAV's `Observation` each tick and returns
  one action per UAV (perfect mutual information across agents).
- UAVs advance in **lockstep** (one graph hop per tick).
- Each UAV starts at a **distinct** node (reproducible via the `"start"` seed
  stream; `--start` fixes agent 0 only).
- **Collisions blocked:** two UAVs cannot end on the same node, and cannot swap
  edges (A→B while B→A) in the same tick. Lower agent id wins vertex conflicts;
  edge swaps block both movers for that tick. Blocked moves do not raise an error;
  the affected UAV simply does not move and stall counters advance.
- Coverage unions across UAVs; **score = max per-agent distance** (makespan).

For debugging, pass `--n-agents 1` (single UAV; total distance equals makespan).
The default challenge uses **3 UAVs**.

## Seeding contract

The master evaluation seed derives separate streams (`exploration_challenge/_internal/seeding.py`):

| Stream | Tag | Used for |
|--------|-----|----------|
| Start nodes | `"start"` | Distinct start selection (same for all submissions) |
| Sensor | `"sensor"` | `drop_prob` draws in the simulator |
| Policy | `"policy"` | Passed to `Explorer.reset` |

Same seed + `n_agents` ⇒ same environment; score differences reflect the policy.
With `drop_prob > 0`, drop outcomes at a node still depend on visit order — average
over multiple seeds for fair comparison.

The sections below expand on these rules with the exact simulator semantics.

## What you control

Each decision, your `Explorer.step(observations, phase)` returns **three**
actions (one per UAV). Each action is an **int** node id: the **next hop** only — a known
neighbour of that UAV's current position, or its current position to wait.

You may only traverse edges that appear in your accumulated map.

## What you can see

At each step you receive one `Observation` per UAV (three observations per tick
in the official challenge):

| Field | Meaning |
|-------|---------|
| `agent_id` | UAV index (`0`, `1`, …) |
| `position` | Current node id |
| `position_xyz` | Current `(x, y, z)` in metres |
| `nodes` | `ObservedNode` tuples within sensor range (`id`, `x`, `y`, `z`) |
| `edges` | `ObservedEdge` tuples among visible nodes (`u`, `v`, `cost`) |
| `visited` | Every node that UAV has physically been at |
| `neighbors(node_id)` | Visible neighbours of a node (from `edges`) |

Edges only connect **visible** nodes. You are not told which visible nodes have
neighbours outside the current view — inferring where to explore next is part of
the challenge. Merge each scan into your own persistent map.

Sensor range is configured at evaluation time in
`exploration_challenge/params.toml` or on `run_eval.py` CLI flags:

- **`k`**: max graph hops visible from your position (eval default in
  `exploration_challenge/params.toml`, currently `4`).
- **`max_turn_deg`**: line-of-sight turn limit per hop (eval default in
  `exploration_challenge/params.toml`, currently `75°`). Sharp corners block vision
  from bending around them. Set to `180` to disable (pure hop ball).
- **`drop_prob`**: per-scan probability each candidate node is missed (eval
  default in `exploration_challenge/params.toml`, currently `0.0` = perfect sensor).
  A missed node is not visible and expansion stops at it, so nodes beyond a miss
  are hidden until a later scan detects the intermediate node. Re-scans are
  independent.

Your persistent map grows as you move: merge nodes and edges from each
observation into whatever state you maintain.

## Two phases, two counters

### Exploration (`phase == "explore"`)

- A node counts toward coverage when it is **detected** in any depth-`k` scan
  (including nodes you fly past without stopping). With `drop_prob > 0`, a node
  may be missed on one scan and detected on a later re-scan.
- Phase ends when `observed_fraction >= explore_threshold` (fraction of **all**
  nodes in the graph, default **90%** from `[eval]`).

### Surveillance (`phase == "surveil"`)

- Starts immediately after exploration completes.
- The surveillance counter **resets to empty**: prior observations do not count.
- A node counts when it re-enters depth-`k` view **during this phase**.
- Phase ends when `surveil_fraction >= surveil_threshold` (fraction of **all**
  nodes, default **90%** from `[eval]`).
- **No return to the start node is required.**

The only carry-over from exploration is **map knowledge**, which makes routing
the second pass cheaper.

## Scoring and failure modes

- **Episode score** = makespan flight distance (max per-agent distance; explore +
  surveil legs combined via the slowest UAV). **Lower is better** — fastest
  complete explore + surveil wins.
- **Incomplete** episodes score `inf`: invalid moves, five consecutive ticks
  with no movement by any UAV, five consecutive ticks where a UAV is
  collision-blocked (multi-agent), or hitting `--max-steps` (default from
  `[eval].max_steps` in `params.toml`, currently `1000`).
- Results are averaged over **multiple seeds** (count from `params.toml` or
  `--seeds`, random distinct starts per seed, or fix agent 0 with `--start`).
- **Suite score:** **`total_score`** is the sum of per-graph mean scores. Lower
  wins — fastest explore + surveil across the suite.

Export detailed runs to JSON with `run_eval.py --output`; see
[`results_format.md`](results_format.md).

## Submitting

1. **Implement:** copy [`random_walk.py`](../exploration_challenge/policies/random_walk.py) into
   a single file (e.g. `<team>_solution.py`) with an `Explorer` class.
2. **Verify locally**:
   ```bash
   uv run run_eval.py --submission <team>_solution.py --graphs graphs/train --quiet
   ```
3. **Hand in**: upload `<team>_solution.py` when the submission link is
   published (include your team name and member names).
4. **Constraints**: Python stdlib plus dependencies declared in
   [`pyproject.toml`](../pyproject.toml). Submit a single policy file.

Phase thresholds (`explore_threshold`, `surveil_threshold`) and the default
graph list are set in `[eval]` in `params.toml`.

## Practical API reminders

```python
for obs in observations:
    for n in obs.nodes:
        ...  # n.id, n.x, n.y, n.z
    for e in obs.edges:
        ...  # e.u, e.v, e.cost
    ...  # merge each UAV's latest view into your own map state
nxt = obs.neighbors(obs.position)[0]  # example: pick a visible neighbour
obs.neighbors(node_id)              # visible neighbours of a node
```

See also [`graph_format.md`](graph_format.md),
[`random_walk.py`](../exploration_challenge/policies/random_walk.py), and the docstrings in
`exploration_challenge/observation.py`.
