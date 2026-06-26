# Graph Exploration & Surveillance Challenge

<p align="center">
  <img src="assets/graph-exploration-header.png" alt="3D graph exploration visualization with highlighted agent paths" width="800"/>
</p>

**Three UAVs** explore an unknown 3D graph. You control all three from one
centralized policy: each tick you choose where every UAV flies next. You only see
nodes within a few hops of each agent's position — no peeking through walls, only
along known connectivity.

**You win by finishing explore + surveil fastest** (lowest makespan flight
distance across both phases) with a team of three UAVs.

1. **Explore**: discover the graph until at least `explore_threshold` of nodes
   have been observed (a node is *observed* the moment it falls within sensor
   range, including nodes you fly past without stopping).
2. **Surveil**: reuse the map you built to plan an efficient re-observation
   sweep, until at least `surveil_threshold` of all nodes fall within sensor
   range *again*.

**Lowest makespan flight distance wins** — the max per-agent distance flown
(slowest UAV sets the score for the run).

Both thresholds are configured in `[eval]` in `exploration_challenge/params.toml`
(default **90%** explore / **90%** surveil), so part of the skill is deciding
what *not* to bother with. You only fly along edges you've discovered.

This challenge models 3D roadmap exploration in pure Python: nodes are
collision-free waypoints, edges are flyable straight lines, and edge cost is
distance.

**Documentation:** [`docs/RULES.md`](docs/RULES.md) (full rulebook),
[`docs/graph_format.md`](docs/graph_format.md) (graph JSON schema),
[`docs/results_format.md`](docs/results_format.md) (eval JSON export).

## Quick start

### With uv (recommended)

[uv](https://docs.astral.sh/uv/) sets up everything in one command: it reads
`pyproject.toml`, creates the environment, and installs dependencies. Nothing
to activate.

First install uv once (see the
[official guide](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
#   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# ...or: pipx install uv  /  pip install uv  /  brew install uv
```

Restart your shell (or `source $HOME/.local/bin/env`) so `uv` is on your PATH,
then:

```bash
# Run the random-walk baseline on the training graphs (env is built automatically):
uv run run_eval.py --graphs graphs/train

# With the Rerun 3D visualizer (--viz-reduced for a lighter static-map mode;
# install the viz extra first: uv sync --extra viz):
uv run run_eval.py --graphs graphs/train --viz

# Evaluate your own submission (3 UAVs by default; seed count from params.toml):
uv run run_eval.py --submission my_solution.py --graphs graphs/train

# Batch eval with a progress bar (suppress per-step output):
uv run run_eval.py --graphs graphs/train --quiet
```

### With pip

```bash
pip install -e .

python run_eval.py --graphs graphs/train            # 3-UAV baseline (default)
python run_eval.py --graphs graphs/train --viz      # 3D visualizer (--viz-reduced for static-map mode)
python run_eval.py --submission my_solution.py --graphs graphs/train
```

## Writing a submission

Start from [`exploration_challenge/policies/random_walk.py`](exploration_challenge/policies/random_walk.py):
the starter baseline. Copy it, rename (e.g. `my_solution.py` or `submission.py`),
and replace `reset` / `step` with your policy. Point the evaluator at your file
in any of these ways (highest precedence first):

1. **CLI**: `--submission my_solution.py` (use this for official hand-in runs)
2. **`params.toml`**: set `submission = "my_solution.py"` under `[eval]` (currently
   points at `exploration_challenge/policies/random_walk.py`)
3. **Auto-detect**: save your file as `submission.py` in the repo root
4. **Default**: `exploration_challenge/policies/random_walk.py` via `params.toml`

```python
class Explorer:
    def reset(self, starts, observations, seed=None): ...
    def step(self, observations, phase):
        # return a list of 3 int next-hop node ids (one per UAV)
        # each action: known neighbour of that UAV, or its current position to wait
        ...
```

Official eval uses **3 UAVs** (`n_agents = 3` in `params.toml`). Each tick you
receive every UAV's `Observation` and return **three** actions. UAVs move in
lockstep (one hop per tick), share perfect mutual information, start at distinct
nodes, and cannot occupy the same node or swap edges in one tick. Score = max
per-agent distance (makespan) — finish explore + surveil with the lowest score to
win.

`phase` is `"explore"` or `"surveil"`. Each action is an **int**: the next node
to fly to (a known neighbour, or current position to wait).

Each `Observation` exposes `position`, `position_xyz`, `nodes` (`ObservedNode`
with `id`, `x`, `y`, `z`), `edges` (`ObservedEdge` with `u`, `v`, `cost`),
`visited`, and `neighbors(node_id)`. Visible edges only connect nodes in the
current view — you are not told which visible nodes have neighbours beyond the
sensor range; merge scans into your own map and infer frontiers yourself.

**For the full rules**: what you can see, how `k` / `max_turn_deg` sensing
works, coverage and phase transitions, scoring, and what counts as an invalid
move. See [`docs/RULES.md`](docs/RULES.md). See [`docs/graph_format.md`](docs/graph_format.md) and
the docstrings in `exploration_challenge/observation.py`.

## Submitting your solution

1. **Implement**: copy [`exploration_challenge/policies/random_walk.py`](exploration_challenge/policies/random_walk.py) into
   a single file (e.g. `<team>_solution.py`) and replace the random-walk logic.
2. **Verify locally**:
   ```bash
   uv run run_eval.py --submission <team>_solution.py --graphs graphs/train --quiet
   ```
3. **Hand in**: upload `<team>_solution.py` via the
   [submission Google Form](https://forms.gle/YOUR_FORM_LINK_HERE) before the
   deadline (include your team name and member names in the form).
4. **Constraints**: stdlib plus dependencies declared in
   [`pyproject.toml`](pyproject.toml). Submit a single policy file.
5. **Ranking**: lower `total_score` (sum of per-graph mean makespan distances)
   wins — fastest explore + surveil across the suite.

## How it works

```
graph JSON (graphs/train/) ──> Simulator (true graph)
                        │  depth-k restricted view
                        ▼
                   Observation ──> Explorer.step ──> next-hop actions
                        │                                │
                        └──────── validates one hop ─────┘
                        │
                   coverage + phase tracking ──> evaluator ──> score
                                                       └─> Rerun viz
```

Coverage, phase transitions, and scoring (makespan flight distance — lowest wins,
`inf` for incomplete runs) are defined in full in [`docs/RULES.md`](docs/RULES.md). The terminal
summary shows **mean score** and **complete** (completion rate) per graph. The
JSON export adds `stdev_score`. See [`docs/results_format.md`](docs/results_format.md).

## Configuration

Evaluation defaults live in `exploration_challenge/params.toml` (`[eval]` section).
Packaged defaults run the **official 3-UAV** challenge (`n_agents = 3`), plus
`seeds = 1`, `max_steps = 1000`,
`k = 4`, `max_turn_deg = 75`, and `drop_prob = 0.0`. Most can be overridden via
CLI flags on `run_eval.py`: `--seeds`, `--max-steps`, `--step-delay`, `--k`,
`--max-turn-deg`, `--drop-prob`, `--n-agents`, `--start`, `--submission`.
Phase thresholds (`explore_threshold`, `surveil_threshold`) and the default
graph list (`graphs`) are set in `params.toml`. By default, start nodes are
random and distinct per seed; set `start` in `params.toml` or pass `--start` to
fix agent 0. Policy resolution: `[eval].submission` in `params.toml`, then
`submission.py` in the repo root, then `random_walk.py`.

**Seeding contract:** the master `--seeds` value derives separate RNG streams for
start nodes, sensor drops, and policy randomness (`exploration_challenge/_internal/seeding.py`), so
the same seed gives every submission the same environment for fair comparison.

If `--graphs` is unset, `run_eval.py` uses the `[eval].graphs` list in
`params.toml` (currently `basic.json`, `double_room.json`, `obstacles.json`,
`sparse.json`, `warehouse.json`, and `large.json` under `graphs/train/`).
Pass `--graphs graphs/train` to evaluate every `*.json` in that folder.

Training graphs live under `graphs/train/` (see [`docs/graph_format.md`](docs/graph_format.md)).

Visualizer modes: `--viz` (full scene refresh each step),
`--viz-reduced` (static map + drone/metrics only).

Install the Rerun visualizer with `uv sync --extra viz` or
`pip install -e ".[viz]"`.

## Layout

```
run_eval.py              # CLI entry point
docs/                    # RULES.md, graph_format.md, results_format.md
graphs/train/            # training graphs
results/                 # eval output from --output
exploration_challenge/
  params.toml            # evaluation defaults ([eval])
  graph_io.py            # node-link JSON load/save + graph helpers
  observation.py         # depth-k restricted view
  simulator.py           # world state, movement, phases, coverage
  evaluator.py           # run episodes, scoring, live stats
  viz/                   # Rerun 3D view (core, mesh, styles, rerun_compat)
  policies/
    random_walk.py       # starter baseline + default eval policy
  _internal/             # config, progress, seeding, sensor
```
