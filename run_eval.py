#!/usr/bin/env python3
"""Evaluate an Explorer submission over a set of challenge graphs.

Official challenge: **3 UAVs**, two phases (explore then surveil). Win by lowest
makespan flight distance (finish both phases fastest).

Examples
--------
    # Evaluate the starter template on the training graphs (3 UAVs by default):
    python run_eval.py --graphs graphs/train

    # Evaluate your own submission (or set submission in params.toml / submission.py):
    python run_eval.py --submission my_solution.py --graphs graphs/train --viz

    # Local dev: copy exploration_challenge/policies/random_walk.py to submission.py
    python run_eval.py --graphs graphs/train

    # Static grey graph; drone motion + metrics only (fastest on large graphs):
    python run_eval.py --graphs graphs/train --viz --viz-reduced

    # Single-UAV debug run:
    python run_eval.py --graphs graphs/train --n-agents 1

    # Export JSON results (optional; see docs/results_format.md):
    python run_eval.py --submission my_solution.py --graphs graphs/train --quiet \
        --output results/eval.json

    # Evaluate specific graphs (multiple paths):
    python run_eval.py --graphs graphs/train/basic.json graphs/train/obstacles.json

    # Evaluate every JSON in a directory:
    python run_eval.py --graphs graphs/train
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

# Make the bundled `exploration_challenge` package importable regardless of CWD.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from exploration_challenge._internal.config import eval_params  # noqa: E402
from exploration_challenge.evaluator import (  # noqa: E402
    MAX_AGENTS,
    export_suite,
    print_summary,
    run_suite,
    write_suite_json,
)
from exploration_challenge.graph_io import load_graph  # noqa: E402

DEFAULT_POLICY = "exploration_challenge.policies.random_walk"
SUBMISSION_FILE = os.path.join(HERE, "submission.py")


def resolve_submission(submission: str | None) -> tuple[str | None, str]:
    """Pick policy source: explicit path/module, then submission.py, then baseline."""
    if submission is not None:
        return submission, submission
    if os.path.exists(SUBMISSION_FILE):
        return SUBMISSION_FILE, "submission.py"
    return None, "random walk baseline"


def load_explorer_class(submission: str | None) -> tuple[type, str, str | None]:
    """Return (Explorer class, source label, export path or None for default baseline)."""
    path, source = resolve_submission(submission)
    if path is None:
        module = importlib.import_module(DEFAULT_POLICY)
        return module.Explorer, source, None

    if path.endswith(".py") and os.path.exists(path):
        spec = importlib.util.spec_from_file_location("submission", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(path)

    if not hasattr(module, "Explorer"):
        raise SystemExit(f"submission '{path}' has no `Explorer` class")
    return module.Explorer, source, path


def preferred_graph_basenames() -> list[str]:
    """Basenames from params.toml [eval].graphs, preserving list order."""
    names = eval_params().get("graphs")
    if not names:
        return []
    return [os.path.basename(name) for name in names]


def order_graph_basenames(basenames: list[str], preferred: list[str]) -> list[str]:
    """Order graph filenames: preferred list first, then any extras alphabetically."""
    rank = {name: i for i, name in enumerate(preferred)}

    def sort_key(name: str) -> tuple[int, int | str]:
        if name in rank:
            return (0, rank[name])
        return (1, name)

    return sorted(basenames, key=sort_key)


def resolve_default_graph_target(name: str) -> str:
    """Map a params.toml graph entry to a concrete filesystem path."""
    if os.path.isabs(name):
        return name
    graphs_root = os.path.join(HERE, "graphs")
    normalized = name.replace("/", os.sep)
    if normalized.startswith("graphs" + os.sep):
        return os.path.join(HERE, normalized)
    if normalized.startswith(("train" + os.sep, "test" + os.sep)):
        return os.path.join(graphs_root, normalized)
    for subdir in ("train", "test"):
        candidate = os.path.join(graphs_root, subdir, name)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(graphs_root, "train", name)


def default_graph_targets() -> list[str]:
    """Resolve default graph paths from params.toml [eval].graphs or graphs/train/."""
    train_dir = os.path.join(HERE, "graphs", "train")
    names = eval_params().get("graphs")
    if names:
        return [resolve_default_graph_target(name) for name in names]
    return [train_dir]


def collect_graph_paths(targets: str | list[str]) -> list[str]:
    if isinstance(targets, str):
        targets = [targets]
    preferred = preferred_graph_basenames()
    paths: list[str] = []
    for target in targets:
        if os.path.isdir(target):
            json_names = [f for f in os.listdir(target) if f.endswith(".json")]
            paths.extend(
                os.path.join(target, f)
                for f in order_graph_basenames(json_names, preferred)
            )
        elif os.path.isfile(target):
            paths.append(target)
        else:
            raise SystemExit(f"no graphs found at '{target}'")
    if not paths:
        raise SystemExit("no graph JSON files found")
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def main() -> None:
    cfg = eval_params()  # defaults from params.toml [eval]; CLI flags override.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--submission",
        default=cfg.get("submission"),
        help="Path to a .py file (or dotted module) with an `Explorer`. "
             "Defaults to [eval].submission in params.toml, then submission.py, "
             "then random_walk.py.",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=None,
        help="Graph JSON file(s) or director(ies). "
             "Defaults to [eval].graphs in params.toml, or all of graphs/train/. "
             "Bare filenames in params.toml resolve under graphs/train/ then graphs/test/.",
    )
    parser.add_argument("--seeds", type=int, default=cfg["seeds"],
                        help="Number of seeds per graph (results averaged).")
    parser.add_argument("--k", type=int, default=cfg["k"],
                        help="Observation depth: graph hops visible from the drone.")
    parser.add_argument(
        "--max-turn-deg",
        type=float,
        default=cfg["max_turn_deg"],
        help="Max per-hop turn angle (degrees) for line-of-sight vision; 180 disables.",
    )
    parser.add_argument(
        "--drop-prob",
        type=float,
        default=cfg.get("drop_prob", 0.0),
        help="Per-scan probability each candidate node is missed by the sensor (0 = perfect).",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=cfg.get("n_agents", 1),
        help="Number of UAVs (official challenge: 3; 1–3 for local testing).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=cfg.get("start"),
        help="Fixed start node id for agent 0; other agents get random distinct nodes per seed.",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Launch the Rerun 3D visualizer.",
    )
    parser.add_argument(
        "--viz-reduced",
        action="store_true",
        help="Static grey graph; update drone motion and metrics only.",
    )
    parser.add_argument("--max-steps", type=int, default=cfg["max_steps"])
    parser.add_argument("--step-delay", type=float, default=cfg["step_delay"],
                        help="Seconds to pause between drone moves (slows live playback).")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step stats.")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write JSON results to this path (see docs/results_format.md).",
    )
    args = parser.parse_args()
    if not 1 <= args.n_agents <= MAX_AGENTS:
        parser.error(f"--n-agents must be between 1 and {MAX_AGENTS}")
    if args.viz_reduced and not args.viz:
        print("warning: --viz-reduced has no effect without --viz")

    explorer_cls, policy_source, export_submission = load_explorer_class(args.submission)
    if not args.quiet:
        print(f"policy: {policy_source}")
    graph_targets = args.graphs if args.graphs is not None else default_graph_targets()
    paths = collect_graph_paths(graph_targets)
    worlds = [load_graph(p) for p in paths]

    viz = None
    if args.viz:
        from exploration_challenge.viz import Visualizer
        viz = Visualizer(reduced=args.viz_reduced)
        if not viz.enabled:
            print("warning: rerun-sdk not available; running without visualization")
            viz = None

    seeds = list(range(args.seeds))
    n_runs = len(worlds) * len(seeds)
    suite = run_suite(
        worlds,
        make_explorer=lambda: explorer_cls(),
        seeds=seeds,
        k=args.k,
        start=args.start,
        n_agents=args.n_agents,
        max_turn_deg=args.max_turn_deg,
        drop_prob=args.drop_prob,
        viz=viz,
        live=not args.quiet,
        max_steps=args.max_steps,
        step_delay=args.step_delay,
        show_progress=args.quiet and n_runs > 1,
    )
    if args.output:
        payload = export_suite(
            suite,
            submission=export_submission,
            eval_settings={
                "k": args.k,
                "max_turn_deg": args.max_turn_deg,
                "drop_prob": args.drop_prob,
                "n_agents": args.n_agents,
                "seeds": seeds,
                "start": args.start,
                "max_steps": args.max_steps,
            },
            graphs=" ".join(graph_targets),
        )
        write_suite_json(args.output, payload)
    print_summary(suite)


if __name__ == "__main__":
    main()
