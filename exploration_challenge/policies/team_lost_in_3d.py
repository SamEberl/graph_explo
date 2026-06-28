"""SW2 Surveil Fusion -- adaptive-spread explore + LOS-ball surveillance tour.

* **Explore** = Frontier sweep, plus adaptive spread, EMA for discovery speed.

* **Surveil** = re-covering with a coverage-tour problem, k-means partition 
  per UAV, nearest-neighbour ordering. 
"""

from __future__ import annotations

import heapq
import math
import random
from collections import deque

from exploration_challenge.observation import Observation


class Explorer:
    # --- lifecycle -------------------------------------------------------------
    def reset(
        self,
        starts: list[int],
        observations: list[Observation],
        seed: int | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.n_agents = len(observations)

        # Persistent known map.
        self.coords: dict[int, tuple[float, float, float]] = {}
        self.adj: dict[int, dict[int, float]] = {}
        self.observed: set[int] = set()
        self.visited: set[int] = set()

        # Per-agent routing state.
        self.target: list[int | None] = [None] * self.n_agents
        self.path: list[list[int]] = [[] for _ in range(self.n_agents)]
        self.is_probe: list[bool] = [False] * self.n_agents
        self.stall: list[int] = [0] * self.n_agents

        # Phase / coverage bookkeeping.
        self.phase = "explore"
        self.seen_phase: set[int] = set()        # nodes observed during surveil
        self._settled_cache: set[int] | None = None
        self._settled_key = -1

        # Surveil-tour state.
        self.tour: list[deque[int]] = [deque() for _ in range(self.n_agents)]
        self._tour_built = False

        # Adaptive-spread bookkeeping (explore).
        self._discovery_ema: float | None = None
        self._prev_metric = 0
        self.spread_mult = 1
        self._explore_stall = 0                   # consecutive ticks w/ no new nodes

        # Tunables.
        self.k = 4                              
        self.max_turn_deg = 75.0                
        self.settle_radius = max(1, self.k - 1)  # revisit-avoidance radius
        self._scan_radius = 0.0                  # euclidean spread radius (lazy)
        self.surveil_target = 0.50               # fraction the viewpoint cover aims for
        self._two_opt_cap = 250                  # max waypoints/tour to 2-opt (cost guard)
        self.ema_alpha = 0.1                     # discovery-rate smoothing
        self.slowdown_ratio = 0.4                # rate < ratio*avg => stalling
        self.spread_boost = 2.5                  # separation widening when stalled
        self.min_signal = 2.0                    # ignore noise when avg is tiny
        self.steps = 0
        self.max_steps = 1000                    
        self.spread_off_frac = 0.5               
        self.stall_limit = 15

        for obs in observations:
            self._ingest(obs)
        self._prev_metric = len(self.observed)

    # --- main loop -------------------------------------------------------------
    def step(self, observations: list[Observation], phase: str) -> list[int]:
        self.steps += 1
        if phase != self.phase:
            self._on_phase_change(phase)

        for obs in observations:
            self._ingest(obs)
            if phase == "surveil":
                self.seen_phase.update(n.id for n in obs.nodes)

        positions = [obs.position for obs in observations]
        self._update_scan_radius()
        self._update_discovery()

        # Tour
        if phase == "surveil" and not self._tour_built:
            self._build_surveil_tour(positions)
            self._tour_built = True

        covered = self._covered_set()
        claimed: list[tuple[float, float, float]] = []
        desired: list[int] = []
        for i, pos in enumerate(positions):
            nxt = self._next_hop(i, pos, covered, claimed)
            desired.append(nxt)

        return self._resolve(positions, desired, covered)

    # --- map maintenance -------------------------------------------------------
    def _ingest(self, obs: Observation) -> None:
        self.visited.update(obs.visited)
        self.visited.add(obs.position)
        for n in obs.nodes:
            self.coords[n.id] = (n.x, n.y, n.z)
            self.observed.add(n.id)
            self.adj.setdefault(n.id, {})
        px, py, pz = obs.position_xyz
        self.coords[obs.position] = (px, py, pz)
        self.observed.add(obs.position)
        self.adj.setdefault(obs.position, {})
        for e in obs.edges:
            self.adj.setdefault(e.u, {})[e.v] = e.cost
            self.adj.setdefault(e.v, {})[e.u] = e.cost

    def _on_phase_change(self, phase: str) -> None:
        self.phase = phase
        # re-plan everyone for a fresh sweep.
        self.seen_phase = set()
        self.target = [None] * self.n_agents
        self.path = [[] for _ in range(self.n_agents)]
        self.is_probe = [False] * self.n_agents
        self.stall = [0] * self.n_agents
        self._settled_cache = None
        self.tour = [deque() for _ in range(self.n_agents)]
        self._tour_built = False
        
        self._discovery_ema = None
        self._prev_metric = len(self.seen_phase) if phase == "surveil" else len(self.observed)
        self.spread_mult = 1.0
        self._explore_stall = 0

    # --- coverage --------------------------------------------------------------
    def _covered_set(self) -> set[int]:
        """Nodes that no longer need attention in the current phase."""
        if self.phase == "surveil":
            return self.seen_phase
        
        key = len(self.visited)
        if self._settled_cache is not None and key == self._settled_key:
            return self._settled_cache
        settled: set[int] = set()
        dq: deque[tuple[int, int]] = deque()
        for v in self.visited:
            if v in self.adj:
                settled.add(v)
                dq.append((v, 0))
        while dq:
            u, d = dq.popleft()
            if d >= self.settle_radius:
                continue
            for w in self.adj.get(u, ()):  # known neighbours only
                if w not in settled:
                    settled.add(w)
                    dq.append((w, d + 1))
        self._settled_cache = settled
        self._settled_key = key
        return settled

    def _update_scan_radius(self) -> None:
        if self._scan_radius > 0.0:
            return
        
        lens = [c for nbrs in self.adj.values() for c in nbrs.values()]
        if lens:
            lens.sort()
            med = lens[len(lens) // 2]
            self._scan_radius = max(med, 1e-6) * self.k

    def _update_discovery(self) -> None:
        """Track newly-covered nodes/tick and widen team spread when it stalls."""
        metric = len(self.seen_phase) if self.phase == "surveil" else len(self.observed)
        found = metric - self._prev_metric
        self._prev_metric = metric

        # Stall detect EMA:
        if self.phase != "surveil":
            self._explore_stall = 0 if found > 0 else self._explore_stall + 1

        if self._discovery_ema is None:
            self._discovery_ema = float(found)
            return
        avg = self._discovery_ema
        self._discovery_ema = (1.0 - self.ema_alpha) * avg + self.ema_alpha * found

        if self.steps >= self.spread_off_frac * self.max_steps:
            # step-efficient greedy sweep.
            self.spread_mult = 1.0
        elif avg >= self.min_signal and found < self.slowdown_ratio * avg:
            self.spread_mult = self.spread_boost
        elif found >= avg:
            self.spread_mult = 1.0

    # --- sensor model ----------------------------------------------------------
    @staticmethod
    def _unit(
        a: tuple[float, float, float] | None,
        b: tuple[float, float, float] | None,
    ) -> tuple[float, float, float] | None:
        if a is None or b is None:
            return None
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length == 0.0:
            return None
        return (dx / length, dy / length, dz / length)

    @staticmethod
    def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _los_ball(self, position: int) -> set[int]:
        """Turn-constrained line-of-sight ball, replicating the simulator's
        sensor on our own reconstructed map. A node is visible only along a path
        whose every edge-to-edge turn is <= ``max_turn_deg`` (first hop free)."""
        coords = self.coords
        if position not in coords:
            return {position}
        visible = {position}
        if self.k <= 0:
            return visible

        cos_thresh = math.cos(math.radians(self.max_turn_deg))
        no_gate = self.max_turn_deg >= 180.0
        p = coords[position]
        queue: deque[tuple[int, tuple[float, float, float], int]] = deque()
        best_depth: dict[tuple[int, int], int] = {}

        for nb in self.adj.get(position, ()):  # first hop is always allowed
            d_out = self._unit(p, coords.get(nb))
            if d_out is None:
                continue
            best_depth[(position, nb)] = 1
            visible.add(nb)
            queue.append((nb, d_out, 1))

        while queue:
            u, d_in, depth = queue.popleft()
            if depth >= self.k:
                continue
            cu = coords[u]
            for w in self.adj.get(u, ()):
                d_out = self._unit(cu, coords.get(w))
                if d_out is None:
                    continue
                if not no_gate and self._dot(d_in, d_out) < cos_thresh:
                    continue
                nd = depth + 1
                if best_depth.get((u, w), 1 << 30) <= nd:
                    continue
                best_depth[(u, w)] = nd
                visible.add(w)
                queue.append((w, d_out, nd))
        return visible

    # --- surveillance tour planning -------------------------------------------
    def _build_surveil_tour(self, positions: list[int]) -> None:
        """Plan a coverage tour over the known map (once per surveil phase)."""
        nodes = [v for v in self.observed if v in self.adj]
        if not nodes:
            return

        # Seed with what the UAVs already see -- the start balls are free.
        covered_seed: set[int] = set(self.seen_phase)
        for p in positions:
            covered_seed |= self._los_ball(p)

        viewpoints = self._cover_viewpoints(nodes, covered_seed)
        if not viewpoints:
            return

        groups = self._kmeans_clusters(viewpoints, positions)
        for i in range(self.n_agents):
            order = self._nn_order(groups[i], positions[i])
            order = self._two_opt(order, positions[i])
            self.tour[i] = deque(order)

    def _cover_viewpoints(self, nodes: list[int], covered_seed: set[int]) -> list[int]:
        """Lazy-greedy set cover: fewest viewpoints whose LOS balls reach the
        target, starting from what is already covered."""
        balls = {v: self._los_ball(v) for v in nodes}
        target_count = int(math.ceil(self.surveil_target * len(nodes)))

        covered = set(covered_seed)
        chosen: list[int] = []
        pq = [(-len(balls[v] - covered), v) for v in nodes]
        heapq.heapify(pq)
        while pq and len(covered) < target_count:
            neg_gain, v = heapq.heappop(pq)
            gain = len(balls[v] - covered)
            if gain != -neg_gain:            # stale estimate -> reinsert fresh
                if gain > 0:
                    heapq.heappush(pq, (-gain, v))
                continue
            if gain == 0:
                break
            chosen.append(v)
            covered |= balls[v]
        return chosen

    def _kmeans_clusters(
        self, nodes: list[int], positions: list[int]
    ) -> list[list[int]]:
        """k-means"""
        cents = [self.coords[positions[i]] for i in range(self.n_agents)]
        groups: list[list[int]] = [[] for _ in range(self.n_agents)]
        for _ in range(8):
            groups = [[] for _ in range(self.n_agents)]
            for v in nodes:
                c = self.coords[v]
                gi = min(range(self.n_agents), key=lambda j: self._sqd(c, cents[j]))
                groups[gi].append(v)
            new_cents: list[tuple[float, float, float]] = []
            for j in range(self.n_agents):
                if groups[j]:
                    pts = [self.coords[v] for v in groups[j]]
                    n = len(pts)
                    new_cents.append((
                        sum(p[0] for p in pts) / n,
                        sum(p[1] for p in pts) / n,
                        sum(p[2] for p in pts) / n,
                    ))
                else:
                    new_cents.append(cents[j])
            if new_cents == cents:
                break
            cents = new_cents
        return groups

    def _nn_order(self, group: list[int], start_node: int) -> list[int]:
        """Greedy euclidean proxy"""
        remaining = set(group)
        cur = self.coords.get(start_node)
        order: list[int] = []
        while remaining:
            nxt = min(remaining, key=lambda v: self._sqd(self.coords[v], cur))
            order.append(nxt)
            remaining.discard(nxt)
            cur = self.coords[nxt]
        return order

    def _two_opt(self, order: list[int], start: int) -> list[int]:
        """2-opt algo
        """
        if not (3 <= len(order) <= self._two_opt_cap):
            return order

        seq = [start] + order
        dmat = self._pairwise_dist(seq)
        inf = math.inf

        def d(a: int, b: int) -> float:
            return dmat.get(a, {}).get(b, inf)

        n = len(seq)
        improved = True
        passes = 0
        while improved and passes < 40:
            improved = False
            passes += 1
            for i in range(n - 1):
                a, b = seq[i], seq[i + 1]
                d_ab = d(a, b)
                for j in range(i + 2, n):
                    c = seq[j]
                    nxt = seq[j + 1] if j + 1 < n else None
                    if nxt is None:                       # reversing the tail
                        delta = d(a, c) - d_ab
                    else:
                        delta = (d(a, c) + d(b, nxt)) - (d_ab + d(c, nxt))
                    if delta < -1e-9:
                        seq[i + 1:j + 1] = seq[i + 1:j + 1][::-1]
                        b = seq[i + 1]
                        d_ab = d(a, b)
                        improved = True
        return seq[1:]

    def _pairwise_dist(self, seq: list[int]) -> dict[int, dict[int, float]]:
        """Graph shortest-path distances among the nodes in ``seq``."""
        members = set(seq)
        dmat: dict[int, dict[int, float]] = {}
        for s in seq:
            if s in dmat:
                continue
            dist = self._dist_from(s)
            dmat[s] = {t: dist[t] for t in members if t in dist}
        return dmat

    def _dist_from(self, start: int) -> dict[int, float]:
        """Dijkstra distance from start to every reachable known node."""
        dist = {start: 0.0}
        if start not in self.adj:
            return dist
        pq: list[tuple[float, int]] = [(0.0, start)]
        while pq:
            d_, u = heapq.heappop(pq)
            if d_ > dist.get(u, math.inf):
                continue
            for v, w in self.adj.get(u, {}).items():
                nd = d_ + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    # --- routing ---------------------------------------------------------------
    def _next_hop(
        self,
        i: int,
        pos: int,
        covered: set[int],
        claimed: list[tuple[float, float, float]],
    ) -> int:
        path = self.path[i]
        
        while path and path[0] == pos:
            path.pop(0)

        target = self.target[i]
        
        done = (target in self.visited) if self.is_probe[i] else (target in covered)
        need_replan = (
            not path
            or path[0] not in self.adj.get(pos, {})        
            or target is None
            or target not in self.observed
            or done
        )
        if need_replan:
            target, path, probe = self._plan(i, pos, covered, claimed)
            self.target[i] = target
            self.path[i] = path
            self.is_probe[i] = probe

        if target is not None:
            claimed.append(self.coords[target])

        if path:
            return path[0]
        
        return self._wander(pos, covered)

    def _plan(
        self,
        i: int,
        pos: int,
        covered: set[int],
        claimed: list[tuple[float, float, float]],
    ) -> tuple[int | None, list[int], bool]:
        """Pick the next target on the known map.

        1. Sweep to the nearest uncovered node, kept away from other UAVs'
           targets.
        2. Same, without the spread constraint.
        3. Probe the nearest unvisited node
        """
        if self.phase == "surveil":
            tour = self.tour[i]
            while tour:
                wp = tour[0]
                if wp in self.seen_phase:
                    tour.popleft()
                    continue
                t, path = self._dijkstra(pos, lambda n, wp=wp: n == wp)
                if t is None:                       # unreachable on known map
                    tour.popleft()
                    continue
                return wp, path, False
        
        if self.phase != "surveil" and self._explore_stall >= self.stall_limit:
            bt, bp = self._breakout(pos)
            if bt is not None:
                return bt, bp, True

        base_r2 = self._scan_radius * self._scan_radius

        def spread_ok(n: int, r2: float) -> bool:
            cx, cy, cz = self.coords[n]
            for (ax, ay, az) in claimed:
                dx, dy, dz = cx - ax, cy - ay, cz - az
                if dx * dx + dy * dy + dz * dz < r2:
                    return False
            return True

        radii = []
        if self.spread_mult > 1.0:
            radii.append(base_r2 * self.spread_mult * self.spread_mult)
        radii.append(base_r2)
        for r2 in radii:
            target, path = self._dijkstra(
                pos,
                lambda n, r2=r2: n not in covered and n in self.observed and spread_ok(n, r2),
            )
            if target is not None:
                return target, path, False

        target, path = self._dijkstra(pos, lambda n: n not in covered and n in self.observed)
        if target is not None:
            return target, path, False

        target, path = self._dijkstra(pos, lambda n: n not in self.visited)
        return target, path, True

    def _breakout(self, pos: int) -> tuple[int | None, list[int]]:
        """Frontier probe"""
        dist = {pos: 0.0}
        prev: dict[int, int] = {}
        pq: list[tuple[float, int]] = [(0.0, pos)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            for v, w in self.adj.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        best = None
        best_key = None
        for n, dn in dist.items():
            if n in self.visited:
                continue
            key = (len(self.adj.get(n, ())), -dn)
            if best_key is None or key < best_key:
                best_key, best = key, n
        if best is None:
            return None, []
        return best, self._rebuild(prev, pos, best)

    def _dijkstra(self, start, is_goal):
        """dijkstra"""
        if start not in self.adj:
            return None, []
        dist = {start: 0.0}
        prev: dict[int, int] = {}
        pq: list[tuple[float, int]] = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            if u != start and is_goal(u):
                return u, self._rebuild(prev, start, u)
            for v, w in self.adj.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return None, []

    @staticmethod
    def _rebuild(prev: dict[int, int], start: int, goal: int) -> list[int]:
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def _wander(self, pos: int, covered: set[int]) -> int:
        nbrs = list(self.adj.get(pos, {}))
        if not nbrs:
            return pos
        fresh = [n for n in nbrs if n not in covered]
        pool = fresh if fresh else nbrs
        return self.rng.choice(pool)

    # --- conflict resolution ---------------------------------------------------
    def _resolve(
        self,
        positions: list[int],
        desired: list[int],
        covered: set[int],
    ) -> list[int]:
        final = list(positions)            
        dest_owner: dict[int, int] = {}    
        committed: dict[int, int] = {}     

        for i in range(self.n_agents):  
            pos = positions[i]
            options = self._move_options(i, pos, desired[i], covered)
            chosen = pos
            for opt in options:
                if opt == pos:
                    chosen = pos
                    break
                if opt in dest_owner:  # vertex conflict
                    continue
                
                if any(committed.get(j) == pos and positions[j] == opt
                       for j in committed):
                    continue
                chosen = opt
                break
            final[i] = chosen
            if chosen != pos:
                dest_owner[chosen] = i
                committed[i] = chosen
                self.stall[i] = 0
                if self.path[i] and self.path[i][0] == chosen:
                    self.path[i].pop(0)
            else:
                self.stall[i] += 1
        return final

    def _move_options(self, i: int, pos: int, desired: int, covered: set[int]) -> list[int]:
        """Preferred next nodes"""
        opts: list[int] = []
        if desired != pos:
            opts.append(desired)
        nbrs = list(self.adj.get(pos, {}))
        tgt = self.target[i]
        if tgt is not None and tgt in self.coords:
            tx, ty, tz = self.coords[tgt]
            nbrs.sort(key=lambda n: self._sqd(self.coords.get(n, (tx, ty, tz)), (tx, ty, tz)))
        else:
            fresh = [n for n in nbrs if n not in covered]
            if fresh:
                nbrs = fresh
        for n in nbrs:
            if n not in opts:
                opts.append(n)
        opts.append(pos)  
        return opts

    @staticmethod
    def _sqd(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
