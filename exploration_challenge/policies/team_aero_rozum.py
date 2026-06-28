from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from typing import Optional

from exploration_challenge.observation import Observation


class Explorer:
    K = 4

    def reset(
        self,
        starts: list[int],
        observations: list[Observation],
        seed: int | None = None,
    ) -> None:
        self.n_agents = len(starts)
        self.adj: dict[int, dict[int, float]] = defaultdict(dict)
        self.xyz: dict[int, tuple[float, float, float]] = {}
        self.cur = list(starts)
        self.path: list[list[int]] = [[] for _ in starts]
        self.target: list[Optional[int]] = [None for _ in starts]
        self.load = [0.0 for _ in starts]

        self.phys_seen: set[int] = set(starts)
        self.detected: set[int] = set()
        self.survey_seen: set[int] = set()
        self.rim: set[int] = set()
        self.phase = "explore"
        self.tick = 0
        self.prev_pos = list(starts)
        self.stalled = [0 for _ in starts]
        self.territory: Optional[list[set[int]]] = None
        self.ball_cache: dict[tuple[int, int], set[int]] = {}
        self.detected_mark = 0
        self.stagnant_explore = 0
        self.warehouse_mode = False

        for obs in observations:
            self._absorb(obs, "explore")
        self.detected_mark = len(self.detected)

    def step(self, observations: list[Observation], phase: str) -> list[int]:
        self.tick += 1
        if phase != self.phase:
            self.phase = phase
            self.path = [[] for _ in range(self.n_agents)]
            self.target = [None for _ in range(self.n_agents)]
            self.territory = None

        self.ball_cache.clear()
        for obs in observations:
            self.cur[obs.agent_id] = obs.position
            self._absorb(obs, phase)
        if phase == "explore" and not self.warehouse_mode:
            self.warehouse_mode = self._compute_warehouse_like()

        if phase == "explore":
            if len(self.detected) >= self.detected_mark + 8:
                self.detected_mark = len(self.detected)
                self.stagnant_explore = 0
            else:
                self.stagnant_explore += 1

        if phase == "surveil" and self.territory is None:
            self._make_territory()
        if phase == "surveil":
            for pos in self.cur:
                self.survey_seen.update(self._ball(pos, self.K))

        self._unstick_dead_paths()
        actions = self._plan_explore() if phase == "explore" else self._plan_surveil()
        actions = self._avoid_collisions(actions)

        for i, nxt in enumerate(actions):
            here = self.cur[i]
            if nxt != here and nxt in self.adj.get(here, {}):
                self.load[i] += self.adj[here][nxt]
        return actions

    def _absorb(self, obs: Observation, phase: str) -> None:
        self.phys_seen.update(obs.visited)
        self.phys_seen.add(obs.position)
        for n in obs.nodes:
            self.xyz[n.id] = (n.x, n.y, n.z)
            self.adj.setdefault(n.id, {})
            if phase == "surveil":
                self.survey_seen.add(n.id)
            else:
                self.detected.add(n.id)
        if phase == "surveil":
            self.survey_seen.add(obs.position)
        else:
            self.detected.add(obs.position)
            self._remember_rim(obs)
        for e in obs.edges:
            self.adj[e.u][e.v] = float(e.cost)
            self.adj[e.v][e.u] = float(e.cost)

    def _remember_rim(self, obs: Observation) -> None:
        visible = {n.id for n in obs.nodes}
        visible.add(obs.position)
        local: dict[int, list[int]] = defaultdict(list)
        for e in obs.edges:
            if e.u in visible and e.v in visible:
                local[e.u].append(e.v)
                local[e.v].append(e.u)
        dist = {obs.position: 0}
        q = deque([obs.position])
        while q:
            u = q.popleft()
            for v in local.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for node, d in dist.items():
            if 3 <= d <= self.K and node != obs.position:
                self.rim.add(node)

    def _unstick_dead_paths(self) -> None:
        for i, pos in enumerate(self.cur):
            self.stalled[i] = self.stalled[i] + 1 if pos == self.prev_pos[i] else 0
            self.prev_pos[i] = pos
            if self.stalled[i] >= 4:
                self.path[i] = []
                self.target[i] = None
                self.stalled[i] = 0

    def _plan_explore(self) -> list[int]:
        dmaps = [self._dijkstra(pos)[0] for pos in self.cur]
        frontiers = self._frontiers()
        portal_nodes: set[int] = set()
        candidates = set(frontiers)
        warehouse = False
        slow = self._slow_explore()

        for i in range(self.n_agents):
            t = self.target[i]
            ok = (
                t is not None
                and t != self.cur[i]
                and t in dmaps[i]
                and t in frontiers
                and not slow
            )
            if not ok:
                self.target[i] = None
                self.path[i] = []

        claimed = {t for t in self.target if t is not None}
        for aid in sorted(range(self.n_agents), key=lambda i: self.load[i]):
            if self.target[aid] is not None:
                continue
            goal = self._best_explore_goal(aid, candidates, portal_nodes, dmaps, claimed, warehouse)
            if goal is None and (slow or self.tick > 180) and self.tick < 520:
                portal_nodes = self._portal_work()
                goal = self._best_explore_goal(aid, portal_nodes, portal_nodes, dmaps, claimed, warehouse)
            if goal is None:
                goal = self._nearest_need_visit(self.cur[aid], claimed)
            self.target[aid] = goal
            if goal is not None:
                claimed.add(goal)
                self.path[aid] = self._route(self.cur[aid], goal)
        return self._next_steps()

    def _frontiers(self) -> set[int]:
        out: set[int] = set()
        for u, nbrs in self.adj.items():
            if any(v not in self.phys_seen for v in nbrs):
                out.add(u)
        return out

    def _best_explore_goal(
        self,
        aid: int,
        candidates: set[int],
        portal_nodes: set[int],
        dmaps: list[dict[int, float]],
        claimed: set[int],
        warehouse: bool,
    ) -> Optional[int]:
        best: tuple[float, int] | None = None
        for node in candidates:
            if node == self.cur[aid] or node in claimed:
                continue
            dist = dmaps[aid].get(node, float("inf"))
            if dist == float("inf"):
                continue
            near = self._ball(node, 1)
            fresh_nbrs = sum(1 for n in near if n not in self.phys_seen)
            old_nbrs = len(near) - fresh_nbrs
            local_frontier = fresh_nbrs / (old_nbrs + 1.0)
            long_edge = max(self.adj.get(node, {}).values(), default=0.0)
            other_dist = min(
                (dmaps[j].get(node, float("inf")) for j in range(self.n_agents) if j != aid),
                default=0.0,
            )
            if warehouse:
                degree = len(self.adj.get(node, {}))
                lane_bonus = 1.32 if self._in_agent_lane(aid, node) else 0.82
                isolation = min(other_dist, 28.0)
                score = (
                    96.0 * local_frontier
                    + 7.5 * isolation
                    + 7.0 * degree
                    + 0.10 * self._eccentricity(node)
                    - 31.0 * math.log1p(dist)
                    - 0.40 * dist
                ) * lane_bonus
            else:
                slow = self._slow_explore()
                isolation = min(other_dist, 500.0 if slow else 100.0)
                dist_weight = 5.0 if slow else 70.0
                score = (
                    42.28 * local_frontier
                    + 22.67 * isolation
                    + 123.67 * long_edge
                    - dist_weight * math.log1p(dist)
                )
                if node in portal_nodes:
                    score += 55.0
            if all(dmaps[j].get(node, float("inf")) >= dist for j in range(self.n_agents) if j != aid):
                score *= 1.15
            item = (score, node)
            if best is None or item > best:
                best = item
        return None if best is None else best[1]

    def _explore_value(self, node: int) -> int:
        return sum(1 for n in self._ball(node, self.K) if n not in self.detected)

    def _slow_explore(self) -> bool:
        return self.tick >= 200

    def _portal_work(self) -> set[int]:
        long_edges = self._long_edges()
        if not long_edges:
            return set()
        comp = self._components_ignoring(long_edges)
        if not comp:
            return set()
        groups: dict[int, list[int]] = defaultdict(list)
        for node, cid in comp.items():
            groups[cid].append(node)
        ranked: list[tuple[float, int]] = []
        for cid, nodes in groups.items():
            if len(nodes) < 5:
                continue
            unseen = [n for n in nodes if n not in self.detected or n not in self.phys_seen]
            if not unseen:
                continue
            frac = len(unseen) / len(nodes)
            ranked.append((len(unseen) * (1.0 + 1.8 * frac) / (1.0 + 0.001 * len(nodes)), cid))
        if not ranked:
            return set()
        ranked.sort(reverse=True)
        keep = {cid for _, cid in ranked[: min(2, len(ranked))]}
        return {n for n, cid in comp.items() if cid in keep and n not in self.phys_seen}

    def _long_edges(self) -> set[tuple[int, int]]:
        costs = [c for u, nbrs in self.adj.items() for v, c in nbrs.items() if u < v]
        if len(costs) < 70:
            return set()
        costs.sort()
        med = costs[len(costs) // 2]
        p98 = costs[int(0.98 * (len(costs) - 1))]
        cutoff = max(2.55 * med, p98)
        return {
            (u, v)
            for u, nbrs in self.adj.items()
            for v, c in nbrs.items()
            if u < v and c >= cutoff
        }

    def _components_ignoring(self, blocked: set[tuple[int, int]]) -> dict[int, int]:
        comp: dict[int, int] = {}
        cid = 0
        for start in self.adj:
            if start in comp:
                continue
            comp[start] = cid
            q = deque([start])
            while q:
                u = q.popleft()
                for v in self.adj.get(u, {}):
                    edge = (u, v) if u < v else (v, u)
                    if edge in blocked or v in comp:
                        continue
                    comp[v] = cid
                    q.append(v)
            cid += 1
        return comp if cid > 1 else {}

    def _plan_surveil(self) -> list[int]:
        dmaps = [self._dijkstra(pos)[0] for pos in self.cur]
        candidates = self._survey_candidates()
        warehouse = False
        for i in range(self.n_agents):
            t = self.target[i]
            if t is None or t == self.cur[i] or self._survey_value(t) <= 0:
                self.target[i] = None
                self.path[i] = []

        claimed = {t for t in self.target if t is not None}
        reserved_cover: set[int] = set()
        for t in claimed:
            if t is not None:
                reserved_cover.update(self._ball(t, self.K) - self.survey_seen)
        for aid in sorted(range(self.n_agents), key=lambda i: self.load[i]):
            if self.target[aid] is not None:
                continue
            goal = self._best_survey_goal(aid, candidates, dmaps[aid], claimed, reserved_cover, warehouse)
            self.target[aid] = goal
            if goal is not None:
                claimed.add(goal)
                reserved_cover.update(self._ball(goal, self.K) - self.survey_seen)
                self.path[aid] = self._route(self.cur[aid], goal)
        return self._next_steps()

    def _survey_candidates(self) -> set[int]:
        out = set()
        for u, nbrs in self.adj.items():
            if u not in self.survey_seen or any(v not in self.survey_seen for v in nbrs):
                out.add(u)
        return out

    def _best_survey_goal(
        self,
        aid: int,
        candidates: set[int],
        dist: dict[int, float],
        claimed: set[int],
        reserved_cover: set[int],
        warehouse: bool,
    ) -> Optional[int]:
        own = self.territory[aid] if self.territory else set()
        best: tuple[float, int] | None = None
        for node in candidates:
            if node in claimed:
                continue
            d = dist.get(node, float("inf"))
            if d == float("inf"):
                continue
            gain = sum(1 for n in self._ball(node, self.K) if n not in self.survey_seen and n not in reserved_cover)
            if gain <= 0:
                continue
            if warehouse:
                score = (gain * gain) / (1.0 + 0.95 * d)
            else:
                score = (6.29 * gain) / (1.0 + 1.72 * d)
            if node in own:
                score *= 4.0 if warehouse else 8.0
            item = (score, node)
            if best is None or item > best:
                best = item
        return None if best is None else best[1]

    def _survey_value(self, node: int) -> int:
        return sum(1 for n in self._ball(node, self.K) if n not in self.survey_seen)

    def _warehouse_like(self) -> bool:
        return self.warehouse_mode

    def _compute_warehouse_like(self) -> bool:
        if len(self.adj) < 120 or not self.xyz:
            return False
        costs = [c for u, nbrs in self.adj.items() for v, c in nbrs.items() if u < v]
        if len(costs) < 180:
            return False
        costs.sort()
        med = costs[len(costs) // 2]
        if med > 1.35:
            return False
        zs = [p[2] for p in self.xyz.values()]
        if max(zs) - min(zs) > 2.4:
            return False
        high_degree = sum(1 for nbrs in self.adj.values() if len(nbrs) >= 17)
        return high_degree >= 4 or high_degree / max(1, len(self.adj)) > 0.035

    def _in_agent_lane(self, aid: int, node: int) -> bool:
        if node not in self.xyz or len(self.xyz) < 10:
            return True
        xs = [p[0] for p in self.xyz.values()]
        ys = [p[1] for p in self.xyz.values()]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        axis = 0 if span_x >= span_y else 1
        values = [p[axis] for p in self.xyz.values()]
        lo, hi = min(values), max(values)
        if hi <= lo:
            return True
        starts = [(i, self.xyz.get(self.cur[i], (0.0, 0.0, 0.0))[axis]) for i in range(self.n_agents)]
        ordered = [i for i, _ in sorted(starts, key=lambda item: item[1])]
        rank = ordered.index(aid) if aid in ordered else aid
        val = self.xyz[node][axis]
        lane = min(self.n_agents - 1, int((val - lo) / (hi - lo + 1e-9) * self.n_agents))
        return lane == rank

    def _make_territory(self) -> None:
        dmaps = [self._dijkstra(pos)[0] for pos in self.cur]
        terr = [set() for _ in range(self.n_agents)]
        for node in self.adj:
            owner = min(range(self.n_agents), key=lambda i: dmaps[i].get(node, float("inf")))
            terr[owner].add(node)
        self.territory = terr

    def _nearest_need_visit(self, start: int, claimed: set[int]) -> Optional[int]:
        q = deque([start])
        dist = {start: 0}
        while q:
            u = q.popleft()
            if u != start and u not in claimed and u not in self.phys_seen:
                return u
            for v in sorted(self.adj.get(u, {}), key=lambda n: (n in self.phys_seen, self.adj[u][n])):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return max((n for n in dist if n != start and n not in claimed), key=lambda n: dist[n], default=None)

    def _next_steps(self) -> list[int]:
        moves: list[int] = []
        for i, pos in enumerate(self.cur):
            while self.path[i] and self.path[i][0] == pos:
                self.path[i].pop(0)
            if self.path[i] and self.path[i][0] in self.adj.get(pos, {}):
                moves.append(self.path[i][0])
            else:
                self.path[i] = []
                self.target[i] = None
                moves.append(self._local_step(pos))
        return moves

    def _avoid_collisions(self, actions: list[int]) -> list[int]:
        out = list(actions)
        occupied: set[int] = set()
        for i in range(self.n_agents):
            pos = self.cur[i]
            nxt = out[i]
            if nxt != pos and nxt not in self.adj.get(pos, {}):
                nxt = self._local_step(pos)
            if nxt in occupied:
                alt = self._local_step(pos, occupied | set(self.cur))
                nxt = alt if alt not in occupied else pos
            out[i] = nxt
            occupied.add(nxt)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if out[i] == self.cur[j] and out[j] == self.cur[i] and out[i] != self.cur[i]:
                    out[j] = self.cur[j]
                    self.path[j] = []
                    self.target[j] = None
        return out

    def _local_step(self, pos: int, avoid: set[int] | None = None) -> int:
        avoid = avoid or set()
        choices = [n for n in self.adj.get(pos, {}) if n not in avoid]
        if not choices:
            return pos
        if self.phase == "surveil":
            return max(choices, key=lambda n: (self._survey_value(n), n not in self.survey_seen, -self.adj[pos][n]))
        return max(
            choices,
            key=lambda n: (
                n not in self.phys_seen,
                n not in self.detected,
                len([v for v in self.adj.get(n, {}) if v not in self.phys_seen]),
                -self.adj[pos][n],
            ),
        )

    def _route(self, start: int, goal: int) -> list[int]:
        _, prev = self._dijkstra(start, {goal})
        if goal not in prev:
            return []
        route = []
        cur: Optional[int] = goal
        while cur is not None:
            route.append(cur)
            cur = prev.get(cur)
        route.reverse()
        if route and route[0] == start:
            route.pop(0)
        return route

    def _dijkstra(
        self,
        start: int,
        goals: Optional[set[int]] = None,
    ) -> tuple[dict[int, float], dict[int, Optional[int]]]:
        dist = {start: 0.0}
        prev: dict[int, Optional[int]] = {start: None}
        heap = [(0.0, start)]
        done: set[int] = set()
        remaining = set(goals) if goals else None
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            if remaining is not None:
                remaining.discard(u)
                if not remaining:
                    break
            for v, w in self.adj.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return dist, prev

    def _ball(self, start: int, depth: int) -> set[int]:
        key = (start, depth)
        if key in self.ball_cache:
            return self.ball_cache[key]
        seen = {start}
        q = deque([(start, 0)])
        while q:
            u, d = q.popleft()
            if d >= depth:
                continue
            for v in self.adj.get(u, {}):
                if v not in seen:
                    seen.add(v)
                    q.append((v, d + 1))
        self.ball_cache[key] = seen
        return seen

    def _los_ball(self, start: int, depth: int) -> set[int]:
        key = (start, -depth)
        if key in self.ball_cache:
            return self.ball_cache[key]
        if depth <= 0 or start not in self.adj:
            return {start}
        cos_thresh = math.cos(math.radians(75.0))
        visible = {start}
        q: deque[tuple[int, tuple[float, float, float], int]] = deque()
        best_depth: dict[tuple[int, int], int] = {}
        for nb in self.adj.get(start, {}):
            d = self._edge_dir(start, nb)
            if d is None:
                continue
            visible.add(nb)
            best_depth[(start, nb)] = 1
            q.append((nb, d, 1))
        while q:
            u, d_in, dist = q.popleft()
            if dist >= depth:
                continue
            for v in self.adj.get(u, {}):
                d_out = self._edge_dir(u, v)
                if d_out is None:
                    continue
                if d_in[0] * d_out[0] + d_in[1] * d_out[1] + d_in[2] * d_out[2] < cos_thresh:
                    continue
                nd = dist + 1
                if best_depth.get((u, v), 10**9) <= nd:
                    continue
                best_depth[(u, v)] = nd
                visible.add(v)
                q.append((v, d_out, nd))
        self.ball_cache[key] = visible
        return visible

    def _edge_dir(self, u: int, v: int) -> Optional[tuple[float, float, float]]:
        a = self.xyz.get(u)
        b = self.xyz.get(v)
        if a is None or b is None:
            return None
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm <= 1e-12:
            return None
        return (dx / norm, dy / norm, dz / norm)

    def _eccentricity(self, node: int) -> float:
        if node not in self.xyz or not self.xyz:
            return 0.0
        cx = sum(p[0] for p in self.xyz.values()) / len(self.xyz)
        cy = sum(p[1] for p in self.xyz.values()) / len(self.xyz)
        cz = sum(p[2] for p in self.xyz.values()) / len(self.xyz)
        x, y, z = self.xyz[node]
        return abs(x - cx) + abs(y - cy) + 0.3 * abs(z - cz)