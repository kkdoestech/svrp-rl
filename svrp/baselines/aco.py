"""Ant Colony Optimization for CVRP (Ant System + elitist ant + 2-opt), planning on estimates.

Each ant builds a full solution: from its current node it picks the next customer with probability
    p(j) ∝ τ_ij^α · η_ij^β,   η_ij = 1 / c_ij   (pheromone^α × greediness^β),
among customers whose (estimated) demand still fits in the truck; if none fits it returns to the depot.
After each iteration pheromone evaporates (τ ← (1−ρ)τ) and every ant deposits 1/L on the edges it used,
plus an extra deposit on the best solution found so far ("elitist ant"). Good edges get reinforced.

All ants are simulated in parallel with numpy.
"""
from __future__ import annotations

import numpy as np

from svrp.baselines.common import plan_cost, plan_to_routes, planning_demand, routes_to_plan

# (alpha = pheromone importance, beta = heuristic importance, ants). The paper's Table 1 lists
# "3/4/20, 6/8/24, 4/9/15, 3/10/12" without labelling the columns; we read them in this order
# and add two textbook settings. Tuned on the validation set (scripts/tune_aco.py).
PARAM_GRID = [(3, 4, 20), (6, 8, 24), (4, 9, 15), (3, 10, 12), (1, 2, 20), (1, 5, 20)]


def two_opt(route: list[int], cost: np.ndarray) -> list[int]:
    """Classic 2-opt on one route (depot at both ends): reverse segments while that shortens it."""
    r = [0] + route + [0]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(r) - 2):
            for j in range(i + 1, len(r) - 1):
                delta = (cost[r[i - 1], r[j]] + cost[r[i], r[j + 1]]
                         - cost[r[i - 1], r[i]] - cost[r[j], r[j + 1]])
                if delta < -1e-12:
                    r[i:j + 1] = r[i:j + 1][::-1]
                    improved = True
    return r[1:-1]


def _construct(tau, eta_b, d, capacity, n_ants, alpha, rng):
    N1 = len(d)
    A = np.arange(n_ants)
    visited = np.zeros((n_ants, N1), dtype=bool)
    visited[:, 0] = True
    pos = np.zeros(n_ants, dtype=np.int64)
    load = np.full(n_ants, capacity, dtype=np.float64)
    seqs = [[0] for _ in range(n_ants)]
    tau_a = tau ** alpha
    while not visited.all():
        feasible = ~visited & (d[None, :] <= load[:, None] + 1e-9)
        has = feasible.any(1)
        unfinished = ~visited.all(1)
        w = tau_a[pos] * eta_b[pos] * feasible
        nxt = np.zeros(n_ants, dtype=np.int64)
        if has.any():
            c = w[has].cumsum(1)                            # roulette wheel: one spin per ant
            u = rng.random(int(has.sum()))[:, None] * c[:, -1:]
            nxt[has] = (c < u).sum(1)
        # ants with nothing feasible go home to refill (nxt stays 0)
        move = unfinished
        for a in A[move]:
            seqs[a].append(int(nxt[a]))
        go_cust = move & (nxt != 0)
        visited[A[go_cust], nxt[go_cust]] = True
        load[go_cust] -= d[nxt[go_cust]]
        load[move & (nxt == 0)] = capacity
        pos[move] = nxt[move]
    for s in seqs:
        if s[-1] != 0:
            s.append(0)
    return seqs


def aco(dem: np.ndarray, cost: np.ndarray, capacity: float, alpha: float = 3, beta: float = 10,
        n_ants: int = 12, iters: int = 60, rho: float = 0.1, elite: float = 5.0, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    d = planning_demand(dem, capacity).astype(np.float64)
    d[0] = 0.0
    N1 = len(d)
    eta = 1.0 / np.maximum(cost, 1e-9)
    np.fill_diagonal(eta, 0.0)
    eta_b = eta ** beta
    eta_b /= eta_b.max()                                   # keep numbers in a safe float range

    tau = np.ones((N1, N1))
    best_plan, best_len = None, np.inf
    for _ in range(iters):
        plans = _construct(tau, eta_b, d, capacity, n_ants, alpha, rng)
        lens = np.array([plan_cost(p, cost) for p in plans])
        k = int(lens.argmin())
        polished = routes_to_plan([two_opt(r, cost) for r in plan_to_routes(plans[k])])
        plen = plan_cost(polished, cost)
        if plen < best_len:
            best_plan, best_len = polished, plen
        tau *= 1.0 - rho
        for p, L in zip(plans, lens):
            p = np.asarray(p)
            np.add.at(tau, (p[:-1], p[1:]), 1.0 / L)
        bp = np.asarray(best_plan)
        np.add.at(tau, (bp[:-1], bp[1:]), elite / best_len)
        tau = np.maximum(tau, 1e-6)
    return best_plan
