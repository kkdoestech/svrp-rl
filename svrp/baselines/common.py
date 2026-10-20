"""Helpers shared by the classical planners."""
from __future__ import annotations

import numpy as np


def routes_to_plan(routes: list[list[int]]) -> list[int]:
    """[[3, 5], [2]] -> [0, 3, 5, 0, 2, 0]"""
    plan = [0]
    for r in routes:
        if r:
            plan.extend(int(i) for i in r)
            plan.append(0)
    return plan


def plan_to_routes(plan) -> list[list[int]]:
    routes, cur = [], []
    for node in plan:
        if node == 0:
            if cur:
                routes.append(cur)
            cur = []
        else:
            cur.append(int(node))
    if cur:
        routes.append(cur)
    return routes


def plan_cost(plan, cost: np.ndarray) -> float:
    """Cost of driving the plan exactly as written (no recourse)."""
    p = list(plan)
    if p[0] != 0:
        p = [0] + p
    if p[-1] != 0:
        p = p + [0]
    p = np.asarray(p)
    return float(cost[p[:-1], p[1:]].sum())


def planning_demand(dem: np.ndarray, capacity: float) -> np.ndarray:
    """Planners can't handle orders bigger than a truck; clip (the executor's recourse covers the rest)."""
    return np.minimum(dem, capacity)
