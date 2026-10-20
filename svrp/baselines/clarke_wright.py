"""Clarke–Wright savings (1964): start with one route per customer, then greedily merge routes.

Saving of joining i -> j into one route instead of two separate depot trips:
    s_ij = c_i0 + c_0j − c_ij
Merge pairs in decreasing order of saving, as long as i ends one route, j starts another,
and the merged route still fits in the truck.
"""
from __future__ import annotations

import numpy as np

from svrp.baselines.common import planning_demand, routes_to_plan


def clarke_wright(dem: np.ndarray, cost: np.ndarray, capacity: float) -> list[int]:
    N1 = len(dem)
    d = planning_demand(dem, capacity)
    route_of = np.arange(N1)                      # route id of each customer (id = its first customer)
    routes = {i: [i] for i in range(1, N1)}
    load = {i: d[i] for i in range(1, N1)}

    i_idx, j_idx = np.where(~np.eye(N1, dtype=bool))
    keep = (i_idx > 0) & (j_idx > 0)
    i_idx, j_idx = i_idx[keep], j_idx[keep]
    s = cost[i_idx, 0] + cost[0, j_idx] - cost[i_idx, j_idx]
    order = np.argsort(-s, kind="stable")

    for k in order:
        if s[k] <= 0:
            break
        i, j = int(i_idx[k]), int(j_idx[k])
        ri, rj = int(route_of[i]), int(route_of[j])
        if ri == rj or load[ri] + load[rj] > capacity + 1e-9:
            continue
        a, b = routes[ri], routes[rj]
        if a[-1] != i or b[0] != j:               # i must end its route, j must start its route
            continue
        merged = a + b
        del routes[rj]
        routes[ri] = merged
        load[ri] += load.pop(rj)
        for c in b:
            route_of[c] = ri
    return routes_to_plan(list(routes.values()))
