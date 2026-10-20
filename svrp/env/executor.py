"""Execute an a-priori plan on the real day ("plan in the morning, then drive it no matter what").

A plan is a node sequence like [0, 3, 5, 0, 2, 0] made from *estimates*. Reality then differs:
  * arrive with too little load  -> detour-to-depot recourse: round trip, refill, finish the customer
  * truck is empty mid-route      -> go back to the depot before the next planned customer
  * customer already fully served -> skip it
This is the classical "recourse" policy from the SVRP literature (the R(x) term in a.md).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def execute(plan, demand: np.ndarray, cost: np.ndarray, capacity: float, delivery: str = "partial"):
    """Returns (total_cost, n_fail). demand: (N+1,), cost: (N+1, N+1), both TRUE values."""
    rem = demand.astype(np.float64).copy()
    pos, load, total, n_fail = 0, capacity, 0.0, 0
    for node in plan:
        node = int(node)
        if node == 0:
            if pos != 0:
                total += cost[pos, 0]
                pos, load = 0, capacity
            continue
        if rem[node] <= EPS:
            continue
        if load <= EPS:                          # empty truck: restock before driving on
            total += cost[pos, 0]
            pos, load = 0, capacity
        total += cost[pos, node]
        pos = node
        if load >= rem[node] - EPS:
            load -= rem[node]
            rem[node] = 0.0
            continue
        n_fail += 1                              # recourse
        if delivery == "partial":
            rem[node] -= load                    # hand over what we have,
            while rem[node] > EPS:               # then shuttle to the depot until the order is complete
                total += cost[node, 0] + cost[0, node]
                give = min(capacity, rem[node])
                rem[node] -= give
                load = capacity - give
        else:
            total += cost[node, 0] + cost[0, node]   # indivisible: go refill, come back, deliver it all
            load = capacity - rem[node]
            rem[node] = 0.0
    if pos != 0:
        total += cost[pos, 0]
    if (rem[1:] > EPS).any():
        missing = np.flatnonzero(rem[1:] > EPS) + 1
        raise ValueError(f"plan does not visit customers {missing.tolist()}")
    return total, n_fail


def execute_many(plans, demand: np.ndarray, cost: np.ndarray, capacity: float, delivery: str = "partial"):
    """plans: one plan per day, or a single plan reused for every day. Returns arrays (costs, n_fail)."""
    B = demand.shape[0]
    single = len(plans) > 0 and np.isscalar(plans[0])
    out = np.empty(B)
    fails = np.empty(B)
    for b in range(B):
        out[b], fails[b] = execute(plans if single else plans[b], demand[b], cost[b], capacity, delivery)
    return out, fails
