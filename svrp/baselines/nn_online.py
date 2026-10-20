"""Nearest-feasible-neighbour, re-decided at every stop (an *online* heuristic, like a driver without a plan).

Rule: drive to the closest (by estimated cost) open customer whose believed demand fits on the truck;
if nobody fits, go back to the depot and refill. Any learned policy should at least beat this.
"""
from __future__ import annotations

import torch

from svrp.data import Batch
from svrp.env.svrp_env import EPS, SVRPEnv


@torch.no_grad()
def nn_online(batch: Batch, delivery: str = "partial", split: bool = False) -> SVRPEnv:
    """split=True ("nn_split"): always drive to the nearest open customer and hand over whatever is on
    the truck; go home only when empty. A control for how much of RL's gain is just "split deliveries"."""
    env = SVRPEnv(batch, delivery)
    inf = torch.tensor(float("inf"), device=batch.cost.device)
    for _ in range(env.max_steps()):
        if env.done.all():
            break
        m = env.mask()
        cust = m.clone()
        cust[:, 0] = False
        c = batch.cost_hat[env.ar, env.pos]
        fits = cust if split else cust & (env.observed_demand() <= env.load[:, None] + EPS)
        nearest_fit = torch.where(fits, c, inf).argmin(1)
        nearest_any = torch.where(cust, c, inf).argmin(1)
        at_depot_open = (env.pos == 0) & cust.any(1)     # at the depot with a full truck: just go
        a = torch.where(fits.any(1), nearest_fit, torch.where(at_depot_open, nearest_any, 0))
        env.step(a)
    return env
