"""The SVRP environment (an MDP), batched: B independent days are simulated in parallel on the GPU.

One truck, node 0 = depot. At every step the policy picks the next node to drive to.

What the policy can see (observe()):
    per node i:  [w, d_i^t / Q, ĉ_i0, …, ĉ_iN]
                 d_i^t is the *estimate* until the truck first arrives at i; from then on it is the true
                 remaining demand (demand is revealed on arrival, like a real customer telling the driver).
    truck:       current node, load / Q
What the policy never sees: the true demand of unvisited customers, the true travel costs.
What it pays: the TRUE cost of every edge it drives (+ automatic recourse under full delivery).
"""
from __future__ import annotations

import torch

from svrp.data import Batch

EPS = 1e-6


class SVRPEnv:
    def __init__(self, batch: Batch, delivery: str = "partial", weather_input: bool = True):
        if delivery not in ("partial", "full"):
            raise ValueError(delivery)
        self.batch = batch
        self.delivery = delivery
        self.weather_input = weather_input
        self.Q = batch.capacity
        B, N1 = batch.demand.shape
        dev = batch.demand.device
        self.ar = torch.arange(B, device=dev)
        self.pos = torch.zeros(B, dtype=torch.long, device=dev)
        self.load = torch.full((B,), self.Q, device=dev)
        self.remaining = batch.demand.clone()
        self.visited = torch.zeros(B, N1, dtype=torch.bool, device=dev)
        self.visited[:, 0] = True
        self.total_cost = torch.zeros(B, device=dev)
        self.n_fail = torch.zeros(B, device=dev)
        self.done = torch.zeros(B, dtype=torch.bool, device=dev)

    # ------------------------------------------------------------------ observation
    @property
    def n_nodes(self) -> int:
        return self.remaining.shape[1]

    def observed_demand(self) -> torch.Tensor:
        """Estimate for customers not visited yet, true remaining demand for those already visited."""
        return torch.where(self.visited, self.remaining, self.batch.dem_hat)

    def node_features(self) -> torch.Tensor:
        """(B, N+1, M + 1 + N+1): the per-node input of the paper's network."""
        b = self.batch
        B, N1 = self.remaining.shape
        w = b.w[:, None, :].expand(B, N1, b.w.shape[1])
        if not self.weather_input:
            w = torch.zeros_like(w)
        return torch.cat([w, (self.observed_demand() / self.Q)[..., None], b.cost_hat], dim=-1)

    def mask(self) -> torch.Tensor:
        """(B, N+1) bool, True = allowed. Never all-False."""
        open_ = self.remaining > EPS
        open_[:, 0] = False
        all_served = ~open_.any(1)
        m = open_ & (self.load > EPS)[:, None]           # an empty truck cannot serve anybody
        m[:, 0] = (self.pos != 0) | all_served           # no depot -> depot loops (unless finished)
        return m

    def max_steps(self) -> int:
        """Upper bound on episode length for ANY legal policy (see the counting argument in SPEC S3)."""
        N = self.n_nodes - 1
        trips_full = int(torch.ceil(self.batch.demand.sum(1).max() / self.Q).item())
        return 3 * N + 2 * trips_full + 2

    # ------------------------------------------------------------------ transition
    def step(self, action: torch.Tensor) -> torch.Tensor:
        """Drive every unfinished day to its chosen node. Returns the cost paid this step (B,)."""
        ar, b = self.ar, self.batch
        active = ~self.done
        step_cost = b.cost[ar, self.pos, action] * active

        is_cust = (action != 0) & active
        rem_a = self.remaining[ar, action]
        fail = is_cust & (self.load < rem_a - EPS)       # not enough on the truck = recourse event
        self.n_fail += fail.float()

        if self.delivery == "partial":
            delivered = torch.minimum(self.load, rem_a) * is_cust
            self.remaining[ar, action] = (rem_a - delivered).clamp_min(0.0)
            self.load = (self.load - delivered).clamp_min(0.0)
        else:
            # indivisible order: drive back to the depot, refill, come back, then deliver everything
            round_trip = b.cost[ar, action, 0] + b.cost[ar, 0, action]
            step_cost = step_cost + fail * round_trip
            load = torch.where(fail, torch.full_like(self.load, self.Q), self.load)
            self.load = torch.where(is_cust, (load - rem_a).clamp_min(0.0), self.load)
            self.remaining[ar, action] = torch.where(is_cust, torch.zeros_like(rem_a), rem_a)

        at_depot = (action == 0) & active
        self.load = torch.where(at_depot, torch.full_like(self.load, self.Q), self.load)
        self.pos = torch.where(active, action, self.pos)
        self.visited[ar, action] |= active
        self.total_cost += step_cost
        self.done = (self.pos == 0) & ~(self.remaining[:, 1:] > EPS).any(1)
        return step_cost

    # ------------------------------------------------------------------ beam search support
    def select(self, idx: torch.Tensor) -> "SVRPEnv":
        """New env whose day k is a copy of this env's day idx[k] (beam search re-uses this)."""
        new = SVRPEnv.__new__(SVRPEnv)
        new.batch = self.batch.index(idx)
        new.delivery, new.weather_input, new.Q = self.delivery, self.weather_input, self.Q
        new.ar = torch.arange(len(idx), device=idx.device)
        for name in ("pos", "load", "remaining", "visited", "total_cost", "n_fail", "done"):
            setattr(new, name, getattr(self, name)[idx].clone())
        return new
