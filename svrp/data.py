"""Turns the generator + estimates into ready-to-use batches of days ("scenarios").

A Batch holds B independent days. For each day we store both
  * the TRUTH (demand, cost), which only the environment's scoring sees, and
  * the BELIEF (dem_hat, cost_hat, w), which is what policies and planners are allowed to use.

Seeds are fixed per role, so the validation and test days are identical for every method and every run
(that makes every comparison *paired*).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch

from svrp.config import EnvConfig
from svrp.env import generator as G
from svrp.env.estimates import knn_estimate

SEED_CITY, SEED_HIST, SEED_VAL, SEED_TEST, SEED_TRAIN = 10_000, 20_000, 30_000, 40_000, 50_000
EVENT_SCALE = 2.0   # event flag value: k-NN then treats event days as far away from normal days


@dataclass
class Batch:
    coords: torch.Tensor     # (B, N+1, 2)
    w: torch.Tensor          # (B, M)        today's weather (+ event flag): the observable external variables
    theta: torch.Tensor      # (B, N+1)      long-run mean demand
    dist: torch.Tensor       # (B, N+1, N+1) distances = long-run mean cost
    demand: torch.Tensor     # (B, N+1)      TRUE demand (hidden until a visit)
    cost: torch.Tensor       # (B, N+1, N+1) TRUE travel cost (charged when driven)
    dem_hat: torch.Tensor    # (B, N+1)      estimate of demand
    cost_hat: torch.Tensor   # (B, N+1, N+1) estimate of cost
    capacity: float

    @property
    def size(self) -> int:
        return self.demand.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.demand.shape[1]

    def _map(self, fn) -> "Batch":
        return replace(self, **{f.name: fn(getattr(self, f.name)) for f in fields(self) if f.name != "capacity"})

    def to(self, device) -> "Batch":
        return self._map(lambda t: t.to(device))

    def index(self, idx) -> "Batch":
        return self._map(lambda t: t[idx])

    def repeat_interleave(self, m: int) -> "Batch":
        return self._map(lambda t: t.repeat_interleave(m, dim=0))

    def believed(self) -> "Batch":
        """The world as the planner imagines it: truth := estimates. Used for a-priori planning."""
        return replace(self, demand=self.dem_hat.clone(), cost=self.cost_hat.clone())


class World:
    """One experimental world (an EnvConfig). Holds the fixed city and its history bank if positions are fixed."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        if cfg.positions == "fixed":
            self.city = G.make_cities(cfg, 1, _gen(SEED_CITY + cfg.city_seed))
            self.hist = self._history(self.city, cfg.history_days, _gen(SEED_HIST + cfg.city_seed))
        elif cfg.positions != "flexible":
            raise ValueError(cfg.positions)

    # --- history bank: past days the k-NN estimator can look up -------------------------------------
    def _history(self, city: G.City, days: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
        C = city.n_cities
        w = G.sample_weather((C, days), self.cfg.n_weather, gen)
        event = G.sample_events((C, days), self.cfg, gen)
        real = G.realize(city, w, self.cfg, gen, event)
        _, key = self._external(w, event)
        return {"key": key, "demand": real["demand"], "cost": real["cost"].flatten(-2)}

    def _external(self, w: torch.Tensor, event: torch.Tensor | None):
        """(what the agent observes, what the k-NN lookup matches on). Without events both are just w.
        A hidden event flag is observed as 0 and ignored by the k-NN lookup."""
        if event is None:
            return w, w
        flag = EVENT_SCALE * event[..., None]
        if self.cfg.event_input:
            ext = torch.cat([w, flag], -1)
            return ext, ext
        return torch.cat([w, torch.zeros_like(flag)], -1), w

    # --- sampling days -------------------------------------------------------------------------------
    def sample(self, n: int, gen: torch.Generator) -> Batch:
        cfg = self.cfg
        if cfg.positions == "fixed":
            city = self.city
            w = G.sample_weather((1, n), cfg.n_weather, gen)            # one city, n days
        else:
            city = G.make_cities(cfg, n, gen)                           # n cities, one day each
            w = G.sample_weather((n, 1), cfg.n_weather, gen)
        event = G.sample_events(w.shape[:2], cfg, gen)
        real = G.realize(city, w, cfg, gen, event)
        obs, key = self._external(w, event)
        # drawn *after* the truth, so the true days are identical whichever estimate we use
        if cfg.positions == "fixed":
            hist = self.hist
        elif cfg.estimate == "knn":
            hist = self._history(city, cfg.history_days_flexible, gen)
        flat = lambda t: t.reshape(n, *t.shape[2:])                    # (C', H, ...) -> (n, ...)

        N1 = cfg.n_customers + 1
        if cfg.estimate == "constant":
            dem_hat = G.clip_demand(city.theta.expand(n, N1), cfg)
            cost_hat = city.dist.expand(n, N1, N1).clone()
        elif cfg.estimate == "oracle_mean":
            dem_hat, cost_hat = flat(real["mean_dem"]), flat(real["mean_cost"])
        elif cfg.estimate == "knn":
            dem_hat = flat(knn_estimate(key, hist["key"], hist["demand"], cfg.knn_k))
            cost_hat = flat(knn_estimate(key, hist["key"], hist["cost"], cfg.knn_k)).view(n, N1, N1)
            dem_hat = G.clip_demand(dem_hat, cfg)
        else:
            raise ValueError(cfg.estimate)

        expand = lambda t: t.expand(n, *t.shape[1:]).clone() if t.shape[0] == 1 else t
        return Batch(
            coords=expand(city.coords), w=flat(obs), theta=expand(city.theta), dist=expand(city.dist),
            demand=flat(real["demand"]), cost=flat(real["cost"]),
            dem_hat=dem_hat, cost_hat=cost_hat, capacity=cfg.capacity,
        )

    def eval_set(self, split: str, n: int = 1000) -> Batch:
        seed = {"val": SEED_VAL, "test": SEED_TEST}[split] + self.cfg.city_seed
        return self.sample(n, _gen(seed))


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def train_generator(seed: int) -> torch.Generator:
    return _gen(SEED_TRAIN + seed)
