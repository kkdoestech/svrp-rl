"""Run a classical planner on every day of a Batch and score its plans on the TRUE days.

    plan  = planner(estimated demand, estimated cost)     # what the dispatcher can know in the morning
    score = execute(plan, true demand, true cost)         # what actually happens on the road

"hindsight" plans on the TRUE values instead: not deployable, only a reference point.
Days are solved in parallel processes (each OR-Tools solve is single-threaded).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from svrp.baselines.aco import aco
from svrp.baselines.clarke_wright import clarke_wright
from svrp.baselines.ortools_solver import default_time_limit, ortools_cvrp
from svrp.data import Batch
from svrp.env.executor import execute_many

PLANNERS = ("cw", "ortools_gls", "ortools_tabu", "aco", "hindsight")


def solve(name: str, dem: np.ndarray, cost: np.ndarray, capacity: float, kw: dict):
    t0 = time.perf_counter()
    if name == "cw":
        plan = clarke_wright(dem, cost, capacity)
    elif name in ("ortools_gls", "hindsight"):
        plan = ortools_cvrp(dem, cost, capacity, metaheuristic="gls", **kw)
    elif name == "ortools_tabu":
        plan = ortools_cvrp(dem, cost, capacity, metaheuristic="tabu", **kw)
    elif name == "aco":
        plan = aco(dem, cost, capacity, **kw)
    else:
        raise ValueError(name)
    return plan, time.perf_counter() - t0


def _solve_star(args):
    return solve(*args)


def run_planner(name: str, batch: Batch, delivery: str, workers: int | None = None, **kw):
    """Returns dict(cost=(B,), n_fail=(B,), sec_per_instance=float)."""
    dem_true = batch.demand.cpu().double().numpy()
    cost_true = batch.cost.cpu().double().numpy()
    if name == "hindsight":
        dem_plan, cost_plan = dem_true, cost_true
    else:
        dem_plan = batch.dem_hat.cpu().double().numpy()
        cost_plan = batch.cost_hat.cpu().double().numpy()
    if name.startswith("ortools") or name == "hindsight":
        kw.setdefault("time_limit", default_time_limit(batch.n_nodes - 1))
    Q = batch.capacity

    same_belief = np.allclose(dem_plan, dem_plan[:1]) and np.allclose(cost_plan, cost_plan[:1])
    if same_belief:     # e.g. constant estimates on a fixed city: one plan serves every day
        plan, sec = solve(name, dem_plan[0], cost_plan[0], Q, kw)
        plans, secs = plan, [sec]
    else:
        jobs = [(name, dem_plan[b], cost_plan[b], Q, kw) for b in range(batch.size)]
        workers = workers or max(1, (os.cpu_count() or 2) // 2)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(_solve_star, jobs, chunksize=max(1, len(jobs) // (4 * workers))))
        plans = [p for p, _ in out]
        secs = [s for _, s in out]
    cost, fails = execute_many(plans, dem_true, cost_true, Q, delivery)
    return {"cost": cost, "n_fail": fails, "sec_per_instance": float(np.mean(secs))}
