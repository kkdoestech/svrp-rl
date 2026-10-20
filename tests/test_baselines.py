import numpy as np
import pytest
import torch

from svrp.baselines.aco import aco
from svrp.baselines.clarke_wright import clarke_wright
from svrp.baselines.common import plan_cost, plan_to_routes, planning_demand
from svrp.baselines.nn_online import nn_online
from svrp.baselines.ortools_solver import ortools_cvrp
from svrp.baselines.runner import run_planner
from svrp.config import EnvConfig
from svrp.data import World
from svrp.env.executor import execute
from svrp.env.svrp_env import SVRPEnv

PLANNERS = {
    "cw": lambda d, c, q: clarke_wright(d, c, q),
    "ortools": lambda d, c, q: ortools_cvrp(d, c, q, time_limit=0.3),
    "aco": lambda d, c, q: aco(d, c, q, iters=15),
}


def _one(cfg, n=1):
    b = World(cfg).eval_set("val", n)
    return b, b.demand[0].double().numpy(), b.cost[0].double().numpy()


@pytest.mark.parametrize("name", list(PLANNERS))
def test_plans_cover_every_customer_once_within_capacity(name):
    cfg = EnvConfig(n_customers=15, fill_rate=0.25)
    b, _, _ = _one(cfg)
    dem, cost = b.dem_hat[0].double().numpy(), b.cost_hat[0].double().numpy()
    plan = PLANNERS[name](dem, cost, cfg.capacity)
    routes = plan_to_routes(plan)
    assert sorted(c for r in routes for c in r) == list(range(1, 16))
    d = planning_demand(dem, cfg.capacity)
    assert all(d[r].sum() <= cfg.capacity + 1e-6 for r in routes)


def replay_in_env(plan, batch, delivery="partial"):
    env = SVRPEnv(batch, delivery)
    for node in plan[1:]:
        if node == 0 and env.pos.item() == 0:
            continue
        env.step(torch.tensor([node]))
    assert env.done.all()
    return env.total_cost.item()


def test_deterministic_world_executor_equals_ortools_objective_and_env():
    """With no randomness the plan is driven exactly as planned: three independent cost computations agree."""
    cfg = EnvConfig(n_customers=12, stochastic_demand=False, stochastic_cost=False,
                    estimate="constant", fill_rate=0.3)
    b, dem, cost = _one(cfg)
    plan, obj = ortools_cvrp(dem, cost, cfg.capacity, time_limit=0.3, return_objective=True)
    exe, fails = execute(plan, dem, cost, cfg.capacity)
    assert fails == 0
    assert exe == pytest.approx(obj, abs=0.01)
    assert exe == pytest.approx(plan_cost(plan, cost), abs=1e-6)
    assert replay_in_env(plan, b) == pytest.approx(exe, abs=1e-4)


def test_executor_recourse_partial_and_full():
    cost = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    dem = np.array([0, 15, 10], dtype=float)
    # plan believes both fit (Q = 20) but 15 + 10 = 25: at customer 2 the truck has 5 left
    tot, fails = execute([0, 1, 2, 0], dem, cost, 20.0, "partial")
    assert fails == 1 and tot == pytest.approx(1 + 1 + 2 + 1)    # 0->1->2, round trip 2->0->2, 2->0
    tot, fails = execute([0, 1, 2, 0], dem, cost, 20.0, "full")
    assert fails == 1 and tot == pytest.approx(1 + 1 + 2 + 1)


def test_nn_online_finishes():
    b = World(EnvConfig(n_customers=10)).eval_set("val", 32)
    env = nn_online(b)
    assert env.done.all()


@pytest.mark.parametrize("name", ["cw", "ortools_gls", "hindsight"])
def test_runner_scores_all_days(name):
    b = World(EnvConfig(n_customers=8)).eval_set("val", 6)
    kw = {"time_limit": 0.2} if name != "cw" else {}
    out = run_planner(name, b, "partial", workers=2, **kw)
    assert out["cost"].shape == (6,) and np.isfinite(out["cost"]).all()
