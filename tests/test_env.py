import pytest
import torch

from svrp.config import EnvConfig
from svrp.data import World
from svrp.env.svrp_env import EPS, SVRPEnv


def random_rollout(env: SVRPEnv, gen: torch.Generator):
    path = [env.pos.clone()]
    for _ in range(env.max_steps()):
        if env.done.all():
            break
        m = env.mask()
        assert m.any(1).all(), "mask must never be empty"
        # an empty truck may only go to the depot; served customers are never allowed
        empty = env.load <= EPS
        assert not m[empty, 1:].any()
        assert not (m[:, 1:] & (env.remaining[:, 1:] <= EPS)).any()
        a = torch.multinomial(m.float(), 1, generator=gen).squeeze(1)
        env.step(a)
        path.append(env.pos.clone())
    return torch.stack(path, 1)


@pytest.mark.parametrize("delivery", ["partial", "full"])
@pytest.mark.parametrize("positions", ["fixed", "flexible"])
def test_random_policy_finishes_and_cost_adds_up(delivery, positions):
    cfg = EnvConfig(n_customers=10, delivery=delivery, positions=positions)
    batch = World(cfg).eval_set("val", 64)
    env = SVRPEnv(batch, delivery)
    path = random_rollout(env, torch.Generator().manual_seed(0))
    assert env.done.all()
    assert (env.remaining[:, 1:] <= EPS).all()
    arcs = batch.cost[torch.arange(64)[:, None], path[:, :-1], path[:, 1:]].sum(1)
    if delivery == "partial":
        torch.testing.assert_close(env.total_cost, arcs)        # cost = sum of realized arcs driven
    else:
        assert (env.total_cost >= arcs - 1e-5).all()           # + automatic recourse round trips
        assert ((env.total_cost > arcs + 1e-5) == (env.n_fail > 0)).all()


def test_demand_is_revealed_only_after_a_visit():
    batch = World(EnvConfig(n_customers=10)).eval_set("val", 8)
    env = SVRPEnv(batch)
    torch.testing.assert_close(env.observed_demand()[:, 1:], batch.dem_hat[:, 1:])
    env.step(torch.full((8,), 3))
    obs = env.observed_demand()
    torch.testing.assert_close(obs[:, 3], env.remaining[:, 3])
    torch.testing.assert_close(obs[:, 4:], batch.dem_hat[:, 4:])


def test_partial_delivery_splits_when_truck_runs_out():
    batch = World(EnvConfig(n_customers=5)).eval_set("val", 1)
    batch.demand[0, 1:] = torch.tensor([15.0, 15.0, 5.0, 5.0, 5.0])
    env = SVRPEnv(batch)
    env.step(torch.tensor([1]))          # 20 on board, deliver 15 -> 5 left
    env.step(torch.tensor([2]))          # deliver 5 of 15 -> truck empty, 10 still open, recourse event
    assert env.load.item() == 0 and env.remaining[0, 2].item() == 10 and env.n_fail.item() == 1
    assert env.mask()[0].tolist() == [True] + [False] * 5   # only the depot is allowed now


def test_select_copies_state_independently():
    batch = World(EnvConfig(n_customers=6)).eval_set("val", 3)
    env = SVRPEnv(batch)
    env.step(torch.tensor([1, 2, 3]))
    sub = env.select(torch.tensor([2, 2]))
    assert sub.pos.tolist() == [3, 3]
    sub.step(torch.tensor([4, 5]))
    assert env.pos.tolist() == [1, 2, 3]
