import pytest
import torch

from svrp.config import EnvConfig, TrainConfig
from svrp.data import World
from svrp.inference import apriori, beam_search, best_of_n, rollout
from svrp.models.paper_net import PaperNet
from svrp.train import reinforce


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = EnvConfig(n_customers=8)
    return cfg, World(cfg).eval_set("val", 16), PaperNet(9, 3, 32)


def test_sampled_rollout_has_gradients(setup):
    cfg, batch, policy = setup
    r = rollout(policy, batch, decode="sample")
    assert torch.isfinite(r.cost).all() and torch.isfinite(r.logp).all()
    (r.cost.detach() * r.logp).mean().backward()
    assert all(p.grad is not None for p in policy.parameters())


def test_beam_width_one_equals_greedy(setup):
    _, batch, policy = setup
    with torch.no_grad():
        greedy = rollout(policy, batch).cost
    most_probable, cheapest = beam_search(policy, batch, 1)
    torch.testing.assert_close(most_probable, greedy)
    torch.testing.assert_close(cheapest, greedy)


def test_beam_and_best_of_n_are_valid(setup):
    _, batch, policy = setup
    most_probable, cheapest = beam_search(policy, batch, 3)
    assert (cheapest <= most_probable + 1e-6).all()
    assert torch.isfinite(best_of_n(policy, batch, 4)).all()


def test_apriori_plans_execute(setup):
    _, batch, policy = setup
    cost, fails = apriori(policy, batch)
    assert cost.shape == (16,) and (cost > 0).all()


def test_graphnet_runs_on_any_size_and_beam1_is_greedy():
    from svrp.models.graph_net import GraphNet
    torch.manual_seed(0)
    policy = GraphNet(3, 32, heads=4, layers=2)
    for n in (6, 12):                                   # same weights, different N
        batch = World(EnvConfig(n_customers=n, positions="flexible")).eval_set("val", 8)
        r = rollout(policy, batch, decode="sample")
        assert torch.isfinite(r.logp).all()
        with torch.no_grad():
            greedy = rollout(policy, batch).cost
        torch.testing.assert_close(beam_search(policy, batch, 1)[0], greedy)


@pytest.mark.parametrize("baseline", ["critic", "shared"])
def test_training_loop_runs_and_saves(tmp_path, monkeypatch, baseline):
    monkeypatch.setattr(reinforce, "RUNS", tmp_path)
    env = EnvConfig(n_customers=6)
    tr = TrainConfig(baseline=baseline, n_shared=4, batch_size=32, epochs=2, epoch_size=64,
                     val_size=32, embed_dim=16, device="cpu", run_name="t")
    run = reinforce.train(env, tr, log=lambda *_: None)
    assert (run / "best.pt").exists() and (run / "log.csv").read_text().count("\n") == 3
    policy, env2, tr2, ckpt = reinforce.load_policy(run)
    assert env2 == env and tr2 == tr
