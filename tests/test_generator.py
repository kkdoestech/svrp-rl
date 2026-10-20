import torch

from svrp.config import EnvConfig
from svrp.data import World, _gen
from svrp.env import generator as G


def test_weather_moments_match_monte_carlo():
    g = _gen(0)
    a = torch.randn(5, 3, 3, generator=g)
    a = 0.5 * (a + a.transpose(-1, -2))
    mean, var = G.weather_moments(a)
    w = G.sample_weather((200_000,), 3, g)
    s = torch.einsum("sm,kmn,sn->ks", w, a, w)
    torch.testing.assert_close(s.mean(1), mean, atol=0.02, rtol=0.02)
    torch.testing.assert_close(s.var(1), var, atol=0.02, rtol=0.03)


def test_signal_ratio_matches_target():
    """The A / B / Γ split of b.md: shares of base², weather variance and noise variance."""
    cfg = EnvConfig(n_customers=20, a_ratio=0.6, b_ratio=0.2, gamma_ratio=0.2)
    city = G.make_cities(cfg, 1, _gen(1))
    w = G.sample_weather((1, 50_000), 3, _gen(2))
    ww = (w[..., :, None] * w[..., None, :]).flatten(-2)
    weather = ww @ city.alpha_dem.transpose(-1, -2) - city.shift_dem[:, None]   # (1, S, N+1)
    theta2 = city.theta[0, 1:] ** 2
    total = theta2 + weather[0, :, 1:].var(0) + city.sigma_dem[0, 1:] ** 2
    assert abs(weather[0, :, 1:].mean(0)).max() < 0.15           # centred: E[weather term] ≈ 0
    for share, target in [(theta2 / total, 0.6), (weather[0, :, 1:].var(0) / total, 0.2),
                          (city.sigma_dem[0, 1:] ** 2 / total, 0.2)]:
        assert (share - target).abs().max() < 0.03


def test_costs_symmetric_clipped_and_zero_diagonal():
    cfg = EnvConfig(n_customers=10)
    b = World(cfg).eval_set("val", 50)
    torch.testing.assert_close(b.cost, b.cost.transpose(1, 2))
    assert (torch.diagonal(b.cost, dim1=1, dim2=2) == 0).all()
    assert (b.cost >= 0.1 * b.dist - 1e-6).all()
    assert (b.demand[:, 1:] >= 1).all() and (b.demand[:, 0] == 0).all()


def test_full_delivery_orders_fit_in_one_truck():
    cfg = EnvConfig(n_customers=10, delivery="full", fill_rate=0.8)
    b = World(cfg).eval_set("val", 200)
    assert (b.demand <= cfg.capacity + 1e-6).all() and (b.dem_hat <= cfg.capacity + 1e-6).all()


def test_deterministic_toggles():
    cfg = EnvConfig(n_customers=10, stochastic_demand=False, stochastic_cost=False)
    b = World(cfg).eval_set("val", 20)
    torch.testing.assert_close(b.demand, G.clip_demand(b.theta, cfg))
    torch.testing.assert_close(b.cost, b.dist)


def test_eval_sets_are_reproducible_and_splits_differ():
    cfg = EnvConfig(n_customers=10)
    a, b = World(cfg).eval_set("test", 30), World(cfg).eval_set("test", 30)
    torch.testing.assert_close(a.demand, b.demand)
    assert not torch.allclose(a.demand, World(cfg).eval_set("val", 30).demand)


def test_truth_does_not_depend_on_estimator():
    for positions in ("fixed", "flexible"):
        truths = [World(EnvConfig(n_customers=10, positions=positions, estimate=e)).eval_set("val", 20)
                  for e in ("constant", "knn", "oracle_mean")]
        for t in truths[1:]:
            torch.testing.assert_close(t.demand, truths[0].demand)
            torch.testing.assert_close(t.cost, truths[0].cost)


def _mse(b):
    return ((b.dem_hat[:, 1:] - b.demand[:, 1:]) ** 2).mean().item()


def test_knn_beats_constant_when_weather_matters():
    kw = dict(n_customers=20, a_ratio=0.6, b_ratio=0.3, gamma_ratio=0.1)
    const = World(EnvConfig(**kw, estimate="constant")).eval_set("val", 500)
    knn = World(EnvConfig(**kw, estimate="knn")).eval_set("val", 500)
    oracle = World(EnvConfig(**kw, estimate="oracle_mean")).eval_set("val", 500)
    assert _mse(oracle) < _mse(knn) < 0.8 * _mse(const)


def test_knn_roughly_ties_constant_without_weather_signal():
    kw = dict(n_customers=20, a_ratio=0.8, b_ratio=0.0, gamma_ratio=0.2)
    const = World(EnvConfig(**kw, estimate="constant")).eval_set("val", 500)
    knn = World(EnvConfig(**kw, estimate="knn")).eval_set("val", 500)
    assert _mse(knn) < 1.2 * _mse(const)


def test_events_boost_demand_near_the_venue_only():
    cfg = EnvConfig(n_customers=30, event_prob=0.5, event_boost=1.0, stochastic_demand=False)
    world = World(cfg)
    b = world.eval_set("val", 400)
    near = world.city.near_venue[0].bool()
    assert near.any() and (~near[1:]).any()
    flag = b.w[:, -1] > 0
    assert 0.35 < flag.float().mean() < 0.65
    # deterministic demand: event days add exactly θ̄ near the venue, nothing elsewhere
    torch.testing.assert_close(b.demand[flag][:, near], 2 * b.theta[flag][:, near])
    torch.testing.assert_close(b.demand[flag][:, ~near], b.demand[~flag][:1, ~near].expand(int(flag.sum()), -1))


def test_hidden_event_flag_changes_only_the_belief():
    kw = dict(n_customers=20, event_prob=0.3, event_boost=1.5)
    seen = World(EnvConfig(**kw, event_input=True)).eval_set("val", 300)
    hidden = World(EnvConfig(**kw, event_input=False)).eval_set("val", 300)
    torch.testing.assert_close(seen.demand, hidden.demand)          # same true days
    assert seen.w.shape[1] == hidden.w.shape[1] == 4 and (hidden.w[:, -1] == 0).all()
    assert _mse(seen) < 0.9 * _mse(hidden)                           # knowing about events helps k-NN
