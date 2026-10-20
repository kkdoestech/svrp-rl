# 15 — The tests, line by line (`tests/`)

| | |
|---|---|
| **Layer** | runs alongside every layer (see the table below) |
| **Files covered** | `tests/test_generator.py` (110 lines), `tests/test_env.py` (70), `tests/test_baselines.py` (84), `tests/test_models.py` (70) |
| **You build** | your own `learning/workspace/tests/` (write them yourself instead of copying) |
| **Time** | 3–5 hours in total, spread over the course |
| **Before this** | the lesson each test file belongs to |

## Why this lesson exists

The earlier lessons told you to *copy* the tests and run them. To rebuild the whole project, you should
be able to **write** them too. A test is a small program that checks another program, and writing good
tests is a skill of its own. In this project the tests caught real bugs: the history-before-truth bug
(lesson 05), and a wrong mask would have been caught too.

When to write each file:

| test file | write it after | what it protects |
|---|---|---|
| `test_generator.py` | lessons 04–05 (event tests: lesson 13) | the random world matches the math |
| `test_env.py` | lesson 08 | the game's rules and its cost accounting |
| `test_baselines.py` | lessons 07–08 (the ACO case: lesson 14) | planners, executor, and env agree with each other |
| `test_models.py` | lessons 10–11 (the GraphNet case: lesson 13) | networks, inference and training run correctly |

A good way to practise: **read only the test's name and docstring, write the test yourself, then compare.**

## Concepts from zero: pytest in 10 minutes

- **Discovery.** `pytest tests` finds every file named `test_*.py`, and in it every function named
  `test_*`, and runs them. A test **passes** if it finishes, and **fails** if any `assert` is false or any
  error is raised.
- **`assert condition`**: plain Python. `assert x == 3` fails with a report showing what `x` was.
- **Comparing floats.** Never use `==` on floats from calculations.
  - `torch.testing.assert_close(a, b, rtol=…, atol=…)`: tensors are equal up to a relative and an absolute
    tolerance.
  - `x == pytest.approx(y, abs=0.01)`: the same for plain numbers.
- **`@pytest.mark.parametrize("name", [v1, v2])`**: run the same test once per value. Stack two of them
  and you get *every combination* (2 × 2 = 4 runs).
- **Fixtures.** A function decorated with `@pytest.fixture` builds something that tests need. A test asks
  for it by having a parameter with the same name. `scope="module"` builds it once per file instead of once
  per test.
- **Built-in fixtures used here:**
  - `tmp_path`: a fresh empty temporary folder.
  - `monkeypatch`: temporarily replace a variable (here: where runs are saved), undone after the test.
- **Selecting tests:** `-k "not event"` runs only tests whose names don't contain "event". `-q` = quiet output.

### What *kinds* of checks are in this project

1. **Math vs simulation:** a formula is compared against 200,000 random samples (`weather_moments`).
2. **Invariants:** things that must *always* hold (costs symmetric, demand ≥ 1, the mask never empty).
3. **Reproducibility:** same seed → same days.
4. **Independence:** changing one setting must not change things it shouldn't (truth vs estimator).
5. **Hand-made examples:** tiny cases computed on paper (15 + 10 on a truck of 20).
6. **Cross-checks:** three independent implementations of "the cost of a plan" must agree.
7. **Random fuzzing:** a *random* legal policy must still finish, and its costs must add up.
8. **Smoke tests:** "does the training loop run end to end and save its files?"

---

## `tests/test_generator.py`

<!-- quote: tests/test_generator.py -->
```python
import torch

from svrp.config import EnvConfig
from svrp.data import World, _gen
from svrp.env import generator as G
```

Tests import the code they check, exactly like any other file. `_gen` (the seeded generator helper from
`data.py`) is imported even though it starts with `_` ("private"). Tests are allowed to reach inside.

<!-- quote: tests/test_generator.py -->
```python
def test_weather_moments_match_monte_carlo():
    g = _gen(0)
    a = torch.randn(5, 3, 3, generator=g)
    a = 0.5 * (a + a.transpose(-1, -2))
    mean, var = G.weather_moments(a)
    w = G.sample_weather((200_000,), 3, g)
    s = torch.einsum("sm,kmn,sn->ks", w, a, w)
    torch.testing.assert_close(s.mean(1), mean, atol=0.02, rtol=0.02)
    torch.testing.assert_close(s.var(1), var, atol=0.02, rtol=0.03)
```

**Math vs simulation.**

- Five random symmetric 3×3 matrices.
- The formula gives their mean and variance.
- 200,000 random weathers are drawn, and S = wᵀaw is computed for every (matrix, weather) pair with
  `einsum`. The string `"sm,kmn,sn->ks"` reads "for each sample s and matrix k, sum over m and n of
  w[s,m]·a[k,m,n]·w[s,n]".
- The simulated mean and variance must match the formula within tolerances that allow for random noise
  (the standard error with 200,000 samples is small).

If your derivation in lesson 04 had a mistake (say 1/5 instead of 4/45), this test fails.

<!-- quote: tests/test_generator.py -->
```python
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
```

**The signal ratio, on the *unclipped* parts.**

- Build one city and 50,000 weathers, and compute each customer's weather term directly.
- Check that it's centred: the mean is ≈ 0 (within 0.15, a loose bound for noise).
- Compute the three shares θ̄², Var(weather) and σ² as fractions of their total. Every customer's shares
  must be within 0.03 of 0.6 / 0.2 / 0.2.

The test uses the city's internal fields (`alpha_dem`, `shift_dem`, `sigma_dem`) and not the final
demands, because clipping at 1 distorts the shares. That's lesson 04's "side effect worth knowing".

<!-- quote: tests/test_generator.py -->
```python
def test_costs_symmetric_clipped_and_zero_diagonal():
    cfg = EnvConfig(n_customers=10)
    b = World(cfg).eval_set("val", 50)
    torch.testing.assert_close(b.cost, b.cost.transpose(1, 2))
    assert (torch.diagonal(b.cost, dim1=1, dim2=2) == 0).all()
    assert (b.cost >= 0.1 * b.dist - 1e-6).all()
    assert (b.demand[:, 1:] >= 1).all() and (b.demand[:, 0] == 0).all()
```

**Invariants of a real batch:** costs are symmetric (`cost == costᵀ`), the diagonal is exactly 0, every cost
is ≥ 10% of its distance (`- 1e-6` forgives float rounding), customers order ≥ 1, and the depot orders 0.

<!-- quote: tests/test_generator.py -->
```python
def test_full_delivery_orders_fit_in_one_truck():
    cfg = EnvConfig(n_customers=10, delivery="full", fill_rate=0.8)
    b = World(cfg).eval_set("val", 200)
    assert (b.demand <= cfg.capacity + 1e-6).all() and (b.dem_hat <= cfg.capacity + 1e-6).all()
```

With Φ = 0.8 (Q = 12.5), many orders would exceed the truck. Under **full** delivery, both the truth and
the estimate must be clipped to ≤ Q. This is an example of choosing settings that make a bug *likely to
show*.

<!-- quote: tests/test_generator.py -->
```python
def test_deterministic_toggles():
    cfg = EnvConfig(n_customers=10, stochastic_demand=False, stochastic_cost=False)
    b = World(cfg).eval_set("val", 20)
    torch.testing.assert_close(b.demand, G.clip_demand(b.theta, cfg))
    torch.testing.assert_close(b.cost, b.dist)
```

Switching randomness off must give exactly the long-run values: demand = clipped θ̄, cost = distance. This
is the starting point for the cross-check in `test_baselines.py`.

<!-- quote: tests/test_generator.py -->
```python
def test_eval_sets_are_reproducible_and_splits_differ():
    cfg = EnvConfig(n_customers=10)
    a, b = World(cfg).eval_set("test", 30), World(cfg).eval_set("test", 30)
    torch.testing.assert_close(a.demand, b.demand)
    assert not torch.allclose(a.demand, World(cfg).eval_set("val", 30).demand)
```

**Reproducibility:** two separate `World`s give identical test days. And the validation days must differ
from the test days, which catches the bug of using the same seed for both.

<!-- quote: tests/test_generator.py -->
```python
def test_truth_does_not_depend_on_estimator():
    for positions in ("fixed", "flexible"):
        truths = [World(EnvConfig(n_customers=10, positions=positions, estimate=e)).eval_set("val", 20)
                  for e in ("constant", "knn", "oracle_mean")]
        for t in truths[1:]:
            torch.testing.assert_close(t.demand, truths[0].demand)
            torch.testing.assert_close(t.cost, truths[0].cost)
```

**Independence:** for fixed *and* flexible positions, the true days must be identical whichever estimator
is used. This test was written because of a real bug: the flexible history was first drawn *before* the
truth (lesson 05). `truths[1:]` compares k-NN and oracle against the first (constant).

<!-- quote: tests/test_generator.py -->
```python
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
```

- `_mse(b)`: the mean squared error of the demand belief over customers (the depot is excluded). The leading
  `_` keeps pytest from treating it as a test.
- **With a strong weather signal** (B = 0.3, little noise), the beliefs must be ranked oracle < k-NN, and
  k-NN must be clearly better than the constant (at least 20% lower error).
- **With no weather signal** (B = 0): k-NN can't help, and is allowed to be slightly worse than the
  constant (it averages only 10 noisy past days), but not more than 20% worse. This guards against
  k-NN being broken in a way that adds error.

<!-- quote: tests/test_generator.py -->
```python
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
```

**Events (lesson 13), on a deterministic world**, so the effect can be checked exactly:

- The venue has some customers near it and some far from it (otherwise the test proves nothing).
- About half the days are event days (`0.35 < … < 0.65`, a loose bound for a random count).
- On event days, near-venue demand is exactly 2·θ̄ (θ̄ + boost 1.0·θ̄). Far customers' demand equals any
  normal day's.
- `b.demand[flag][:, near]` is boolean indexing twice: event days only, then near customers only.
  `expand(int(flag.sum()), -1)` repeats one normal day's row to compare with every event day.

<!-- quote: tests/test_generator.py -->
```python
def test_hidden_event_flag_changes_only_the_belief():
    kw = dict(n_customers=20, event_prob=0.3, event_boost=1.5)
    seen = World(EnvConfig(**kw, event_input=True)).eval_set("val", 300)
    hidden = World(EnvConfig(**kw, event_input=False)).eval_set("val", 300)
    torch.testing.assert_close(seen.demand, hidden.demand)          # same true days
    assert seen.w.shape[1] == hidden.w.shape[1] == 4 and (hidden.w[:, -1] == 0).all()
    assert _mse(seen) < 0.9 * _mse(hidden)                           # knowing about events helps k-NN
```

Visible vs hidden event flag: **the same true days**, the same input size (4 = 3 weather + 1 flag), the hidden
flag is always 0, and k-NN's error is at least 10% lower when it can see the flag.

---

## `tests/test_env.py`

<!-- quote: tests/test_env.py -->
```python
import pytest
import torch

from svrp.config import EnvConfig
from svrp.data import World
from svrp.env.svrp_env import EPS, SVRPEnv
```

`pytest` is imported for `parametrize`. `EPS` is imported so the test uses *the same* tolerance as the env.

<!-- quote: tests/test_env.py -->
```python
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
```

**A random legal policy, with checks at every step** (a *helper*, not a test: no `test_` prefix).

- Record the path (positions), starting at the depot.
- Every step:
  - the mask is never empty;
  - an empty truck may only go to the depot (`m[empty, 1:]` = the customer columns of the empty days);
  - a served customer is never allowed.
- Then choose a random **legal** move. `torch.multinomial(m.float(), 1)` treats the mask as weights, so
  every allowed node is equally likely.
- Return the path as (B, T+1).

This is *fuzzing*: a random policy explores strange situations a sensible policy never would.

<!-- quote: tests/test_env.py -->
```python
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
```

Two stacked `parametrize`s give **4 runs**: partial/full × fixed/flexible. In each:

- 64 days played to the end;
- all done, and nobody left unserved;
- **cost accounting**: rebuild the cost of every road driven from the path. `batch.cost[days, from, to]` is
  the arange indexing trick (lesson 02 section 5) with a column of day indices. Under partial delivery the
  env's total must equal it exactly. Under full delivery it can only be larger (the automatic recourse
  round trips), and it's larger **exactly** on the days with recourse events.

<!-- quote: tests/test_env.py -->
```python
def test_demand_is_revealed_only_after_a_visit():
    batch = World(EnvConfig(n_customers=10)).eval_set("val", 8)
    env = SVRPEnv(batch)
    torch.testing.assert_close(env.observed_demand()[:, 1:], batch.dem_hat[:, 1:])
    env.step(torch.full((8,), 3))
    obs = env.observed_demand()
    torch.testing.assert_close(obs[:, 3], env.remaining[:, 3])
    torch.testing.assert_close(obs[:, 4:], batch.dem_hat[:, 4:])
```

**Reveal-on-arrival:** at the start, the observed demand equals the beliefs. Send every truck to customer 3.
Now customer 3 shows its true remaining demand, while customers 4+ still show beliefs.

<!-- quote: tests/test_env.py -->
```python
def test_partial_delivery_splits_when_truck_runs_out():
    batch = World(EnvConfig(n_customers=5)).eval_set("val", 1)
    batch.demand[0, 1:] = torch.tensor([15.0, 15.0, 5.0, 5.0, 5.0])
    env = SVRPEnv(batch)
    env.step(torch.tensor([1]))          # 20 on board, deliver 15 -> 5 left
    env.step(torch.tensor([2]))          # deliver 5 of 15 -> truck empty, 10 still open, recourse event
    assert env.load.item() == 0 and env.remaining[0, 2].item() == 10 and env.n_fail.item() == 1
    assert env.mask()[0].tolist() == [True] + [False] * 5   # only the depot is allowed now
```

**A hand-made split example.** The test overwrites one day's true demand with chosen numbers, then plays
two moves and checks the exact state: load 0, 10 units still open at customer 2, one recourse event, and
only the depot allowed. This is the env's partial-delivery rule, checked against arithmetic you can do in
your head.

<!-- quote: tests/test_env.py -->
```python
def test_select_copies_state_independently():
    batch = World(EnvConfig(n_customers=6)).eval_set("val", 3)
    env = SVRPEnv(batch)
    env.step(torch.tensor([1, 2, 3]))
    sub = env.select(torch.tensor([2, 2]))
    assert sub.pos.tolist() == [3, 3]
    sub.step(torch.tensor([4, 5]))
    assert env.pos.tolist() == [1, 2, 3]
```

`select` copies must be **independent**: stepping the copies must not move the original trucks.

---

## `tests/test_baselines.py`

<!-- quote: tests/test_baselines.py -->
```python
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
```

A dict of three planners behind one interface. The `lambda`s fix the extra arguments: a short OR-Tools
time limit and few ACO iterations, so the tests stay fast. (Until lesson 14, delete the `aco` import and
the `"aco"` entry.)

<!-- quote: tests/test_baselines.py -->
```python
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
```

- `_one(cfg)`: one validation day, and its true demand and cost as NumPy arrays.
- For **each planner** (3 runs, via `parametrize` over the dict's keys), with Φ = 0.25 (≈ 4 customers per
  trip, so the routing is non-trivial):
  - every customer appears **exactly once** (sorting the flattened routes must give 1…15);
  - every route's planned load is ≤ Q.

<!-- quote: tests/test_baselines.py -->
```python
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
```

**The cross-check: three independent calculations must agree.**

- `replay_in_env` drives a plan through the *environment*, step by step. It skips extra depot entries
  (the env forbids depot → depot).
- In a world with **no randomness** and constant estimates (so the beliefs are exactly the truth), the
  plan is driven exactly as planned. So:
  1. the executor sees no recourse (`fails == 0`);
  2. its cost equals OR-Tools' own objective (up to the ×1000 integer rounding, `abs=0.01`);
  3. it equals `plan_cost`;
  4. it equals the env replay.

If any one of the three implementations had a different idea of the rules, this test would catch it.

<!-- quote: tests/test_baselines.py -->
```python
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
```

- The recourse arithmetic from lesson 07's hand example, under both delivery rules. The comment spells out
  the arithmetic, so a failure is easy to understand.
- `nn_online` finishes on 32 days.

<!-- quote: tests/test_baselines.py -->
```python
@pytest.mark.parametrize("name", ["cw", "ortools_gls", "hindsight"])
def test_runner_scores_all_days(name):
    b = World(EnvConfig(n_customers=8)).eval_set("val", 6)
    kw = {"time_limit": 0.2} if name != "cw" else {}
    out = run_planner(name, b, "partial", workers=2, **kw)
    assert out["cost"].shape == (6,) and np.isfinite(out["cost"]).all()
```

The runner scores every day for CW, OR-Tools and hindsight, with 2 worker processes and a short time limit:
the output has one finite cost per day. This also checks that the multiprocessing path works on your system.

---

## `tests/test_models.py`

<!-- quote: tests/test_models.py -->
```python
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
```

A **module-scoped fixture**: one config, 16 validation days and a small PaperNet (D = 32 instead of 128, to
be fast) are built once and shared by the tests that ask for `setup`. `torch.manual_seed(0)` makes the
network's random start repeatable.

<!-- quote: tests/test_models.py -->
```python
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
```

- **Gradients flow:** a sampled rollout gives finite costs and log-probs. A REINFORCE-style loss can
  `backward()`, and *every* parameter receives a gradient. A parameter with `grad is None` would be one the
  network never uses: a wiring bug.
- **Beam width 1 = greedy**, both outputs, to the last digit. The strongest single check of the beam search
  bookkeeping.

<!-- quote: tests/test_models.py -->
```python
def test_beam_and_best_of_n_are_valid(setup):
    _, batch, policy = setup
    most_probable, cheapest = beam_search(policy, batch, 3)
    assert (cheapest <= most_probable + 1e-6).all()
    assert torch.isfinite(best_of_n(policy, batch, 4)).all()


def test_apriori_plans_execute(setup):
    _, batch, policy = setup
    cost, fails = apriori(policy, batch)
    assert cost.shape == (16,) and (cost > 0).all()
```

Beam search with W = 3: the cheapest beam can't be more expensive than the most probable one.
Best-of-4 gives finite costs. A-priori plans execute, and give one positive cost per day.

<!-- quote: tests/test_models.py -->
```python
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
```

**Size-agnostic GraphNet:** *the same weights* run on N = 6 and N = 12 (flexible positions). Sampling gives
finite log-probs, and beam1 = greedy again. The import is inside the test, so the rest of this file
works before `graph_net.py` exists (deselect it with `-k "not graphnet"` until lesson 13).

<!-- quote: tests/test_models.py -->
```python
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
```

**A smoke test of the whole training loop**, for both baselines:

- `monkeypatch.setattr(reinforce, "RUNS", tmp_path)` redirects the runs folder into a temporary directory,
  so the test never litters `runs/`.
- A tiny config: 6 customers, batch 32, 2 epochs of 64 days, D = 16, on the CPU. It takes seconds.
- The checks:
  - `best.pt` exists;
  - `log.csv` has exactly 3 lines (header + 2 epochs);
  - `load_policy` returns *equal* configs. Dataclasses compare field by field, so this proves the JSON
    round-trip loses nothing.

## Rebuild it yourself

For each file: read the test names in this lesson, write the tests yourself, and compare. Then run:

```bash
uv run python -m pytest tests -q
```

With the whole course done: **34 passed** (11 generator + 7 env + 9 baselines + 7 models).

## Common mistakes

- `==` on floats: use `assert_close` / `approx`.
- Tests that depend on each other's order or state: each test must build what it needs (fixtures help).
- Tolerances so loose that a bug still passes (e.g. a share check with ±0.3): pick bounds that a
  *wrong* implementation would violate.
- Testing only the "happy path": the fuzzing test and the full-delivery runs exist because odd cases are
  where bugs live.

## Check your understanding

1. Why does `test_signal_ratio_matches_target` use the city's internal weights instead of `World.eval_set` demands?
2. How many test runs do the two stacked `parametrize` decorators on `test_random_policy_finishes_and_cost_adds_up` create?
3. What bug would `test_eval_sets_are_reproducible_and_splits_differ` catch that a test with only the first assert would miss?
4. Why does the cross-check use a world with *no randomness*?

**Answers** (try the questions first, then check here)

1. Clipping at 1 changes the variance of the final demands, so the shares would be off for a correct
   generator. The internal weights test the math before clipping.
2. 2 × 2 = **4**.
3. Using the same seed for validation and test (then you'd be choosing settings on the test days).
4. Only then is "driven exactly as planned" true, so all calculations must agree exactly, and any
   difference is a bug, not randomness.

## Read more

- pytest documentation, "Get started" and "How to parametrize": [docs.pytest.org](https://docs.pytest.org/)
- `torch.testing.assert_close` in the PyTorch docs.

**Next:** [16 — Helper scripts](16_helper_scripts.md)
