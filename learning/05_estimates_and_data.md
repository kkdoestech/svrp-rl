# 05 — Truth vs belief: estimates and batches (`svrp/env/estimates.py`, `svrp/data.py`)

| | |
|---|---|
| **Layer** | 3 |
| **Files covered** | `svrp/env/estimates.py` (36 lines), `svrp/data.py` (143 lines), `tests/test_generator.py` |
| **You build** | `mysvrp/env/estimates.py`, `mysvrp/data.py` |
| **Time** | 3–4 hours |
| **Before this** | [04 — Generator](04_generator.md); `part 6.md` ("Variable estimates") |

## Why this layer exists

The generator makes the *truth*: what each customer actually orders today. But a dispatcher in the
morning doesn't know the truth. They only have:

- today's **weather forecast** (free to look up, as the paper stresses),
- the **history**: what happened on past days,
- and the long-run **averages**.

So for every day we need two things side by side: the **truth** (used *only* to score what happened)
and the **belief** (the only thing planners and the AI are allowed to look at). Mixing the two up is the
most common way to fool yourself in this field: the AI "cheats" by seeing tomorrow, and the results
look amazing but mean nothing. This layer keeps them strictly apart, in one object called a **Batch**.

The paper's estimators (`part 6.md`):

- **Constant:** "Customer A always orders 10 boxes."
- **k-NN:** "Today is hot and humid. On the 10 past days most like today, customer A ordered ~15."

We add a third, **oracle_mean**: the exact noise-free average given the weather. No real dispatcher has
it. It tells us how much weather could help *at best*.

## Concepts from zero

### k-nearest neighbours (k-NN), with numbers

Today's weather: w = (0.1, 0, 0). The history has three days:

| past day | weather | customer i ordered | distance ‖w_g − w‖ |
|---|---|---|---|
| A | (0.2, 0, 0) | 12 | 0.1 |
| B | (0.0, 0, 0) | 10 | 0.1 |
| C | (0.9, 0, 0) | 20 | 0.8 |

With **k = 2**, take the 2 closest days (A and B). Their weights are 1/distance: 1/0.1 = 10 each. The
estimate is (10·12 + 10·10)/(10 + 10) = **11**.

With **k = 3**: weights 10, 10, 1.25 (= 1/0.8), so (10·12 + 10·10 + 1.25·20)/21.25 = 245/21.25 = **11.53**.
Day C is far from today, so it barely counts. That is **inverse-distance weighting**, the paper's formula:

```
d̂_i = Σ_g (ξ_ig / ‖w_g − w‖) / Σ_g (1 / ‖w_g − w‖)
```

Why does k-NN beat the constant? Because the weather effect is a real, *fixed* pattern (lesson 04). Days
with similar weather have similar demand, apart from the noise. And averaging k = 10 days also averages
out much of their noise.

### Paired comparison and fixed seeds

Imagine method X scores 8.0 on 1,000 random days and method Y scores 7.9 on 1,000 *other* random days.
Is Y better, or were its days just easier? You can't tell.

Now give **both methods the same 1,000 days**. Then each day gives a direct comparison (X's cost minus
Y's cost on the *same* day), and the luck of the days cancels out. That is a **paired comparison**. It is
far more sensitive, and it's why every method in `LOG.md` is scored on identical days.

To get identical days, every *role* gets its own fixed random seed:

| role | seed | used for |
|---|---|---|
| city | 10,000 + city_seed | positions, θ̄, α of the fixed city |
| history | 20,000 + city_seed | the past days k-NN looks up |
| validation | 30,000 + city_seed | 1,000 days to choose checkpoints and settings |
| test | 40,000 + city_seed | 1,000 days touched only for final results |
| training | 50,000 + train seed | a never-ending stream of fresh days |

Validation and test are separate on purpose. If you pick settings by looking at the test days, your test
score becomes optimistic, because you've tuned to those exact days. (This is called *overfitting to the
test set*.)

## Code walkthrough

### `svrp/env/estimates.py`, lines 1–12: the header

<!-- quote: svrp/env/estimates.py -->
```python
"""What the agent *believes* about today's demands and travel costs before it visits anybody.

constant     d̂ = θ̄,          ĉ = distance          (long-run averages; ignores the weather)
knn          inverse-distance-weighted mean of the k past days whose weather was most similar to today
oracle_mean  d̂ = E[ξ | w],   ĉ = E[c | w]          (perfect knowledge of the weather effect, no noise;
                                                     a ceiling for how much weather can possibly help)

The k-NN formula from the paper:  d̂_i = Σ_g (ξ_ig / ‖w_g − w‖) / Σ_g (1 / ‖w_g − w‖)
"""
from __future__ import annotations

import torch
```

The docstring is a small table of the three beliefs, plus the paper's k-NN formula. Writing the formula
in the docstring means anyone can check the code against the paper. The file imports only `torch`:
`knn_estimate` is pure tensor math and knows nothing about cities or configs. That makes it easy to test
on its own (the k = 2 / k = 3 example above).

### `svrp/env/estimates.py`, lines 15–36: k-NN for many days at once

<!-- quote: svrp/env/estimates.py -->
```python
def knn_estimate(
    w_query: torch.Tensor,        # (C', S, M)  today's weather, S days per city
    w_hist: torch.Tensor,         # (C, H, M)   weather of past days (C = C' or 1)
    values_hist: torch.Tensor,    # (C, H, D)   what was observed on those days (demands or flattened costs)
    k: int,
    chunk: int = 256,
) -> torch.Tensor:
    """Returns (C', S, D)."""
    Cq, S, _ = w_query.shape
    C, H, D = values_hist.shape
    k = min(k, H)
    out = torch.empty(Cq, S, D, dtype=values_hist.dtype, device=values_hist.device)
    cities = torch.arange(Cq, device=w_query.device)[:, None, None] if C > 1 else 0
```

- The inputs carry a city dimension. For a fixed city, C = 1 history serves all S query days. For
  flexible positions, each of the C′ cities has its own history (C = C′).
- `values_hist` is "whatever we want to estimate". For demand, D = N+1 numbers per day. For costs, D = P
  (the whole cost matrix flattened). **One function serves both.**
- `k = min(k, H)`: you can't take more neighbours than there are days.
- `cities` is the day/city index used in the gather below: `arange` if each city has its own history, or
  just 0 if everyone uses history 0.

<!-- quote: svrp/env/estimates.py -->
```python
    for s0 in range(0, S, chunk):                       # chunked so big N does not blow up memory
        wq = w_query[:, s0:s0 + chunk]
        d = torch.cdist(wq, w_hist.expand(Cq, H, -1))   # (C', s, H) weather distances
        dk, idx = d.topk(k, dim=-1, largest=False)
        wgt = 1.0 / (dk + 1e-6)
        wgt = wgt / wgt.sum(-1, keepdim=True)
        vals = values_hist[cities, idx]                 # (C', s, k, D)
        out[:, s0:s0 + chunk] = (wgt[..., None] * vals).sum(-2)
    return out
```

- **Chunks of 256 query days.** For N = 100, one day's costs are 10,201 numbers. 1,000 days × 10
  neighbours × 10,201 would be about 400 MB at once. Chunking keeps it small.
- `torch.cdist(...)`: the distance from each query weather to every history weather, shape (C′, s, H).
  `w_hist.expand(Cq, H, -1)` makes the shapes line up without copying (`-1` = "keep this size").
- `topk(k, largest=False)`: the k **smallest** distances (`dk`) and *which* history days they are (`idx`).
- `1/(dk + 1e-6)`: inverse distance. The `+1e-6` prevents division by zero if a past day has exactly
  today's weather.
- Normalise the weights to sum to 1.
- `values_hist[cities, idx]`: fetch the k neighbours' values, shape (C′, s, k, D). This is the arange
  indexing trick (lesson 02 section 5) with an extra dimension.
- Weighted sum over the k neighbours (dim −2), written into the output.

### `svrp/data.py`, lines 1–18: the header

<!-- quote: svrp/data.py -->
```python
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
```

- The docstring states the file's two big ideas: **truth vs belief** in one object, and **fixed seeds per
  role** so every comparison is paired.
- `dataclass, fields, replace`: `replace(obj, a=…)` makes a *copy* of a dataclass with some fields changed
  (used by `_map` and `believed`).
- Project imports: config, the generator (as `G`, so calls read `G.make_cities`), and `knn_estimate`. Layer 3
  uses layers 1 and 2 and nothing above them.

### `svrp/data.py`, lines 20–58: seeds and the `Batch`

<!-- quote: svrp/data.py -->
```python
SEED_CITY, SEED_HIST, SEED_VAL, SEED_TEST, SEED_TRAIN = 10_000, 20_000, 30_000, 40_000, 50_000
EVENT_SCALE = 2.0   # event flag value: k-NN then treats event days as far away from normal days
```

The role seeds from the table above (`10_000` is just `10000` with a readable underscore).
`EVENT_SCALE` is for lesson 13.

<!-- quote: svrp/data.py -->
```python
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
```

Read the comments as a contract:

- **Allowed to see:** `coords`, `w`, `theta`, `dist`, `dem_hat`, `cost_hat`, `capacity`.
- **Forbidden:** `demand` and `cost`. Only the environment (lesson 08) and the executor (lesson 07) touch
  them, to charge the true price or to *reveal* a customer's demand on arrival.

<!-- quote: svrp/data.py -->
```python
    @property
    def size(self) -> int:
        return self.demand.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.demand.shape[1]
```

Two read-only conveniences: `batch.size` is B (days) and `batch.n_nodes` is N+1, both read from the shape
of `demand`. Code elsewhere writes `batch.size` instead of the cryptic `batch.demand.shape[0]`.

<!-- quote: svrp/data.py -->
```python
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
```

- `_map(fn)` applies `fn` to every tensor field (all except the float `capacity`) and returns a new Batch.
  Three one-liners then follow:
  - `to(device)`: move every tensor to the GPU or CPU.
  - `index(idx)`: keep some days, e.g. `batch.index(slice(0, 100))` for the first 100.
  - `repeat_interleave(m)`: each day m times in a row (for the shared baseline, lesson 11).
- `believed()`: a Batch where the "truth" is replaced by the beliefs. Running a policy on it is
  **planning in your head**: "if the estimates were right, what would I do?" Lesson 10 uses this for
  a-priori RL plans.

### `svrp/data.py`, lines 61–90: the `World`

<!-- quote: svrp/data.py -->
```python
class World:
    """One experimental world (an EnvConfig). Holds the fixed city and its history bank if positions are fixed."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        if cfg.positions == "fixed":
            self.city = G.make_cities(cfg, 1, _gen(SEED_CITY + cfg.city_seed))
            self.hist = self._history(self.city, cfg.history_days, _gen(SEED_HIST + cfg.city_seed))
        elif cfg.positions != "flexible":
            raise ValueError(cfg.positions)
```

For a fixed city, the city and its 1,000-day history are made **once**, each with its own seed. For
flexible positions there's nothing to precompute: every day gets a brand-new city in `sample`. Any other
value of `positions` is a typo, so the code stops with an error instead of guessing.

<!-- quote: svrp/data.py -->
```python
    # --- history bank: past days the k-NN estimator can look up -------------------------------------
    def _history(self, city: G.City, days: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
        C = city.n_cities
        w = G.sample_weather((C, days), self.cfg.n_weather, gen)
        event = G.sample_events((C, days), self.cfg, gen)
        real = G.realize(city, w, self.cfg, gen, event)
        _, key = self._external(w, event)
        return {"key": key, "demand": real["demand"], "cost": real["cost"].flatten(-2)}
```

The history bank is just "past days": real weather, real demand, real costs. The costs are flattened to P
numbers per day for `knn_estimate`. The `key` is what k-NN matches on: the weather (plus the event flag
in lesson 13). In your first version write `"key": w` and skip the event lines. `_external` (lines
81–90) only matters once events exist. Without events it returns `(w, w)`.

### `svrp/data.py`, lines 92–129: `sample`, making a Batch of n days

<!-- quote: svrp/data.py -->
```python
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
```

- Fixed: weather of shape (1, n, 3), meaning *1 city × n days*.
- Flexible: n new cities, weather (n, 1, 3), meaning *n cities × 1 day each*.
- Either way, `realize` returns tensors of shape (C′, H, …) with C′·H = n.

<!-- quote: svrp/data.py -->
```python
        # drawn *after* the truth, so the true days are identical whichever estimate we use
        if cfg.positions == "fixed":
            hist = self.hist
        elif cfg.estimate == "knn":
            hist = self._history(city, cfg.history_days_flexible, gen)
        flat = lambda t: t.reshape(n, *t.shape[2:])                    # (C', H, ...) -> (n, ...)
```

**A subtle line that matters.** In flexible mode, the k-NN history is generated from the *same* random
stream `gen`. If it were drawn *before* `realize`, then switching `estimate` from `constant` to `knn`
would shift the stream and change the true days too, and the comparison would no longer be paired. So
the history is drawn **after** the truth. The test `test_truth_does_not_depend_on_estimator` checks
exactly this. (The first version of this code had the bug, and this test was written to catch it.)

`flat` merges the first two dims, (C′, H, …) → (n, …), whichever of C′ or H is 1.

<!-- quote: svrp/data.py -->
```python
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
```

The three beliefs:

- **constant:** θ̄ and the plain distances (clipped the same way as real demand, so ≥ 1 and ≤ Q under full
  delivery).
- **oracle_mean:** the noise-free values the generator kept for us.
- **knn:** two calls to `knn_estimate`, one for demand and one for the flattened costs, which are reshaped
  back to (n, N1, N1).

<!-- quote: svrp/data.py -->
```python
        expand = lambda t: t.expand(n, *t.shape[1:]).clone() if t.shape[0] == 1 else t
        return Batch(
            coords=expand(city.coords), w=flat(obs), theta=expand(city.theta), dist=expand(city.dist),
            demand=flat(real["demand"]), cost=flat(real["cost"]),
            dem_hat=dem_hat, cost_hat=cost_hat, capacity=cfg.capacity,
        )
```

For a fixed city, the coordinates are (1, N1, 2) but the Batch needs (n, N1, 2): one copy per day.
`expand` + `clone` makes real copies (lesson 02 section 11 explains why the clone matters).

### `svrp/data.py`, lines 131–143: fixed eval sets and the training stream

<!-- quote: svrp/data.py -->
```python
    def eval_set(self, split: str, n: int = 1000) -> Batch:
        seed = {"val": SEED_VAL, "test": SEED_TEST}[split] + self.cfg.city_seed
        return self.sample(n, _gen(seed))


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def train_generator(seed: int) -> torch.Generator:
    return _gen(SEED_TRAIN + seed)
```

- `eval_set("test")` always makes a *fresh* generator from the fixed test seed, so it returns the same
  1,000 days every time it is called, in any script, on any PC.
- `train_generator(seed)` is the training stream. Training keeps calling `world.sample(512, gen)` with the
  *same* `gen`, which keeps advancing, so every training batch is new. Different training seeds give
  different streams (that's what "3 seeds" means in `LOG.md`).

## Rebuild it yourself

1. `mysvrp/env/estimates.py`: `knn_estimate`. Test it on the k = 2 / k = 3 example above, with H = 3 days
   and D = 1 value.
2. `mysvrp/data.py` without events: the seeds, `Batch` (all methods), `World.__init__`, `_history`
   (returning `"key": w`), `sample` (with `key = obs = w`), `eval_set`, `_gen`, `train_generator`.

## Verify

1. Copy `tests/test_generator.py` into `learning/workspace/tests/` (with `svrp` → `mysvrp`). Run it, skipping
   the event tests until lesson 13:
   ```bash
   uv run python -m pytest tests/test_generator.py -q -k "not event"
   ```
   All 9 remaining tests must pass ("9 passed, 2 deselected"). They check the signal ratio, symmetry, clipping, reproducibility,
   *truth does not depend on the estimator*, and that k-NN beats the constant when weather matters.

2. **Fingerprint** of the N = 10 test days. If yours matches, your world is identical to the reference,
   and later lessons can compare against `LOG.md` number for number:
   ```bash
   uv run python -c "
   import sys; sys.path.insert(0, '.')
   from mysvrp.config import EnvConfig
   from mysvrp.data import World
   b = World(EnvConfig(n_customers=10)).eval_set('test', 1000)
   print(f'demand sum {b.demand.sum().item():.3f}   cost sum {b.cost.sum().item():.3f}')
   print(f'dem_hat sum {b.dem_hat.sum().item():.3f}   cost_hat sum {b.cost_hat.sum().item():.3f}')
   print('day 0 true  ', [round(x, 1) for x in b.demand[0, 1:].tolist()])
   print('day 0 belief', [round(x, 1) for x in b.dem_hat[0, 1:].tolist()])
   "
   ```
   Reference output:
   ```
   demand sum 106183.961   cost sum 64299.719
   dem_hat sum 105432.062   cost_hat sum 64125.859
   day 0 true   [21.3, 23.7, 9.3, 1.0, 1.0, 6.0, 20.4, 12.0, 7.3, 5.0]
   day 0 belief [21.3, 19.3, 10.6, 7.6, 3.6, 4.3, 10.0, 11.5, 13.3, 8.9]
   ```
   Look at day 0: customer 4 was believed to want 7.6 but wanted 1.0, and customer 7 was believed 10.0
   but wanted 20.4. Those surprises are what the rest of the project is about. (Customer 1 shows
   21.3 for both only because of rounding: the exact values are 21.32 and 21.28.)

3. How good is each belief? Mean squared error over the same 1,000 test days, at N = 10:

   | estimate | MSE vs true demand |
   |---|---|
   | constant | 57.07 |
   | k-NN (k = 10) | 33.23 |
   | oracle_mean | 29.44 |

   k-NN gets most of the way to the oracle. The oracle is not 0, because the noise ε is truly
   unpredictable.

## Common mistakes

- **Using `demand` where `dem_hat` belongs.** It silently makes everything look better. Rule: only the
  env, the executor and `hindsight` may read `demand`/`cost`.
- **Drawing the flexible history before `realize`**: `test_truth_does_not_depend_on_estimator` fails.
- **Forgetting `k = min(k, H)`**: `topk` crashes when the history is shorter than k.
- **Not clipping the k-NN demand**: an estimate can come out < 1 near the clip, or > Q under full
  delivery. The fullness test catches the latter.

## Check your understanding

1. Why does the Batch store both `demand` and `dem_hat`?
2. Two methods are evaluated with `World(cfg).eval_set("test")` in two separate scripts, a week apart.
   Do they see the same days? Why?
3. What goes wrong if you pick the learning rate by looking at test-set scores?
4. In the k-NN example, what is the estimate with k = 1? Why might a small k be bad?

**Answers** (try the questions first, then check here)

1. `dem_hat` is what decisions may use. `demand` is what actually happens, used only for scoring and for
   revealing a customer's demand on arrival.
2. Yes: `eval_set` builds a new generator from the fixed test seed every time, so the random numbers, and
   the days, are identical.
3. The test score becomes biased, because you tuned to those specific days. A fresh set of days would
   probably score worse. Use validation for choices and test only to report.
4. A and B tie at distance 0.1; whichever `topk` returns first gives 12 or 10. With k = 1 the estimate is
   one past day's demand, *including that day's noise*. Larger k averages the noise away (but too large
   a k pulls in days with different weather).


## Read more

- k-NN regression: *An Introduction to Statistical Learning* (free PDF at statlearning.com), section 3.5.
- Why train / validation / test splits matter: same book, chapter 5 (cross-validation).

**Next:** [06 — Plans and classical solvers](06_plans_and_classical_solvers.md)
