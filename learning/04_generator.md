# 04 — The random world: turning the paper's equation into code (`svrp/env/generator.py`)

| | |
|---|---|
| **Layer** | 2 |
| **Files covered** | `svrp/env/generator.py` (174 lines), and later `tests/test_generator.py` |
| **You build** | `learning/workspace/mysvrp/env/generator.py` |
| **Time** | 4–6 hours (the math section deserves slow reading) |
| **Before this** | [02 — Tensors](02_tensors.md), [03 — Config](03_config.md); re-read `a.md` and `b.md` |

## Why this layer exists

In OR-Tools you give the solver *the* demand of each customer and *the* distance of each road. In the
stochastic VRP, nobody knows those numbers in advance. They change every day.

To train and test anything, we need a **machine that produces realistic random days**: on a hot,
humid day these customers order more, the roads are slower, plus some unpredictable noise. That machine
is this file. It implements the equation from your notes (`a.md`, section 2):

```
θ_i = θ̄_i + Σ_m Σ_n α_imn · w_m · w_n + ε_i
      base     weather effect               noise
```

and the signal ratio A / B / Γ from `b.md`.

**One design decision is the most important line of thinking in this lesson.** In the authors' public
code, the secret weights α are drawn *fresh for every day*. Then "hot days mean more demand" is true on
Monday and false on Tuesday: weather has no consistent effect, so nothing can learn it. Their own
headline finding ("weather info helps up to 2%") could not be reproduced with that code. Here, α is
drawn **once per city** and stays fixed. The weather is then a real, learnable clue.

## Concepts from zero

### City vs day

- A **city** is everything that stays the same over time: where the customers live, each customer's
  average demand θ̄_i, the distances, and the secret weather weights α. In the default setting
  (`positions = "fixed"`) there is **one** city, the same every day (like B2B deliveries to the same
  shops).
- A **day** (a *scenario*) is what changes: today's weather w and today's noise ε, which together give
  today's actual demands ξ_i and travel costs c_ij.

### The demand formula, piece by piece

For customer i on a day with weather w = (w₁, w₂, w₃), each w_m ∈ [−1, 1]:

1. **Base** θ̄_i ∈ [5, 15]: what this customer orders on an average day.
2. **Weather effect** S_i(w) = wᵀ α_i w − E[wᵀ α_i w]. The quadratic form Σ α_mn w_m w_n (lesson 00, C4)
   captures interactions such as "hot **and** humid" (w₁·w₃). We subtract its average, so on average
   the weather adds nothing and **E[ξ_i] ≈ θ̄_i**. Without the subtraction, θ̄_i would not be the mean,
   and the fill rate Φ = E[ξ]/Q would be wrong.
3. **Noise** ε_i ~ N(0, σ_i²): the unpredictable part.

Then ξ_i = max(1, θ̄_i + S_i(w) + ε_i): nobody orders less than 1 unit.

**Travel cost** uses the same recipe per *edge*: c_ij = max(0.1·d_ij, d_ij + S_ij(w) + ε_ij), where the
base is the distance d_ij. The noise is symmetric: today's i→j costs the same as j→i.

### The signal ratio: three buckets of "size"

`b.md` splits the total size T = θ̄² + E[S²] + E[ε²] into shares:

A = θ̄² / T (constant part), B = E[S²] / T (weather part), Γ = E[ε²] / T (noise part).

We **choose** A, B, Γ (default 0.6, 0.2, 0.2) and must build the world to match. Trick: if we make

```
Var(S) = (B/A)·θ̄²      and      Var(ε) = (Γ/A)·θ̄²,
```

then T = θ̄²·(1 + B/A + Γ/A) = θ̄²·(A + B + Γ)/A = θ̄²/A when A + B + Γ = 1. So θ̄²/T = A, Var(S)/T = B,
and Var(ε)/T = Γ. Exactly the targets.

Worked numbers for a customer with θ̄ = 10 at 0.6 / 0.2 / 0.2: Var(S) = (0.2/0.6)·100 = 33.3, so the
weather std is 5.77. The noise std is also 5.77. The total std is √(33.3 + 33.3) = 8.2. **This customer
orders 10 ± 8 units.** That's a very noisy world, and it explains why planning ahead fails so badly in
`experiments/LOG.md` (E1).

How do we make Var(S) equal a target? Draw random raw weights, compute Var(S) for them with a formula,
then **multiply α by √(target / Var)**. Scaling α by c scales S by c and its variance by c² (lesson 00,
B3).

### The variance formula (why 4/45 and 1/9?)

With a = sym(α) (symmetric, lesson 00 C4):

```
S = Σ_m a_mm·w_m²  +  2·Σ_{m<n} a_mn·w_m·w_n
```

Every term here is **uncorrelated** with every other (e.g. E[w₁²·w₁w₂] = E[w₁³]·E[w₂] = 0, because odd
powers of a symmetric variable average to 0). So the variances simply add:

- Var(a_mm·w_m²) = a_mm²·Var(w²) = a_mm² · (1/5 − 1/9) = a_mm² · **4/45**
- Var(2·a_mn·w_m·w_n) = 4·a_mn²·E[w_m²]·E[w_n²] = 4·a_mn² · **1/9** (the mean is 0, so Var = E[square])

and **E[S] = Σ_m a_mm · E[w²] = Σ_m a_mm / 3**. These are exactly lines 25–41 of the file. The test
checks them against a simulation, so you don't have to trust the algebra blindly.

## Code walkthrough

### Lines 1–28: docstring, imports, the three constants

<!-- quote: svrp/env/generator.py -->
```python
"""Stochastic world generator: the equation from a.md, turned into code.

    θ = θ̄  +  wᵀ α w  −  E[wᵀ α w]  +  ε
        base   weather effect (centred)   noise

θ is either a customer's demand ξ_i (θ̄_i ~ U[5,15]) or the travel cost of an edge c_ij (θ̄_ij = distance).

Shapes (C = number of cities, P = (N+1)² edges, K = M² weather products, node 0 = depot):
    coords (C, N+1, 2)      theta (C, N+1)      dist (C, N+1, N+1)
    alpha_dem (C, N+1, K)   alpha_cost (C, P, K)
    shift_* / sigma_*       same shape as the quantity they belong to

Everything is sampled with an explicit torch.Generator on the CPU, so a seed always gives the same world.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from svrp.config import EnvConfig
```

- **The docstring is the file's summary.** It gives the formula, what θ stands for, every tensor shape, and
  the promise "a seed always gives the same world". Write your own docstring *first* when you rebuild a
  file; it forces you to know what the file is for.
- `from __future__ import annotations`: lets type hints mention any type freely (lesson 00, A3).
  Every project file starts with it.
- `dataclass, fields`: for the `City` container below.
- `from svrp.config import EnvConfig`: the **only** project import. The generator sits directly above
  config in the layer map, so it may use config and nothing else.

<!-- quote: svrp/env/generator.py -->
```python
DEPOT_XY = 0.5

# For w ~ U(-1, 1):  E[w²] = 1/3,  Var(w²) = E[w⁴] − E[w²]² = 1/5 − 1/9 = 4/45,  E[w_m² w_n²] = 1/9 (m≠n)
_E_W2 = 1.0 / 3.0
_VAR_W2 = 4.0 / 45.0
_E_W2W2 = 1.0 / 9.0
```

- The docstring (lines 1–14) lists every shape. Read it: you will come back to it often.
- `DEPOT_XY = 0.5`: the depot sits in the middle of the unit square, as in the paper.
- The constants are the moments you derived in lesson 00. The leading `_` means "private to this module".

### Lines 31–42: `weather_moments`, the formula above in code

<!-- quote: svrp/env/generator.py -->
```python
def weather_moments(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and variance of S = wᵀ a w for symmetric a (..., M, M) and independent w_m ~ U(-1, 1).

    S = Σ_m a_mm w_m² + 2 Σ_{m<n} a_mn w_m w_n, and all those terms are uncorrelated, so
        E[S]   = Σ_m a_mm / 3
        Var(S) = (4/45) Σ_m a_mm² + 4·(1/9) Σ_{m<n} a_mn²
    """
    diag = torch.diagonal(a, dim1=-2, dim2=-1)
    mean = diag.sum(-1) * _E_W2
    off = torch.triu(a, diagonal=1)
    var = _VAR_W2 * (diag ** 2).sum(-1) + 4.0 * _E_W2W2 * (off ** 2).sum((-2, -1))
    return mean, var
```

- `a` has shape (..., M, M): any number of leading dims (cities, customers, edges), with an M×M matrix
  at the end. The function computes the answer for all of them at once.
- `torch.diagonal(a, dim1=-2, dim2=-1)`: the a_mm entries, shape (..., M).
- `mean = Σ a_mm / 3`.
- `torch.triu(a, diagonal=1)` keeps only the entries **above** the diagonal (m < n) and zeros the rest.
- `var`: the two sums from the derivation. `.sum((-2, -1))` sums over both matrix dims.

### Lines 45–63: `City`, a container for everything fixed

<!-- quote: svrp/env/generator.py -->
```python
@dataclass
class City:
    coords: torch.Tensor
    theta: torch.Tensor
    dist: torch.Tensor
    alpha_dem: torch.Tensor
    shift_dem: torch.Tensor
    sigma_dem: torch.Tensor
    alpha_cost: torch.Tensor
    shift_cost: torch.Tensor
    sigma_cost: torch.Tensor
    near_venue: torch.Tensor    # (C, N+1) 1.0 for customers close to the event venue (all 0 if events are off)

    @property
    def n_cities(self) -> int:
        return self.coords.shape[0]

    def index(self, idx) -> "City":
        return City(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})
```

- For demand: `alpha_dem` holds the scaled weights, `shift_dem` the E[wᵀαw] to subtract, and
  `sigma_dem` the noise std. The same three exist for cost.
- `near_venue` is for lesson 13's events. Leave it out for now.
- `index(idx)` returns a City with every field indexed, e.g. `city.index([0, 3])` keeps cities 0 and 3.
  The dict comprehension over `fields(self)` means you never have to update `index` when you add a field.

### Lines 66–74: `_scaled_alpha`, hitting the target variance

<!-- quote: svrp/env/generator.py -->
```python
def _scaled_alpha(raw: torch.Tensor, base: torch.Tensor, b_over_a: float):
    """Symmetrise raw weights and scale them so Var(weather term) = (B/A)·base²."""
    a = 0.5 * (raw + raw.transpose(-1, -2))
    _, var = weather_moments(a)
    target = b_over_a * base ** 2
    scale = torch.sqrt(target / var.clamp_min(1e-12))
    a = a * scale[..., None, None]
    mean, _ = weather_moments(a)
    return a.flatten(-2), mean          # flattened (…, K) weights and the E[wᵀaw] to subtract
```

1. Symmetrise: a = (raw + rawᵀ)/2. This doesn't change S (lesson 00 C4), and the variance formula needs it.
2. Compute the variance the random weights *happen* to give.
3. `target = (B/A)·base²`, with `base` = θ̄_i for demand and d_ij for cost.
4. `scale = √(target / var)`. `clamp_min(1e-12)` avoids dividing by zero.
5. Multiply each M×M matrix by its own scale. `scale[..., None, None]` adds two size-1 dims so the
   scale broadcasts over the matrix.
6. Recompute the mean **after** scaling (the mean scales too), to use as the shift.
7. Return the weights flattened to K = 9 numbers (to multiply with the 9 products w_m·w_n), plus the shift.

Edge cases fall out naturally: the depot has base 0, so target 0, scale 0, and **no weather effect**.
The diagonal edges (i→i) have distance 0, so they get no effect either. If B = 0, all scales are 0 and
the weather does nothing.

### Lines 77–106: `make_cities`, drawing the fixed part

<!-- quote: svrp/env/generator.py -->
```python
def make_cities(cfg: EnvConfig, n: int, gen: torch.Generator) -> City:
    N, M = cfg.n_customers, cfg.n_weather
    coords = torch.rand(n, N + 1, 2, generator=gen)
    coords[:, 0] = DEPOT_XY
    theta = cfg.demand_low + (cfg.demand_high - cfg.demand_low) * torch.rand(n, N + 1, generator=gen)
    theta[:, 0] = 0.0
    dist = torch.cdist(coords, coords)

    b_over_a = cfg.b_ratio / cfg.a_ratio
    noise_scale = (cfg.gamma_ratio / cfg.a_ratio) ** 0.5

    raw_dem = torch.randn(n, N + 1, M, M, generator=gen)
    alpha_dem, shift_dem = _scaled_alpha(raw_dem, theta, b_over_a)

    raw_cost = torch.randn(n, N + 1, N + 1, M, M, generator=gen)
    raw_cost = 0.5 * (raw_cost + raw_cost.transpose(1, 2))        # same weather effect for i→j and j→i
    alpha_cost, shift_cost = _scaled_alpha(raw_cost, dist, b_over_a)
```

- `n` cities at once (1 for a fixed city, B for flexible positions).
- Positions: U[0, 1]², then node 0 is overwritten with the depot (0.5, 0.5).
- θ̄ ~ U[5, 15] via `low + (high − low)·U(0, 1)`. The depot's θ̄ is set to 0.
- `dist`: all pairwise Euclidean distances, shape (n, N+1, N+1).
- `b_over_a` = B/A. `noise_scale` = √(Γ/A), so the noise **std** is `noise_scale·θ̄` (std = √variance).
- Demand weights: one 3×3 matrix per node, scaled to target (B/A)·θ̄².
- Cost weights: one 3×3 matrix per edge. Line 92 makes them **symmetric in (i, j)**: (A + Aᵀ)/2 over the
  node dimensions 1 and 2. Then a rainy day slows i→j and j→i equally.

<!-- quote: svrp/env/generator.py -->
```python
    near_venue = torch.zeros(n, N + 1)
    if cfg.event_prob > 0:              # drawn last, so worlds without events keep the exact same RNG stream
        venue = torch.rand(n, 1, 2, generator=gen)
        near_venue = ((coords - venue).norm(dim=-1) < cfg.event_radius).float()
        near_venue[:, 0] = 0.0

    return City(
        coords=coords, theta=theta, dist=dist,
        alpha_dem=alpha_dem, shift_dem=shift_dem, sigma_dem=noise_scale * theta,
        alpha_cost=alpha_cost.flatten(1, 2), shift_cost=shift_cost, sigma_cost=noise_scale * dist,
        near_venue=near_venue,
    )
```

- Lines 95–99 are the event venue (lesson 13). The comment states an important rule: **random draws
  happen in a fixed order**. A new feature draws *after* everything else, so old worlds stay
  bit-identical. For now, just write `near_venue = torch.zeros(n, N + 1)`.
- `alpha_cost.flatten(1, 2)`: (n, N+1, N+1, K) → (n, P, K). Every edge becomes one row, so a single matrix
  product handles all edges (section 12 of lesson 02).

### Lines 109–121: weather (and events)

<!-- quote: svrp/env/generator.py -->
```python
def sample_weather(shape: tuple[int, ...], n_weather: int, gen: torch.Generator) -> torch.Tensor:
    return 2.0 * torch.rand(*shape, n_weather, generator=gen) - 1.0


def _weather_products(w: torch.Tensor) -> torch.Tensor:
    return (w[..., :, None] * w[..., None, :]).flatten(-2)      # (..., M²): all w_m · w_n
```

- `sample_weather((C, H), 3, gen)` gives shape (C, H, 3) of U(−1, 1) values. `*shape` unpacks the tuple
  into separate arguments.
- `_weather_products`: all 9 products, via the outer-product broadcast from lesson 02 section 4.
- `sample_events` (lines 109–113) returns `None` when events are off. Skip it until lesson 13.

### Lines 124–161: `realize`, making one or many actual days

<!-- quote: svrp/env/generator.py -->
```python
def realize(city: City, w: torch.Tensor, cfg: EnvConfig, gen: torch.Generator,
            event: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Draw the actual demands and travel costs for given weather (and event flags, shape (C', H)).

    w has shape (C', H, M): H days for each of C' cities. `city` has C = C' cities, or C = 1
    (one city shared by every day). Returns tensors of shape (C', H, N+1) and (C', H, N+1, N+1):
        demand, cost       the realized values (what actually happens)
        mean_dem, mean_cost   the noise-free values E[· | w] (used by the oracle_mean estimate)
    """
```

The signature takes the city, the weather, the config, the random stream, and (lesson 13) optional event
flags with a default of `None`, so older callers that don't pass events still work. The docstring is the
contract for shapes: `w` is (C′, H, M), and the outputs are (C′, H, …). It also names the two kinds of
output, actual values and noise-free means. The body:

<!-- quote: svrp/env/generator.py -->
```python
    Cq, H, _ = w.shape
    N1 = city.theta.shape[1]
    ww = _weather_products(w)                                              # (C', H, K)

    weather_dem = ww @ city.alpha_dem.transpose(-1, -2) - city.shift_dem[:, None, :]        # (C', H, N+1)
    weather_cost = (ww @ city.alpha_cost.transpose(-1, -2)).view(Cq, H, N1, N1) - city.shift_cost[:, None]

    eps_dem = torch.randn(Cq, H, N1, generator=gen) * city.sigma_dem[:, None, :]
    eps_cost = torch.triu(torch.randn(Cq, H, N1, N1, generator=gen), diagonal=1)
    eps_cost = (eps_cost + eps_cost.transpose(-1, -2)) * city.sigma_cost[:, None]
```

The shapes are the key. `w` is (C′, H, M): H days for each of C′ cities. The `city` has either C = C′
cities (flexible: each day its own city) or C = 1 (fixed: one city shared by all days). Broadcasting in
`@` (lesson 02 section 12) makes **one line work for both cases**.

- `ww`: (C′, H, 9).
- `weather_dem`: (C′, H, 9) @ (C, 9, N1) → (C′, H, N1). That is Σ_k α_ik·(w_m w_n)_k for every city, day
  and node, minus the shift. `shift_dem[:, None, :]` is (C, 1, N1), broadcast over the H days.
- `weather_cost`: the same over P edges, then `.view(Cq, H, N1, N1)` turns the P edges back into a matrix.
- `eps_dem`: N(0, 1) samples times each customer's σ.
- `eps_cost`: symmetric noise (the `triu` + mirror trick from lesson 02 section 10), times σ_ij.

<!-- quote: svrp/env/generator.py -->
```python
    theta = city.theta[:, None, :]
    dist = city.dist[:, None]
    if cfg.stochastic_demand:
        mean_dem, demand = theta + weather_dem, theta + weather_dem + eps_dem
    else:
        mean_dem = demand = theta.expand(Cq, H, N1)
```

- Two versions are kept for each quantity: `mean_dem` (no noise) = E[ξ | w], used later by the "oracle"
  estimate, and `demand`, the actual value.
- If demand is not stochastic, it is just θ̄ every day. `expand` makes the H copies without using memory.
- Notice the noise was drawn **even if it's not used**. That keeps the random stream identical whatever
  the switches are, so turning off stochastic demand doesn't change the travel-cost noise.

Lines 150–152 add the event boost (lesson 13). Lines 153–156 do the same for cost:

<!-- quote: svrp/env/generator.py -->
```python
    if cfg.stochastic_cost:
        mean_cost, cost = dist + weather_cost, dist + weather_cost + eps_cost
    else:
        mean_cost = cost = dist.expand(Cq, H, N1, N1)

    return {
        "demand": clip_demand(demand, cfg), "mean_dem": clip_demand(mean_dem, cfg),
        "cost": clip_cost(cost, dist), "mean_cost": clip_cost(mean_cost, dist),
    }
```

### Lines 164–174: clipping to realistic values

<!-- quote: svrp/env/generator.py -->
```python
def clip_demand(d: torch.Tensor, cfg: EnvConfig) -> torch.Tensor:
    """Customers order at least 1 unit; under full delivery an order must fit in one truck."""
    hi = cfg.capacity if cfg.delivery == "full" else float("inf")
    d = d.clamp(1.0, hi).clone()
    d[..., 0] = 0.0                     # the depot has no demand
    return d


def clip_cost(c: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """A trip never costs less than 10% of its distance (diagonal stays exactly 0)."""
    return torch.maximum(c, 0.1 * dist)
```

- Demand is clipped to ≥ 1. Under **full** delivery an order can't be split, so it must also fit in one
  truck (≤ Q).
- `clamp(1, …)` also turned the depot's 0 into 1, so line 168 resets it to 0.
- Costs are clipped to ≥ 10% of the distance. With 82% relative noise, unclipped costs could go
  negative, which makes no sense. On the diagonal, `0.1 · 0 = 0` keeps self-loops at exactly 0.

**Side effect worth knowing:** clipping cuts off the low tail, so the average demand ends up slightly
*above* θ̄ (you'll see it in the check below). The signal-ratio math describes the world *before*
clipping.

## Rebuild it yourself

Order matters: each step can be checked before the next.

1. Constants + `weather_moments`. Check it with part 1 of the script below.
2. `City` (without `near_venue` for now, or with it always zeros), `_scaled_alpha`, `make_cities`.
3. `sample_weather`, `_weather_products`, `realize`, `clip_demand`, `clip_cost`.

**Draw random numbers in exactly the same order as the reference**: coords → theta → raw_dem →
raw_cost in `make_cities`, then eps_dem → eps_cost in `realize`. Any order gives a valid world. But
only the same order gives the *same* world, and then your later results can be compared number for
number with `experiments/LOG.md`.

## Verify

The full `tests/test_generator.py` also needs `data.py` (next lesson). For now, save this as
`learning/workspace/check04.py` and run `uv run python check04.py` from `learning/workspace`:

```python
import sys
sys.path.insert(0, ".")
import torch
from mysvrp.config import EnvConfig
from mysvrp.env import generator as G

g = torch.Generator(); g.manual_seed(0)

# 1. the closed-form moments vs a simulation with 200,000 random weathers
a = torch.randn(2, 3, 3, generator=g); a = 0.5 * (a + a.transpose(-1, -2))
mean, var = G.weather_moments(a)
w = G.sample_weather((200_000,), 3, g)
s = torch.einsum("sm,kmn,sn->ks", w, a, w)
print("mean: formula", [round(x, 3) for x in mean.tolist()], " simulated", [round(x, 3) for x in s.mean(1).tolist()])
print("var : formula", [round(x, 3) for x in var.tolist()], " simulated", [round(x, 3) for x in s.var(1).tolist()])

# 2. one city of 5 customers, 50,000 days: are the three shares 0.6 / 0.2 / 0.2 (before clipping)?
cfg = EnvConfig(n_customers=5)
city = G.make_cities(cfg, 1, g)
w = G.sample_weather((1, 50_000), 3, g)
ww = (w[..., :, None] * w[..., None, :]).flatten(-2)
weather = (ww @ city.alpha_dem.transpose(-1, -2) - city.shift_dem[:, None])[0, :, 1:]   # (days, customers)
base2 = city.theta[0, 1:] ** 2
noise2 = city.sigma_dem[0, 1:] ** 2
total = base2 + weather.var(0) + noise2
print("A share", [round(x, 3) for x in (base2 / total).tolist()])
print("B share", [round(x, 3) for x in (weather.var(0) / total).tolist()])
print("G share", [round(x, 3) for x in (noise2 / total).tolist()])
print("weather term mean (should be ~0):", [round(x, 2) for x in weather.mean(0).tolist()])

# 3. the clip at 1 unit pushes the average demand slightly ABOVE theta_bar
real = G.realize(city, w, cfg, g)
print("theta_bar         ", [round(x, 2) for x in city.theta[0, 1:].tolist()])
print("avg clipped demand", [round(x, 2) for x in real["demand"][0, :, 1:].mean(0).tolist()])
print("cost symmetric:", torch.allclose(real["cost"], real["cost"].transpose(-1, -2)))
```

Output of the reference implementation (yours should match exactly if you drew in the same order;
otherwise the *shares* must still be 0.6 / 0.2 / 0.2):

```
mean: formula [-0.02, 0.896]  simulated [-0.019, 0.9]
var : formula [0.194, 0.971]  simulated [0.193, 0.972]
A share [0.6, 0.6, 0.6, 0.599, 0.6]
B share [0.2, 0.2, 0.2, 0.201, 0.2]
G share [0.2, 0.2, 0.2, 0.2, 0.2]
weather term mean (should be ~0): [-0.0, 0.04, 0.0, -0.01, 0.01]
theta_bar          [8.89, 14.45, 14.56, 8.8, 6.4]
avg clipped demand [9.26, 15.32, 15.52, 9.31, 6.8]
cost symmetric: True
```

After lesson 05, copy `tests/test_generator.py` (see the workspace instructions in the [README](README.md))
and run `uv run python -m pytest tests/test_generator.py -q`. All tests there must pass.

## Common mistakes

- **Forgetting the shift.** The weather term then has a non-zero mean, θ̄ stops being the average, and
  the fill rate is off. The "weather term mean ≈ 0" line catches it.
- **Using variance where std is needed.** Noise is `noise_scale * θ̄` with `noise_scale = √(Γ/A)`, not
  Γ/A. A wrong G share (0.07 or 0.43 instead of 0.2) is the symptom.
- **Not symmetrising before the variance formula.** The formula assumes a symmetric matrix, so the B
  share comes out wrong.
- **Shape errors in `realize`**: print `ww.shape`, `city.alpha_dem.shape` and `weather_dem.shape`. They
  should be (1, H, 9), (1, N1, 9) and (1, H, N1) for a fixed city.

## Check your understanding

1. Why must α be fixed per city, rather than drawn per day like in the authors' code?
2. A customer has θ̄ = 12 with ratios 0.6 / 0.2 / 0.2. What are the std of its weather term and of its noise?
3. With B = 0, what does `_scaled_alpha` return, and what does the weather do?
4. Why is the noise drawn even when `stochastic_demand=False`?
5. Why is `mean_dem` stored next to `demand`?

**Answers** (try the questions first, then check here)

1. If α changes every day, the same weather has a different effect each day, so the weather carries no
   learnable information. Fixed α makes "this weather → this demand" a stable pattern.
2. Var = (0.2/0.6)·144 = 48 for both, so both stds are √48 ≈ 6.93.
3. All-zero weights (scale 0) and zero shift. The weather has no effect at all.
4. So the stream of random numbers is the same whatever the switches are. Turning one feature off
   doesn't silently change all the other random values.
5. It is E[demand | weather], the best possible guess from the weather. The `oracle_mean` estimate
   (lesson 05) uses it as a ceiling for how much weather can help.


## Read more

- `a.md` and `b.md` (your notes), then the paper's Section 3.
- The authors' generator, to see the "α per day" issue yourself: github.com/Zangir/SVRP, file `env/weather.py`.
- Variance of sums: any probability textbook, chapter "Expectation and variance".

**Next:** [05 — Estimates and data](05_estimates_and_data.md)
