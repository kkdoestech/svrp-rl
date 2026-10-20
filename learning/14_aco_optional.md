# 14 (optional) — Ant Colony Optimization: the paper's champion (`svrp/baselines/aco.py`)

| | |
|---|---|
| **Layer** | optional (a 4th classical planner) |
| **Files covered** | `svrp/baselines/aco.py` (101 lines), `scripts/tune_aco.py` (39), the ACO parts of `runner.py` and `tests/test_baselines.py` |
| **You build** | `mysvrp/baselines/aco.py`, `learning/workspace/scripts/tune_aco.py` |
| **Time** | 2–3 hours |
| **Before this** | [06](06_plans_and_classical_solvers.md), [07](07_executor_and_runner.md); `part 6.md` ("Ant Colony Optimization") |

## Why this lesson exists

In the paper, ACO is the strongest classical method, and the headline claim is "RL beats ACO by 3.43%".
To check a claim like that, you need your own ACO, **tuned fairly**. Otherwise it's easy to beat a weak
version. It's also a beautiful algorithm, and a nice contrast to RL: both learn from many tries, but ACO
learns about *one* day, while RL learns a policy for *all* days.

(In this project's world ACO turned out to be the *weakest* classical planner: 15.01 vs OR-Tools' 14.53 at
N = 20, `LOG.md` E1. Implementations differ, which is exactly why the project compares against OR-Tools.)

## Concepts from zero

### Ants and pheromone

Real ants find short paths to food: each ant leaves pheromone on its path, ants prefer paths with more
pheromone, and pheromone evaporates. Short paths are walked more often per unit time, so they collect more
pheromone, which attracts more ants. **The colony converges on short paths without any ant knowing the map.**

The algorithm copies that. Each iteration:

1. **Construct.** Each ant builds a full solution. From its current node it picks the next customer j with
   probability
   ```
   p(j) ∝ τ_ij^α · η_ij^β        τ = pheromone on edge i→j,  η = 1 / cost_ij ("greediness")
   ```
   among customers whose demand still fits in the truck. If none fits, it goes back to the depot.
   α weighs the colony's experience, and β weighs plain greed.
2. **Evaporate:** τ ← (1 − ρ)·τ. Old information fades, so the colony can change its mind.
3. **Deposit:** every ant adds 1/L (L = its solution's cost) on each edge it used. Cheaper solutions deposit
   more. The **elitist ant**, the best solution so far, deposits extra (`elite / L_best`).

### The roulette wheel

To pick j with probability ∝ weights w = [0, 2, 1, 1] (node 0 not allowed): the cumulative sums are
[0, 2, 3, 4]. Spin u uniformly in [0, 4). The answer is the first position whose cumulative sum is > u:
u = 2.5 → node 2. In code: count how many cumulative sums are **< u**: `(c < u).sum()` = 2 → node 2. Nodes
with weight 0 can never be chosen, because their cumulative sum equals the previous one.

### 2-opt (local polish)

A route that crosses itself can always be shortened by "uncrossing": take the route
0 → A → **C → B** → D → 0, reverse the middle segment, and get 0 → A → **B → C** → D → 0. Only two edges
change, A–C and B–D become A–B and C–D, so the change in cost is

```
Δ = c(A,B) + c(C,D) − c(A,C) − c(B,D)
```

If Δ < 0, do it. Repeat until no reversal helps.

## Code walkthrough

### Lines 1–15: the header

<!-- quote: svrp/baselines/aco.py -->
```python
"""Ant Colony Optimization for CVRP (Ant System + elitist ant + 2-opt), planning on estimates.

Each ant builds a full solution: from its current node it picks the next customer with probability
    p(j) ∝ τ_ij^α · η_ij^β,   η_ij = 1 / c_ij   (pheromone^α × greediness^β),
among customers whose (estimated) demand still fits in the truck; if none fits it returns to the depot.
After each iteration pheromone evaporates (τ ← (1−ρ)τ) and every ant deposits 1/L on the edges it used,
plus an extra deposit on the best solution found so far ("elitist ant"). Good edges get reinforced.

All ants are simulated in parallel with numpy.
"""
from __future__ import annotations

import numpy as np

from svrp.baselines.common import plan_cost, plan_to_routes, planning_demand, routes_to_plan
```

The docstring states the probability rule, the evaporation and the deposits: the three ideas from the
"Concepts" section. ACO is pure NumPy, and it reuses lesson 06's plan helpers (`plan_cost` to price an
ant's plan, `plan_to_routes` / `routes_to_plan` around 2-opt, and `planning_demand` for clipping).

### Lines 17–20: the parameter grid (and an honest note)

<!-- quote: svrp/baselines/aco.py -->
```python
# (alpha = pheromone importance, beta = heuristic importance, ants). The paper's Table 1 lists
# "3/4/20, 6/8/24, 4/9/15, 3/10/12" without labelling the columns; we read them in this order
# and add two textbook settings. Tuned on the validation set (scripts/tune_aco.py).
PARAM_GRID = [(3, 4, 20), (6, 8, 24), (4, 9, 15), (3, 10, 12), (1, 2, 20), (1, 5, 20)]
```

The paper lists four ACO settings without saying which number is which. The comment states the
interpretation used, and adds two standard settings. When you can't know something, write down your
assumption.

### Lines 23–36: 2-opt

<!-- quote: svrp/baselines/aco.py -->
```python
def two_opt(route: list[int], cost: np.ndarray) -> list[int]:
    """Classic 2-opt on one route (depot at both ends): reverse segments while that shortens it."""
    r = [0] + route + [0]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(r) - 2):
            for j in range(i + 1, len(r) - 1):
                delta = (cost[r[i - 1], r[j]] + cost[r[i], r[j + 1]]
                         - cost[r[i - 1], r[i]] - cost[r[j], r[j + 1]])
                if delta < -1e-12:
                    r[i:j + 1] = r[i:j + 1][::-1]
                    improved = True
    return r[1:-1]
```

- Put the depot at both ends, then try every segment r[i..j] that doesn't include the depots.
- `delta` is the formula above, with A = r[i−1], C = r[i], B = r[j], D = r[j+1]. (It assumes symmetric
  costs, which is true here: lesson 04 made them symmetric.)
- Reverse with slice assignment: `r[i:j+1] = r[i:j+1][::-1]`. Keep looping while anything improved.
- `-1e-12` instead of `< 0`: don't loop forever over float noise.

### Lines 39–70: all ants walk at once (vectorised)

<!-- quote: svrp/baselines/aco.py -->
```python
def _construct(tau, eta_b, d, capacity, n_ants, alpha, rng):
    N1 = len(d)
    A = np.arange(n_ants)
    visited = np.zeros((n_ants, N1), dtype=bool)
    visited[:, 0] = True
    pos = np.zeros(n_ants, dtype=np.int64)
    load = np.full(n_ants, capacity, dtype=np.float64)
    seqs = [[0] for _ in range(n_ants)]
    tau_a = tau ** alpha
```

Like the environment (lesson 08), but for ants: one row per ant. `tau ** alpha` is computed once per
iteration, not once per step.

<!-- quote: svrp/baselines/aco.py -->
```python
    while not visited.all():
        feasible = ~visited & (d[None, :] <= load[:, None] + 1e-9)
        has = feasible.any(1)
        unfinished = ~visited.all(1)
        w = tau_a[pos] * eta_b[pos] * feasible
        nxt = np.zeros(n_ants, dtype=np.int64)
        if has.any():
            c = w[has].cumsum(1)                            # roulette wheel: one spin per ant
            u = rng.random(int(has.sum()))[:, None] * c[:, -1:]
            nxt[has] = (c < u).sum(1)
```

- `feasible`: unvisited customers whose demand fits on *this ant's* truck (broadcasting: (1, N1) vs (n_ants, 1)).
- `w`: τ^α · η^β from each ant's current node to every node, zeroed where infeasible. `tau_a[pos]` picks
  one row per ant.
- The roulette wheel for all ants that have an option: one spin each, scaled to each ant's total weight.

**A real bug that was fixed here.** The first version normalised the weights to probabilities and drew
u in [0, 1). Rounding could make the last cumulative probability 0.99999999 < u, and then the "chosen
index" was N1, one past the end. Scaling u by the *unnormalised* total `c[:, -1:]` guarantees u < total,
so the index is always valid. Edge cases like this are why rare crashes happen "once in 10,000 days".

<!-- quote: svrp/baselines/aco.py -->
```python
        # ants with nothing feasible go home to refill (nxt stays 0)
        move = unfinished
        for a in A[move]:
            seqs[a].append(int(nxt[a]))
        go_cust = move & (nxt != 0)
        visited[A[go_cust], nxt[go_cust]] = True
        load[go_cust] -= d[nxt[go_cust]]
        load[move & (nxt == 0)] = capacity
        pos[move] = nxt[move]
    for s in seqs:
        if s[-1] != 0:
            s.append(0)
    return seqs
```

Ants with no feasible customer keep `nxt = 0` and go home (refill). Finished ants stop moving. Update the
visited set, the loads and the positions, all vectorised. Only the sequence bookkeeping is a Python loop.
When every ant has visited everyone, each sequence is closed with a final depot visit (if it doesn't already
end at 0), so every ant returns a valid plan in lesson 06's format.

### Lines 73–101: the colony's loop

<!-- quote: svrp/baselines/aco.py -->
```python
def aco(dem: np.ndarray, cost: np.ndarray, capacity: float, alpha: float = 3, beta: float = 10,
        n_ants: int = 12, iters: int = 60, rho: float = 0.1, elite: float = 5.0, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    d = planning_demand(dem, capacity).astype(np.float64)
    d[0] = 0.0
    N1 = len(d)
    eta = 1.0 / np.maximum(cost, 1e-9)
    np.fill_diagonal(eta, 0.0)
    eta_b = eta ** beta
    eta_b /= eta_b.max()                                   # keep numbers in a safe float range
```

- The signature's defaults (α = 3, β = 10, 12 ants, 60 iterations, ρ = 0.1, elite weight 5, seed 0) are one
  of the paper's settings. The tuned values from `results/aco_params.json` override them in evaluation.
- A seeded random generator: same input, same answer (reproducible baselines).
- η = 1/cost (with no division by zero, and no self-loops). Raising to β = 10 can overflow float range for
  tiny costs, so it's normalised by its maximum. This doesn't change any probability, since only *ratios*
  matter in p(j) ∝ ….

<!-- quote: svrp/baselines/aco.py -->
```python
    tau = np.ones((N1, N1))
    best_plan, best_len = None, np.inf
    for _ in range(iters):
        plans = _construct(tau, eta_b, d, capacity, n_ants, alpha, rng)
        lens = np.array([plan_cost(p, cost) for p in plans])
        k = int(lens.argmin())
        polished = routes_to_plan([two_opt(r, cost) for r in plan_to_routes(plans[k])])
        plen = plan_cost(polished, cost)
        if plen < best_len:
            best_plan, best_len = polished, plen
        tau *= 1.0 - rho
        for p, L in zip(plans, lens):
            p = np.asarray(p)
            np.add.at(tau, (p[:-1], p[1:]), 1.0 / L)
        bp = np.asarray(best_plan)
        np.add.at(tau, (bp[:-1], bp[1:]), elite / best_len)
        tau = np.maximum(tau, 1e-6)
    return best_plan
```

- Every iteration: construct, price each ant's plan (with lesson 06's `plan_cost`), and 2-opt only the
  iteration's best (polishing all would be slow). Keep the best plan ever seen.
- Evaporate, then deposit 1/L on each ant's edges. **`np.add.at`**, not `tau[i, j] += …`: if a plan uses
  the same edge twice (e.g. depot → 3 twice), plain fancy-index `+=` would add only once, while
  `np.add.at` adds every time.
- The elitist deposit on the best plan. A floor of 1e-6 so that no edge's pheromone ever reaches 0 (which
  would forbid it forever).

### `scripts/tune_aco.py`: fair tuning on validation

<!-- quote: scripts/tune_aco.py -->
```python
"""Pick ACO's (alpha, beta, ants) on VALIDATION days so ACO is a fair, tuned opponent.

    uv run python scripts/tune_aco.py --n_customers 20 --n_days 100
Writes results/aco_params.json, which evaluate.py then uses.
"""
import argparse
import json
from pathlib import Path

from svrp.baselines.aco import PARAM_GRID
from svrp.baselines.runner import run_planner
from svrp.config import EnvConfig, add_args, from_args
from svrp.data import World


def main():
    p = argparse.ArgumentParser()
    add_args(p, EnvConfig)
    p.add_argument("--n_days", type=int, default=100)
    a = p.parse_args()
    cfg = from_args(a, EnvConfig)
    batch = World(cfg).eval_set("val", a.n_days)

    best = None
    for alpha, beta, ants in PARAM_GRID:
        kw = {"alpha": alpha, "beta": beta, "n_ants": ants}
        out = run_planner("aco", batch, cfg.delivery, **kw)
        cost = out["cost"].mean()
        print(f"alpha={alpha:2d} beta={beta:2d} ants={ants:2d}:  val cost {cost:.4f}  "
              f"({out['sec_per_instance']*1000:.0f} ms/day)")
        if best is None or cost < best[0]:
            best = (cost, kw)
    out = Path(__file__).resolve().parents[1] / "results" / "aco_params.json"
    out.write_text(json.dumps(best[1]))
    print(f"best {best[1]} -> {out}")


if __name__ == "__main__":     # required on Windows: worker processes re-import this file
    main()
```

The whole script, top to bottom:

- **Docstring:** how to run it and what it writes.
- **Imports:** the grid, the runner, the config tools and the world.
- **`main()`, the arguments:** every `EnvConfig` field is a flag (so you can tune for another world,
  e.g. `--fill_rate 0.2`), plus `--n_days` (100 validation days is enough to rank 6 settings, and keeps
  tuning fast).
- **The loop:** for each (α, β, ants), run ACO through the runner on the **validation** days (never test),
  print the mean cost and speed, and remember the best. `best` starts as `None`, so the first setting
  always becomes the current best.
- **Saving:** the winning settings are written as JSON to `results/aco_params.json`, which `evaluate.py`
  loads automatically.
- **The guard:** the comment marks the bug from lesson 07. The first version of this script had no
  `__main__` guard and crashed with `BrokenProcessPool`.

## Rebuild it yourself

1. `two_opt`. Test it on a crossed 4-customer route you draw on paper.
2. `_construct`. Check that every ant visits every customer exactly once and respects capacity.
3. `aco`. Then add the `aco` import and branch back into your `runner.py` (lesson 07).
4. `scripts/tune_aco.py` (with `main()` + the guard).

## Verify

Restore the `aco` lines in your copy of `tests/test_baselines.py` and run:

```bash
uv run python -m pytest tests/test_baselines.py -q
uv run python -m scripts.tune_aco --n_customers 20 --n_days 100
```

The project's tuning result (`LOG.md`, E0):

```
alpha= 3 beta= 4 ants=20:  val cost 16.1777
alpha= 6 beta= 8 ants=24:  val cost 16.5883
alpha= 4 beta= 9 ants=15:  val cost 16.6286
alpha= 3 beta=10 ants=12:  val cost 16.4947
alpha= 1 beta= 2 ants=20:  val cost 15.7491
alpha= 1 beta= 5 ants=20:  val cost 16.2476
best {'alpha': 1, 'beta': 2, 'n_ants': 20}
```

The textbook setting beat all four of the paper's. Then ACO on the 1,000 test days: **8.102 at N = 10**
(+1.0% vs OR-Tools) and **15.006 at N = 20** (+3.3%).

## Check your understanding

1. What happens with α = 0? With β = 0?
2. Why evaporate at all?
3. Why `np.add.at` instead of `tau[p[:-1], p[1:]] += 1/L`?
4. The paper says RL beats ACO by 3.4%; here RL beats ACO by ~25% at N = 10. Give two reasons why such
   numbers don't transfer between papers.

**Answers** (try the questions first, then check here)

1. α = 0: the pheromone is ignored, so every ant is just a randomised greedy nearest-neighbour, and no
   learning happens. β = 0: costs are ignored, so ants start out choosing randomly and rely only on
   pheromone, which converges slowly and badly.
2. Without it, early (often bad) solutions keep their pheromone forever and the colony can't change its mind.
3. Fancy-index `+=` writes each repeated index once, while `np.add.at` accumulates duplicates correctly.
4. Different implementations and tuning of the baseline, and a different world: noise level, fill rate
   and demand range all change how much adapting is worth (see `LOG.md`, E1/E2).

## Read more

- Dorigo & Stützle, *Ant Colony Optimization* (MIT Press, 2004), chapters 1–3.
- Bullnheimer, Hartl & Strauss (1999), "An improved Ant System algorithm for the Vehicle Routing Problem",
  *Annals of Operations Research* 89: ACO with elitist ants for the CVRP.

**You've finished the course.** Go back to the [README](README.md) layer map. You have now built every box
in it. A good next step is to design your own experiment (a new external variable, a bigger N, or a new
baseline), predict its result in your log, and then run it.
