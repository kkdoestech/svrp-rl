# 07 — Reality check: executing plans with recourse (`svrp/env/executor.py`, `svrp/baselines/runner.py`)

| | |
|---|---|
| **Layer** | 5 |
| **Files covered** | `svrp/env/executor.py` (66 lines), `svrp/baselines/runner.py` (70), `tests/test_baselines.py` (part) |
| **You build** | `mysvrp/env/executor.py`, `mysvrp/baselines/runner.py` |
| **Time** | ~3 hours (+ a few minutes of computer time for the milestone) |
| **Before this** | [06 — Classical solvers](06_plans_and_classical_solvers.md); `a.md` section 1 (the R(x) "oops" penalty) |

## Why this layer exists

Lesson 06's planners produce beautiful plans **on paper**. But the truck drives in the real world, where
customer 7 wants 20.4 units instead of the 10.0 we believed. This layer answers: *what does the plan
actually cost when reality differs?*

That missing piece is the **recourse** term R(x) from your notes (`a.md`): "if a truck arrives and the
customer wants more than expected, it must drive back to the depot, reload, and come back." The
executor simulates exactly that, for every plan, on every true day.

This is also the layer where you run your **first real experiment**. It shows *why the paper exists*:
planning in the morning from estimates loses about a third of the money to uncertainty.

## Concepts from zero

### The recourse rules ("detour-to-depot")

Driving a plan like `[0, 3, 5, 0, 2, 0]` on the real day:

1. **Arrive at a customer with enough load**: deliver everything, continue.
2. **Arrive with too little load** (a *failure*, a *recourse event*): hand over what's on the truck, then
   shuttle depot ↔ customer (a round trip each time) until the order is complete. Then continue the plan.
3. **The truck is empty** before the next planned customer: go to the depot first, refill, then continue.
4. **A planned customer has already been fully served** (a plan can list someone twice): skip it.
5. **Full delivery** (indivisible orders): on failure, nothing is handed over. One round trip refills,
   then the whole order is delivered.

### A tiny example, by hand

Three nodes, every trip costs 1, Q = 20. True demands: customer 1 wants 15, customer 2 wants 10. The plan
believed both fit in one trip: `[0, 1, 2, 0]`.

| step | action | cost so far | load |
|---|---|---|---|
| 0 → 1 | deliver 15 | 1 | 5 |
| 1 → 2 | only 5 on board, but 10 needed: **failure** | 2 | hand over 5, so 0 |
| 2 → 0 → 2 | shuttle: refill, come back, deliver the last 5 | 4 | 15 |
| 2 → 0 | the plan says go home | **5** | |

The plan "cost" 3 on paper, and **5** on the road, with 1 recourse event. `tests/test_baselines.py` checks
exactly this example.

### Hindsight: the impossible opponent

What if the planner knew the true demands and true costs in advance? Run OR-Tools on the *truth*. That's
**hindsight**: no real dispatcher can do it, but it shows **how much money uncertainty costs**. The gap
between "plan from estimates" and "plan from the truth" is the prize an adaptive method can try to win.

## Code walkthrough

### `executor.py`, lines 1–13: the header

<!-- quote: svrp/env/executor.py -->
```python
"""Execute an a-priori plan on the real day ("plan in the morning, then drive it no matter what").

A plan is a node sequence like [0, 3, 5, 0, 2, 0] made from *estimates*. Reality then differs:
  * arrive with too little load  -> detour-to-depot recourse: round trip, refill, finish the customer
  * truck is empty mid-route      -> go back to the depot before the next planned customer
  * customer already fully served -> skip it
This is the classical "recourse" policy from the SVRP literature (the R(x) term in a.md).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6
```

The docstring lists the recourse rules in plain words, and they are exactly the rules the code
implements, in the same order. The executor is plain Python + NumPy: it handles one day at a time, and
its job is to be *obviously correct*, not fast. `EPS` is the floating-point tolerance explained below.

### `executor.py`, lines 16–37: driving the plan

<!-- quote: svrp/env/executor.py -->
```python
def execute(plan, demand: np.ndarray, cost: np.ndarray, capacity: float, delivery: str = "partial"):
    """Returns (total_cost, n_fail). demand: (N+1,), cost: (N+1, N+1), both TRUE values."""
    rem = demand.astype(np.float64).copy()
    pos, load, total, n_fail = 0, capacity, 0.0, 0
    for node in plan:
        node = int(node)
        if node == 0:
            if pos != 0:
                total += cost[pos, 0]
                pos, load = 0, capacity
            continue
```

- `rem` = remaining demand per node. It's a **copy**, so the caller's array is never modified (a
  function that silently changes its inputs causes very confusing bugs).
- State: the truck is at `pos` with `load` units, the cost paid so far is `total`, and `n_fail` counts
  recourse events.
- A depot entry in the plan: drive home (unless already there) and refill. Two zeros in a row cost
  nothing extra.

<!-- quote: svrp/env/executor.py -->
```python
        if rem[node] <= EPS:
            continue
        if load <= EPS:                          # empty truck: restock before driving on
            total += cost[pos, 0]
            pos, load = 0, capacity
        total += cost[pos, node]
        pos = node
        if load >= rem[node] - EPS:
            load -= rem[node]
            rem[node] = 0.0
            continue
```

- Rule 4: an already-served customer is skipped.
- Rule 3: the truck is empty, so restock first.
- Drive to the customer and pay the **true** cost of that road.
- Rule 1: enough load, so deliver everything.

**What is `EPS = 1e-6`?** Floating-point arithmetic is slightly inexact: 20 − 15.3 − 4.7 might give
0.0000000001 instead of 0. Comparing with a tiny tolerance treats such crumbs as zero. Without it, the
truck would sometimes "fail" at a customer who needs 0.0000000001 more units.

### `executor.py`, lines 38–55: recourse

<!-- quote: svrp/env/executor.py -->
```python
        n_fail += 1                              # recourse
        if delivery == "partial":
            rem[node] -= load                    # hand over what we have,
            while rem[node] > EPS:               # then shuttle to the depot until the order is complete
                total += cost[node, 0] + cost[0, node]
                give = min(capacity, rem[node])
                rem[node] -= give
                load = capacity - give
        else:
            total += cost[node, 0] + cost[0, node]   # indivisible: go refill, come back, deliver it all
            load = capacity - rem[node]
            rem[node] = 0.0
```

- Rule 2 (partial): hand over the whole load, then **loop** round trips. A single order can be bigger
  than one truck (21.3 > 20), so it might take several shuttles. Each trip brings Q and delivers what is
  still needed. `load` ends up as whatever is left on the truck.
- Rule 5 (full): one round trip, then deliver it all. (Under full delivery, demands are ≤ Q, see the
  generator's `clip_demand`, so one shuttle is always enough.)

<!-- quote: svrp/env/executor.py -->
```python
    if pos != 0:
        total += cost[pos, 0]
    if (rem[1:] > EPS).any():
        missing = np.flatnonzero(rem[1:] > EPS) + 1
        raise ValueError(f"plan does not visit customers {missing.tolist()}")
    return total, n_fail
```

- End of the day: drive home.
- **Safety check:** if a plan forgot a customer, stop with a clear error. A planner bug then can't hide
  as "a suspiciously cheap plan". (`np.flatnonzero` gives the positions of the True values; `+ 1`
  because we skipped the depot.)

### `executor.py`, lines 58–66: many days

<!-- quote: svrp/env/executor.py -->
```python
def execute_many(plans, demand: np.ndarray, cost: np.ndarray, capacity: float, delivery: str = "partial"):
    """plans: one plan per day, or a single plan reused for every day. Returns arrays (costs, n_fail)."""
    B = demand.shape[0]
    single = len(plans) > 0 and np.isscalar(plans[0])
    out = np.empty(B)
    fails = np.empty(B)
    for b in range(B):
        out[b], fails[b] = execute(plans if single else plans[b], demand[b], cost[b], capacity, delivery)
    return out, fails
```

`plans` is either *one* plan (a list of numbers, reused every day) or *a list of plans* (one per day).
`np.isscalar(plans[0])` tells them apart: is the first element a number, or a list? A plain Python loop
over 1,000 days is fast enough here (~40 steps each).

### `runner.py`, lines 1–23: the header and the list of planners

<!-- quote: svrp/baselines/runner.py -->
```python
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
```

- The docstring's two lines, `plan = planner(estimates)` and `score = execute(plan, truth)`, are the whole
  idea of a-priori evaluation.
- `ProcessPoolExecutor` runs work in several Python *processes* (explained below).
- It imports every planner, `Batch`, and `execute_many`. **For now, leave out the `aco` import** (lesson 14).
- `PLANNERS` is a tuple of the names this runner understands. `evaluate.py` checks `method in PLANNERS` to
  decide whether to send a method here.

### `runner.py`, lines 26–42: one interface for every planner

<!-- quote: svrp/baselines/runner.py -->
```python
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
```

- One function maps a planner's **name** to the call, and times it.
- `hindsight` is OR-Tools GLS; the only difference is *what it's given* (the truth, see below).
- `_solve_star` unpacks a tuple of arguments. `ProcessPoolExecutor.map` passes one object per job, so
  each job is a tuple.
- **For now, leave out the `aco` import and branch.** ACO is the optional lesson 14.

### `runner.py`, lines 45–70: plan from beliefs, score on truth, in parallel

<!-- quote: svrp/baselines/runner.py -->
```python
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
```

**The most important four lines of the classical side:** planners get `dem_hat`/`cost_hat` (the beliefs),
and only `hindsight` gets the truth. Every number goes to NumPy in float64 (`.double()`). `setdefault`
adds the default time limit unless the caller chose one.

<!-- quote: svrp/baselines/runner.py -->
```python
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
```

- **Shortcut:** if every day has the *same* belief (constant estimates on a fixed city ignore the
  weather), one plan serves all days. Solve once instead of 1,000 times.
- Otherwise, one job per day, run in parallel **processes**. OR-Tools uses one CPU core per solve, so 6
  workers on a 6-core i5 solve 6 days at a time. The default `os.cpu_count() // 2` = 12 threads / 2 = 6.
- `chunksize` sends jobs to workers in groups, which cuts communication overhead.
- Then every plan is executed on its true day.

**Why the `if __name__ == "__main__":` guard matters here (Windows).** On Windows a new process starts
a *fresh* Python, which **re-imports your script** to find `_solve_star`. If your script's top level
starts an evaluation, each worker starts its own evaluation, which starts workers, and so on: a crash
(`BrokenProcessPool`). This exact crash happened while building the project (`tune_aco.py`'s first
version). Every script that uses the runner keeps its work inside `main()` behind the guard.

## Rebuild it yourself

1. `mysvrp/env/executor.py`: `execute`, then `execute_many`. Check the hand example above first.
2. `mysvrp/baselines/runner.py` without ACO: `PLANNERS`, `solve`, `_solve_star`, `run_planner`.

## Verify

**Unit check.** From `learning/workspace`:

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from mysvrp.env.executor import execute
cost = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
dem = np.array([0, 15, 10], dtype=float)
print(execute([0, 1, 2, 0], dem, cost, 20.0, 'partial'), execute([0, 1, 2, 0], dem, cost, 20.0, 'full'))
"
```

Expected: `(5.0, 1) (5.0, 1)` (numerically 5.0 might print as `np.float64(5.0)`).

**Day 0 on the road.** Lesson 06's two plans, now driven on day 0's truth. Save as `check07.py`:

```python
import sys
sys.path.insert(0, ".")
from mysvrp.config import EnvConfig
from mysvrp.data import World
from mysvrp.baselines.common import plan_cost
from mysvrp.baselines.clarke_wright import clarke_wright
from mysvrp.baselines.ortools_solver import ortools_cvrp
from mysvrp.env.executor import execute

b = World(EnvConfig(n_customers=10)).eval_set("test", 1000)
dh, ch = b.dem_hat[0].double().numpy(), b.cost_hat[0].double().numpy()   # belief
d, c = b.demand[0].double().numpy(), b.cost[0].double().numpy()          # truth
for name, plan in [("cw", clarke_wright(dh, ch, 20.0)), ("ortools", ortools_cvrp(dh, ch, 20.0, time_limit=0.5))]:
    tot, f = execute(plan, d, c, 20.0)
    print(f"{name:8s} planned {plan_cost(plan, ch):.3f}  executed {tot:.3f}  recourse events {f}")
hp = ortools_cvrp(d, c, 20.0, time_limit=0.5)
print("hindsight executed %.3f" % execute(hp, d, c, 20.0)[0])
```

Reference:

```
cw       planned 6.517  executed 12.636  recourse events 3
ortools  planned 6.202  executed 12.304  recourse events 3
hindsight executed 10.864
```

Day 0 is a heavy day (the true total demand is 107 units, about 5.4 full trucks). The plans nearly
**double** in cost on the road. Even hindsight pays 10.86, because that much demand simply needs many
trips.

**Milestone: the baseline table (1,000 test days).** Save as `milestone07.py`. The `main()` + guard is
required on Windows:

```python
import sys
sys.path.insert(0, ".")
from mysvrp.config import EnvConfig
from mysvrp.data import World
from mysvrp.baselines.runner import run_planner


def main():
    cfg = EnvConfig(n_customers=10)
    batch = World(cfg).eval_set("test", 1000)
    for name in ["cw", "ortools_gls", "hindsight"]:
        out = run_planner(name, batch, cfg.delivery)
        print(f"{name:12s} mean cost {out['cost'].mean():.4f}   recourse/day {out['n_fail'].mean():.2f}")


if __name__ == "__main__":
    main()
```

It takes ~3 minutes (OR-Tools: 0.5 s × 1,000 days ÷ 6 workers, twice). Compare with `experiments/LOG.md`, E1:

```
cw           mean cost 8.0010   recourse/day 2.04
ortools_gls  mean cost 8.0198   recourse/day 2.09
hindsight    mean cost 6.0364   recourse/day 1.21
```

CW must match exactly if your world is identical (lesson 05 fingerprint). OR-Tools can differ in the
3rd decimal (time-limited search). Then **the finding**: planning from estimates costs **33% more than
hindsight**, and the clever optimiser (OR-Tools) is no better than the simple one (CW). In this world,
not knowing the day hurts far more than imperfect optimising. That's the opening RL will use.

(Hindsight still shows recourse events: customers whose true order is bigger than a whole truck need
shuttles even with perfect knowledge.)

The real test file `tests/test_baselines.py` also imports the environment and `nn_online` (lesson 08) and
`aco` (lesson 14), so it can't even be imported yet. The unit check, day 0 and the milestone above are
your tests for now. You'll run `test_baselines.py` at the end of lesson 08.

## Common mistakes

- **Modifying the caller's demand array** (forgetting `.copy()`): the second plan you evaluate sees
  already-served customers and looks impossibly cheap.
- **One shuttle only** for orders bigger than a truck: the safety check raises "plan does not visit
  customers [...]".
- **Hindsight given the beliefs** (or a planner given the truth): your table looks wrong in a way that
  is easy to miss. Check that `hindsight` < `ortools_gls` by roughly 25%.
- **No `__main__` guard** in the milestone script on Windows: `BrokenProcessPool`.

## Check your understanding

1. In the hand example, what does the plan `[0, 2, 1, 0]` cost on the road? (Same demands, Q = 20.)
2. Why is hindsight not a fair opponent, and why do we still compute it?
3. `cw` planned 6.517 for day 0 but paid 12.636. Name the two separate reasons for the difference.
4. Why are the workers *processes* and not *threads*?

**Answers** (try the questions first, then check here)

1. 0 → 2: deliver 10, load 10. 2 → 1: needs 15, has 10, so a failure: hand over 10, shuttle 1 → 0 → 1
   (cost 2) and deliver 5. 1 → 0. Total 1 + 1 + 2 + 1 = **5**, also 1 failure. Here the order didn't
   matter because every road costs 1. With real distances it often does.
2. It uses information nobody has in the morning. It measures how much uncertainty costs, which is the
   maximum an adaptive method could win back (under no-split rules).
3. (a) Recourse: true demands were larger than believed, so there were extra depot trips.
   (b) True travel costs differ from the estimated `cost_hat`. The plan is priced at the real road costs.
4. Python threads can't run Python/OR-Tools work truly in parallel because of the GIL (global interpreter
   lock). Separate processes each have their own interpreter and core.

## Read more

- Gendreau, Laporte & Séguin (1996), "Stochastic vehicle routing", *European Journal of Operational Research*
  88(1). The classic survey; read section 2 on a-priori optimisation and recourse.
- Python docs, `concurrent.futures.ProcessPoolExecutor`, and the "Safe importing of main module" note
  in the `multiprocessing` docs.

**Next:** [08 — The environment (MDP)](08_environment.md)
