# 06 — Plans and classical solvers (`svrp/baselines/common.py`, `clarke_wright.py`, `ortools_solver.py`)

| | |
|---|---|
| **Layer** | 4 |
| **Files covered** | `svrp/baselines/common.py` (44 lines), `svrp/baselines/clarke_wright.py` (44), `svrp/baselines/ortools_solver.py` (65) |
| **You build** | the same three files in `mysvrp/baselines/` |
| **Time** | 3–4 hours |
| **Before this** | [05 — Estimates and data](05_estimates_and_data.md); `part 6.md` ("The Competitors") |

## Why this layer exists

To claim "RL is better", you need opponents. The paper's opponents are classical methods that **plan
the whole day in the morning** from the estimates, then stick to the plan (an *a-priori* plan, `part 6.md`).
This lesson builds the planners. The next lesson builds the simulator that drives their plans on the
real, random day.

You already know the strongest one: **Google OR-Tools**. This lesson connects the OR-Tools CVRP example
you learned to the code here.

## Concepts from zero

### The plan format: one list of nodes

The project has **one truck**, which can return to the depot to refill and leave again. A whole day is
one list:

```
[0, 3, 5, 0, 2, 0]   =  depot → 3 → 5 → depot (refill) → 2 → depot
```

Each segment between depot visits is a **trip** (a *route*). `[[3, 5], [2]]` is the same day as a list
of routes. In OR-Tools language each route is a "vehicle". One truck doing 3 trips costs the same as 3
trucks doing 1 trip each, since nobody pays for a truck standing still. So we let OR-Tools use as many
"vehicles" as needed, then put their routes one after another.

### Clarke–Wright savings, with numbers

Start with the laziest plan: one trip per customer, depot → i → depot. Then ask, for each pair: *if
the truck goes i → j directly instead of returning home in between, how much do we save?*

```
saving s_ij = c_i0 + c_0j − c_ij
             (drive home from i) + (drive out to j) − (drive i → j directly)
```

Toy example: depot 0, three customers, each with demand 5, truck capacity Q = 10:

| cost | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 0 | 4 | 4 | 5 |
| **1** | 4 | 0 | 2 | 7 |
| **2** | 4 | 2 | 0 | 3 |
| **3** | 5 | 7 | 3 | 0 |

Savings: s₁₂ = 4 + 4 − 2 = **6**, s₂₃ = 4 + 5 − 3 = **6**, s₁₃ = 4 + 5 − 7 = **2**.

1. Highest first: s₁₂ = 6 → merge into trip [1, 2] (load 5 + 5 = 10 ≤ 10 ✓).
2. s₂₃ = 6 → adding 3 to trip [1, 2] would make load 15 > 10 ✗. Skip.
3. s₁₃ = 2 → 1 is inside a trip that is already full ✗. Skip.

Result `[0, 1, 2, 0, 3, 0]`, cost 4 + 2 + 4 + 5 + 5 = **20** (instead of 8 + 8 + 10 = 26: saved 6 ✓). The
real function gives exactly this answer.

It's greedy: it never undoes a merge. That makes it very fast (0.1 ms) but not optimal.

### OR-Tools: from the tutorial you know to this code

| Google's CVRP tutorial | this project |
|---|---|
| `create_data_model()` with `distance_matrix`, `demands`, `vehicle_capacities` | the estimates `cost_hat`, `dem_hat` and Q |
| `RegisterTransitCallback(distance_callback)` (a Python function) | `RegisterTransitMatrix(c_int)`: give the whole matrix at once (much faster: no Python call per arc) |
| `RegisterUnaryTransitCallback(demand_callback)` | `RegisterUnaryTransitVector(d_int)` |
| `AddDimensionWithVehicleCapacity(..., "Capacity")` | the same |
| `PATH_CHEAPEST_ARC` + `GUIDED_LOCAL_SEARCH` + time limit | the same, plus a `TABU_SEARCH` option |
| `print_solution()` walks `NextVar` | the same walk, collecting routes into a plan |

Two OR-Tools facts matter here:

- **It needs integers.** Distances like 0.4137 are multiplied by 1,000 and rounded (413). Demands are
  ×100.
- **Metaheuristics** escape bad local solutions. *Guided Local Search* (GLS) penalises features of the
  current solution that keep showing up. *Tabu Search* forbids recently undone moves. The paper's
  "Tabu Search" opponent is OR-Tools' `TABU_SEARCH` here.

## Code walkthrough

### `common.py`: converting and pricing plans

<!-- quote: svrp/baselines/common.py -->
```python
"""Helpers shared by the classical planners."""
from __future__ import annotations

import numpy as np
```

A one-line docstring is enough for a small helper file. It needs only NumPy: the classical side of the
project works on NumPy arrays (OR-Tools and these loops don't use tensors).

<!-- quote: svrp/baselines/common.py -->
```python
def routes_to_plan(routes: list[list[int]]) -> list[int]:
    """[[3, 5], [2]] -> [0, 3, 5, 0, 2, 0]"""
    plan = [0]
    for r in routes:
        if r:
            plan.extend(int(i) for i in r)
            plan.append(0)
    return plan
```

Start at the depot. For each **non-empty** route (OR-Tools returns empty routes for unused vehicles;
`if r:` skips them), add its customers and then a depot visit. `int(i)` turns NumPy integers into plain
Python ints, which print cleanly.

<!-- quote: svrp/baselines/common.py -->
```python
def plan_to_routes(plan) -> list[list[int]]:
    routes, cur = [], []
    for node in plan:
        if node == 0:
            if cur:
                routes.append(cur)
            cur = []
        else:
            cur.append(int(node))
    if cur:
        routes.append(cur)
    return routes
```

The reverse: walk the plan, start a new route at every 0, and keep the non-empty ones. The final
`if cur:` handles a plan that doesn't end with 0.

<!-- quote: svrp/baselines/common.py -->
```python
def plan_cost(plan, cost: np.ndarray) -> float:
    """Cost of driving the plan exactly as written (no recourse)."""
    p = list(plan)
    if p[0] != 0:
        p = [0] + p
    if p[-1] != 0:
        p = p + [0]
    p = np.asarray(p)
    return float(cost[p[:-1], p[1:]].sum())
```

- Make sure the plan starts and ends at the depot.
- `p[:-1]` = every node except the last ("from"), and `p[1:]` = every node except the first ("to").
  `cost[from, to]` picks all the arcs at once (NumPy pairs the two index arrays element by element, like
  in lesson 02 section 5), then we sum them.
- Example: `p = [0, 3, 5, 0]` → arcs (0,3), (3,5), (5,0).
- "No recourse" means this is the price **if everything goes as planned**. Lesson 07 computes the price
  when it doesn't.

<!-- quote: svrp/baselines/common.py -->
```python
def planning_demand(dem: np.ndarray, capacity: float) -> np.ndarray:
    """Planners can't handle orders bigger than a truck; clip (the executor's recourse covers the rest)."""
    return np.minimum(dem, capacity)
```

A customer believed to need 21.3 units with Q = 20 can't be put on *any* CVRP route. That would make
the CVRP infeasible. So planners see at most Q. Reality still charges the full order, and lesson 07's
executor handles the extra with depot shuttles.

### `clarke_wright.py`, lines 1–12: the header

<!-- quote: svrp/baselines/clarke_wright.py -->
```python
"""Clarke–Wright savings (1964): start with one route per customer, then greedily merge routes.

Saving of joining i -> j into one route instead of two separate depot trips:
    s_ij = c_i0 + c_0j − c_ij
Merge pairs in decreasing order of saving, as long as i ends one route, j starts another,
and the merged route still fits in the truck.
"""
from __future__ import annotations

import numpy as np

from svrp.baselines.common import planning_demand, routes_to_plan
```

The docstring *is* the algorithm in three sentences, including the savings formula. If you can write
this docstring from memory, you understand Clarke–Wright. It imports the two helpers it needs from
`common.py`.

### `clarke_wright.py`, lines 15–44

<!-- quote: svrp/baselines/clarke_wright.py -->
```python
def clarke_wright(dem: np.ndarray, cost: np.ndarray, capacity: float) -> list[int]:
    N1 = len(dem)
    d = planning_demand(dem, capacity)
    route_of = np.arange(N1)                      # route id of each customer (id = its first customer)
    routes = {i: [i] for i in range(1, N1)}
    load = {i: d[i] for i in range(1, N1)}
```

The data structures of the lazy start:

- `routes`: a dict *route id → list of customers*. At first each customer has its own route, whose id is
  the customer's number.
- `load`: route id → total (planned) demand.
- `route_of[c]`: which route customer c is currently in. When routes merge, the survivor keeps its id,
  and the absorbed customers are re-labelled.

<!-- quote: svrp/baselines/clarke_wright.py -->
```python
    i_idx, j_idx = np.where(~np.eye(N1, dtype=bool))
    keep = (i_idx > 0) & (j_idx > 0)
    i_idx, j_idx = i_idx[keep], j_idx[keep]
    s = cost[i_idx, 0] + cost[0, j_idx] - cost[i_idx, j_idx]
    order = np.argsort(-s, kind="stable")
```

All savings at once, with NumPy:

- `np.eye(N1)` is the identity matrix (True on the diagonal). `~` flips it, so `np.where` returns all
  pairs (i, j) with i ≠ j, as two arrays.
- Keep only customer pairs (drop the depot, node 0).
- `s` holds every saving in one vectorised line. Ordered pairs are kept, so i → j and j → i are separate
  candidates: with asymmetric costs they could differ.
- `np.argsort(-s)` sorts from the largest saving down. `kind="stable"` breaks ties by the original order,
  so the result is deterministic.

<!-- quote: svrp/baselines/clarke_wright.py -->
```python
    for k in order:
        if s[k] <= 0:
            break
        i, j = int(i_idx[k]), int(j_idx[k])
        ri, rj = int(route_of[i]), int(route_of[j])
        if ri == rj or load[ri] + load[rj] > capacity + 1e-9:
            continue
        a, b = routes[ri], routes[rj]
        if a[-1] != i or b[0] != j:               # i must end its route, j must start its route
            continue
        merged = a + b
        del routes[rj]
        routes[ri] = merged
        load[ri] += load.pop(rj)
        for c in b:
            route_of[c] = ri
    return routes_to_plan(list(routes.values()))
```

- Savings ≤ 0 mean merging *costs* money. Because the list is sorted, everything after is worse, so stop.
- Skip if i and j are already together, or the merged load won't fit. (`+ 1e-9` forgives a
  floating-point hair: 10.000000001 should still count as "fits in 10".)
- **The end-point rule:** we can only glue route a to route b at the joint "…i] + [j…". So i must be
  the *last* stop of its route, and j the *first* stop of its route. A customer in the middle of a route
  can't be connected to anything new.
- Merge: append b to a, delete b, add the loads (`load.pop(rj)` returns b's load *and* removes the entry),
  and re-label b's customers.
- Finally turn the surviving routes into a plan.

### `ortools_solver.py`, lines 1–14: the header

<!-- quote: svrp/baselines/ortools_solver.py -->
```python
"""Google OR-Tools CVRP on the *estimated* demands and costs (the classical a-priori approach).

This is the tool you already know: it plans as if the estimates were the truth. The executor then
drives the plan on the real day and pays for recourse when reality differs.
Two metaheuristics: Guided Local Search (usually OR-Tools' strongest) and Tabu Search (the paper's TS).
"""
from __future__ import annotations

import math

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from svrp.baselines.common import planning_demand, routes_to_plan
```

- The docstring says what makes this "a-priori": it plans *as if the estimates were the truth*.
- `pywrapcp` is OR-Tools' Python wrapper for the routing solver (the same import as in Google's tutorial).
  `routing_enums_pb2` holds the named options (first-solution strategies, metaheuristics).
- `math` is for `math.ceil` in the vehicle count.

### `ortools_solver.py`, lines 16–26: settings

<!-- quote: svrp/baselines/ortools_solver.py -->
```python
COST_SCALE = 1000      # OR-Tools needs integers: costs in 1/1000 units
DEMAND_SCALE = 100

METAHEURISTICS = {
    "gls": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    "tabu": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
}


def default_time_limit(n_customers: int) -> float:
    return 0.5 if n_customers <= 10 else 1.0 if n_customers <= 20 else 3.0
```

- The scales turn real numbers into OR-Tools integers. ×1000 keeps 3 decimals of cost, which is plenty.
- The dict maps short names to OR-Tools' enum values, so callers write `metaheuristic="tabu"`.
- Time limits grow with problem size: 0.5 s at N ≤ 10, 1 s at N ≤ 20, 3 s above. On 1,000 test days × 6
  CPU workers, that is ~1.5 to 3 minutes per evaluation.

### `ortools_solver.py`, lines 29–51: building the model

<!-- quote: svrp/baselines/ortools_solver.py -->
```python
def ortools_cvrp(dem: np.ndarray, cost: np.ndarray, capacity: float, time_limit: float = 1.0,
                 metaheuristic: str = "gls", return_objective: bool = False):
    N1 = len(dem)
    d = planning_demand(dem, capacity)
    d_int = [int(round(x * DEMAND_SCALE)) for x in d]
    cap_int = int(round(capacity * DEMAND_SCALE))
    c_int = np.rint(cost * COST_SCALE).astype(np.int64).tolist()
    n_veh = min(N1 - 1, 2 * math.ceil(d[1:].sum() / capacity) + 1)
```

- Demands and capacity become integers in hundredths of a unit. Costs become thousandths, via
  `np.rint` (round the whole matrix), `astype(np.int64)`, then `.tolist()`, because OR-Tools wants plain
  Python lists.
- **How many "vehicles" (trips) to offer?** At least `ceil(total demand / Q)` trips are needed. Bin
  packing can need more (demands don't split nicely), so we allow 2× that + 1, but never more than one
  trip per customer (N1 − 1). Too few would be infeasible. Too many makes OR-Tools slower.

<!-- quote: svrp/baselines/ortools_solver.py -->
```python
    manager = pywrapcp.RoutingIndexManager(N1, n_veh, 0)
    routing = pywrapcp.RoutingModel(manager)
    arc = routing.RegisterTransitMatrix(c_int)
    routing.SetArcCostEvaluatorOfAllVehicles(arc)
    dem_cb = routing.RegisterUnaryTransitVector(d_int)
    routing.AddDimensionWithVehicleCapacity(dem_cb, 0, [cap_int] * n_veh, True, "load")
```

The same steps as the tutorial:

- `RoutingIndexManager(nodes, vehicles, depot)` translates between *node numbers* (our 0…N) and OR-Tools'
  internal *indices*.
- `RegisterTransitMatrix` registers the cost of every arc. `SetArcCostEvaluatorOfAllVehicles` makes it
  the objective: minimise the total arc cost.
- `RegisterUnaryTransitVector` registers each node's demand. The `"load"` dimension adds it up along each
  route, with slack 0, capacity `cap_int` for every vehicle, and `True` = start every route at load 0.

<!-- quote: svrp/baselines/ortools_solver.py -->
```python
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = METAHEURISTICS[metaheuristic]
    params.time_limit.FromMilliseconds(int(time_limit * 1000))
    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("OR-Tools found no solution")
```

- First solution: `PATH_CHEAPEST_ARC` (greedy: keep extending the route with the cheapest arc).
- Then improve it with GLS or Tabu until the time limit. **A metaheuristic only stops at the time
  limit.** Without one, it would run forever.
- `None` means no feasible solution. With the generous `n_veh` that shouldn't happen, but if it did,
  failing loudly beats returning garbage.

### `ortools_solver.py`, lines 53–65: reading the answer

<!-- quote: svrp/baselines/ortools_solver.py -->
```python
    routes = []
    for v in range(n_veh):
        idx, r = routing.Start(v), []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                r.append(node)
            idx = sol.Value(routing.NextVar(idx))
        routes.append(r)
    plan = routes_to_plan(routes)
    if return_objective:
        return plan, sol.ObjectiveValue() / COST_SCALE
    return plan
```

For each vehicle, start at its start index and follow `NextVar` (the solver's answer to "where next?")
until the end, collecting customer nodes. Unused vehicles give `[]`, which `routes_to_plan` skips.
`return_objective=True` also returns OR-Tools' own total cost (converted back from integers). The tests
use it to prove that our executor prices plans exactly like OR-Tools does.

## Rebuild it yourself

1. `mysvrp/baselines/common.py`: the four helpers. Try `routes_to_plan([[3, 5], [], [2]])` → `[0, 3, 5, 0, 2, 0]`.
2. `mysvrp/baselines/clarke_wright.py`. Test it on the toy example above.
3. `mysvrp/baselines/ortools_solver.py`. If you remember the tutorial, write it from memory first, then
   compare.

## Verify

Save as `learning/workspace/check06.py` and run `uv run python check06.py` from `learning/workspace`:

```python
import sys
sys.path.insert(0, ".")
import numpy as np
from mysvrp.config import EnvConfig
from mysvrp.data import World
from mysvrp.baselines.common import plan_cost, plan_to_routes, planning_demand
from mysvrp.baselines.clarke_wright import clarke_wright
from mysvrp.baselines.ortools_solver import ortools_cvrp

# the toy example from this lesson
c = np.array([[0, 4, 4, 5], [4, 0, 2, 7], [4, 2, 0, 3], [5, 7, 3, 0]], float)
d = np.array([0, 5, 5, 5], float)
p = clarke_wright(d, c, 10.0)
print("toy", p, plan_cost(p, c))

# day 0 of the N=10 test set, planned on the BELIEFS
cfg = EnvConfig(n_customers=10)
b = World(cfg).eval_set("test", 1000)
dem, cost = b.dem_hat[0].double().numpy(), b.cost_hat[0].double().numpy()
for name, plan in [("cw", clarke_wright(dem, cost, cfg.capacity)),
                   ("ortools", ortools_cvrp(dem, cost, cfg.capacity, time_limit=0.5))]:
    print(name, plan, "planned cost %.3f" % plan_cost(plan, cost))
    print("   loads per route", [round(float(planning_demand(dem, cfg.capacity)[r].sum()), 1) for r in plan_to_routes(plan)])
```

Reference output:

```
toy [0, 1, 2, 0, 3, 0] 20.0
cw [0, 1, 0, 2, 0, 3, 6, 0, 5, 7, 0, 8, 0, 9, 0, 10, 4, 0] planned cost 6.517
   loads per route [20.0, 19.3, 14.9, 13.6, 11.5, 13.3, 16.5]
ortools [0, 4, 0, 5, 7, 0, 10, 3, 0, 1, 0, 6, 8, 0, 2, 0, 9, 0] planned cost 6.202
   loads per route [7.6, 13.6, 19.6, 20.0, 15.8, 19.3, 13.3]
```

- OR-Tools may find a slightly different plan on your PC (a metaheuristic stopped by a clock depends on
  CPU speed), but every route load must be ≤ 20, and its planned cost should be at or below CW's.
- Customer 1 is believed to need 21.3 > Q, so both planners give it a trip of its own (load clipped to 20.0).
- **Remember this:** on paper OR-Tools is 5% cheaper than CW here. In lesson 07 you'll see that on the
  *real* days the two end up tied. Better planning on wrong numbers buys little.

The full `tests/test_baselines.py` needs the next lesson's files. Run it at the end of lesson 07.

## Common mistakes

- **Passing floats to OR-Tools**: it raises a type error or silently truncates. Scale and round.
- **Forgetting `planning_demand`**: one believed order > Q makes the CVRP infeasible, and you get
  "no solution".
- **CW without the end-point check**: you get "routes" that branch in the middle, which aren't valid trips.
- **Too few vehicles** (`n_veh = ceil(sum/Q)`): sometimes infeasible because of bin packing.

## Check your understanding

1. Compute s₃₁ in the toy example. Why doesn't CW merge 3 with 1?
2. Why can one truck's whole day be modelled as several OR-Tools "vehicles"?
3. What would go wrong with `COST_SCALE = 1` for costs between 0 and 1.4?
4. Why does `plan_cost` give a lower number than what the truck will actually pay on a real day?

**Answers** (try the questions first, then check here)

1. s₃₁ = c₃₀ + c₀₁ − c₃₁ = 5 + 4 − 7 = 2 > 0. But by the time it's considered, 1 is in the full trip
   [1, 2] (load 10), so adding 3 would exceed Q.
2. Trips are independent apart from starting and ending at the depot, and an idle truck costs nothing.
   So "3 vehicles × 1 trip" and "1 truck × 3 trips" have the same total cost.
3. Every cost rounds to 0 or 1, so OR-Tools optimises nonsense.
4. It assumes the estimates are true. In reality some customers want more than believed, and the truck
   must make extra depot trips (recourse), which `plan_cost` doesn't include.

## Read more

- Google OR-Tools, *Capacitated Vehicle Routing Problem*: [developers.google.com/optimization/routing/cvrp](https://developers.google.com/optimization/routing/cvrp)
- OR-Tools, *Routing options* (first-solution strategies, metaheuristics, time limits): [developers.google.com/optimization/routing/routing_options](https://developers.google.com/optimization/routing/routing_options)
- Clarke & Wright (1964), "Scheduling of vehicles from a central depot to a number of delivery points",
  *Operations Research* 12(4). The original two-page idea.

**Next:** [07 — Executor and runner](07_executor_and_runner.md)
