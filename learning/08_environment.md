# 08 — The environment: the game the AI plays (`svrp/env/svrp_env.py`, `svrp/baselines/nn_online.py`)

| | |
|---|---|
| **Layer** | 6 |
| **Files covered** | `svrp/env/svrp_env.py` (116 lines), `svrp/baselines/nn_online.py` (33), `tests/test_env.py`, `tests/test_baselines.py` |
| **You build** | `mysvrp/env/svrp_env.py`, `mysvrp/baselines/nn_online.py` |
| **Time** | 4–6 hours |
| **Before this** | [05](05_estimates_and_data.md) (Batch), [07](07_executor_and_runner.md) (recourse); `part 5.md` part 1 (the inputs) |

## Why this layer exists

Lesson 07 drove a plan made in the morning. The paper's AI does something different: it **decides one
stop at a time**, looking at what it has learned so far ("customer 2 wanted way more than expected;
my truck is now empty"). That is *reoptimisation*, and `part 6.md` found it beats a-priori planning.

To learn by trial and error, the AI needs a **game**:

- a state it can look at,
- a set of legal moves,
- rules for what each move does,
- and a score.

In reinforcement learning this game is called the **environment**, and mathematically it's a **Markov
Decision Process (MDP)**. This file is that game, for 512 days at once on the GPU.

## Concepts from zero

### The MDP vocabulary, applied to delivery

| RL word | here |
|---|---|
| **state** | where the truck is, its load, what is still left to deliver, what has been revealed so far |
| **observation** | the part of the state the AI may see (not the hidden true demands) |
| **action** | the next node to drive to (0 = depot, 1…N = a customer) |
| **transition** | drive there, pay the road's true cost, deliver, update the load |
| **cost** (negative reward) | the true travel cost of each step. The episode's total is what we minimise |
| **episode** | one whole day: start full at the depot, end when everything is delivered and the truck is home |
| **policy** π | the decision rule: observation → action (later, the neural network) |
| **mask** | the list of legal actions in this state |

### Hidden until you arrive

This is what makes it *stochastic* VRP and not plain VRP: **a customer's true demand is revealed only
when the truck arrives**. Before that, the AI sees the estimate (`dem_hat`). After the first visit, it
sees the true *remaining* amount. Real drivers work the same way: the shop tells you what it needs when
you arrive.

### Batched simulation and `done`

All B days advance in lock-step: one `step(action)` call moves all B trucks, each to its own chosen
node. Days finish at different times. A finished day must stay frozen while the others continue. The
`done` flag (one per day) does that: finished days pay 0 and nothing changes.

### Play the game by hand (do this first!)

From `learning/workspace` after you've written the env, or from the project root with `svrp` right now:

```python
import sys; sys.path.insert(0, ".")
import torch
from svrp.config import EnvConfig
from svrp.data import World
from svrp.env.svrp_env import SVRPEnv
torch.set_printoptions(precision=1, sci_mode=False)
b = World(EnvConfig(n_customers=5)).eval_set("test", 1)
env = SVRPEnv(b)
print("true demand ", b.demand[0])
print("believed    ", env.observed_demand()[0])
for a in [3, 1, 0, 2, 0, 2, 4, 5, 0, 5, 0]:
    c = env.step(torch.tensor([a]))
    print(f"-> {a}: paid {c.item():.3f}  load {env.load.item():4.1f}  observed {env.observed_demand()[0]}  "
          f"mask {env.mask()[0].int().tolist()}  fails {int(env.n_fail.item())}  done {env.done.item()}")
```

Output (Q = 20):

```
true demand  tensor([ 0.0,  1.0, 37.2, 13.2,  2.4, 16.1])
believed     tensor([ 0.0,  3.4, 26.6,  6.3,  5.4, 20.5])
-> 3: paid 0.698  load  6.8  observed tensor([ 0.0,  3.4, 26.6,  0.0,  5.4, 20.5])  mask [1, 1, 1, 0, 1, 1]  fails 0  done False
-> 1: paid 1.068  load  5.8  observed tensor([ 0.0,  0.0, 26.6,  0.0,  5.4, 20.5])  mask [1, 0, 1, 0, 1, 1]  fails 0  done False
-> 0: paid 0.068  load 20.0  observed tensor([ 0.0,  0.0, 26.6,  0.0,  5.4, 20.5])  mask [0, 0, 1, 0, 1, 1]  fails 0  done False
-> 2: paid 0.230  load  0.0  observed tensor([ 0.0,  0.0, 17.2,  0.0,  5.4, 20.5])  mask [1, 0, 0, 0, 0, 0]  fails 1  done False
-> 0: paid 0.230  load 20.0  observed tensor([ 0.0,  0.0, 17.2,  0.0,  5.4, 20.5])  mask [0, 0, 1, 0, 1, 1]  fails 1  done False
-> 2: paid 0.230  load  2.8  observed tensor([ 0.0,  0.0,  0.0,  0.0,  5.4, 20.5])  mask [1, 0, 0, 0, 1, 1]  fails 1  done False
-> 4: paid 0.665  load  0.4  observed tensor([ 0.0,  0.0,  0.0,  0.0,  0.0, 20.5])  mask [1, 0, 0, 0, 0, 1]  fails 1  done False
-> 5: paid 0.277  load  0.0  observed tensor([ 0.0,  0.0,  0.0,  0.0,  0.0, 15.7])  mask [1, 0, 0, 0, 0, 0]  fails 2  done False
-> 0: paid 0.646  load 20.0  observed tensor([ 0.0,  0.0,  0.0,  0.0,  0.0, 15.7])  mask [0, 0, 0, 0, 0, 1]  fails 2  done False
-> 5: paid 0.646  load  4.3  observed tensor([0., 0., 0., 0., 0., 0.])  mask [1, 0, 0, 0, 0, 0]  fails 2  done False
-> 0: paid 0.646  load 20.0  observed tensor([0., 0., 0., 0., 0., 0.])  mask [1, 0, 0, 0, 0, 0]  fails 2  done True
```

Read it slowly:

- **Customer 3** was believed to need 6.3 but needed 13.2: delivered, and the load dropped from 20 to 6.8.
- **At the depot the mask is `[0, …]`**: you may not drive depot → depot.
- **Customer 2** was believed 26.6 but needed **37.2**. The truck hands over all 20, 17.2 is still open,
  the load is 0, and the mask allows *only the depot*. That's recourse event 1.
- **Customer 5**: driving there with only 0.4 on board was a bad decision (made deliberately here, to show
  a split). Only 0.4 was delivered, a second trip was needed, and that's fail 2.
- The episode ends (`done True`) only when everyone is served *and* the truck is home.

## Code walkthrough

### Lines 1–19: the contract

<!-- quote: svrp/env/svrp_env.py -->
```python
"""The SVRP environment (an MDP), batched: B independent days are simulated in parallel on the GPU.

One truck, node 0 = depot. At every step the policy picks the next node to drive to.

What the policy can see (observe()):
    per node i:  [w, d_i^t / Q, ĉ_i0, …, ĉ_iN]
                 d_i^t is the *estimate* until the truck first arrives at i; from then on it is the true
                 remaining demand (demand is revealed on arrival, like a real customer telling the driver).
    truck:       current node, load / Q
What the policy never sees: the true demand of unvisited customers, the true travel costs.
What it pays: the TRUE cost of every edge it drives (+ automatic recourse under full delivery).
"""
from __future__ import annotations

import torch

from svrp.data import Batch

EPS = 1e-6
```

The docstring states, in words, what the policy can see, what it can never see, and what it pays. It's
worth memorising; everything in the class enforces it. The only project import is `Batch` (the env reads
a batch of days). `EPS = 1e-6` is the same floating-point tolerance as in the executor.

### Lines 22–40: the state

<!-- quote: svrp/env/svrp_env.py -->
```python
class SVRPEnv:
    def __init__(self, batch: Batch, delivery: str = "partial", weather_input: bool = True):
        if delivery not in ("partial", "full"):
            raise ValueError(delivery)
        self.batch = batch
        self.delivery = delivery
        self.weather_input = weather_input
        self.Q = batch.capacity
        B, N1 = batch.demand.shape
        dev = batch.demand.device
        self.ar = torch.arange(B, device=dev)
        self.pos = torch.zeros(B, dtype=torch.long, device=dev)
        self.load = torch.full((B,), self.Q, device=dev)
        self.remaining = batch.demand.clone()
        self.visited = torch.zeros(B, N1, dtype=torch.bool, device=dev)
        self.visited[:, 0] = True
        self.total_cost = torch.zeros(B, device=dev)
        self.n_fail = torch.zeros(B, device=dev)
        self.done = torch.zeros(B, dtype=torch.bool, device=dev)
```

The whole state is seven tensors, one entry per day. All are created on the same device as the batch:

| tensor | shape | start value | meaning |
|---|---|---|---|
| `pos` | (B,) long | 0 | where each truck is (integer node index) |
| `load` | (B,) | Q | how much is on the truck |
| `remaining` | (B, N1) | true demand | what each customer still needs (**hidden** from the policy until visited) |
| `visited` | (B, N1) bool | depot True | has the truck been there yet (= has demand been revealed) |
| `total_cost` | (B,) | 0 | money paid so far |
| `n_fail` | (B,) | 0 | recourse events so far |
| `done` | (B,) bool | False | finished? |

- `self.ar` is the arange from lesson 02 section 5, made once and reused in every step.
- `remaining = batch.demand.clone()`: a copy, because the env edits it in place and the batch must stay
  intact (other methods are scored on the same batch).

### Lines 42–58: what the policy sees

<!-- quote: svrp/env/svrp_env.py -->
```python
    # ------------------------------------------------------------------ observation
    @property
    def n_nodes(self) -> int:
        return self.remaining.shape[1]
```

A section comment (the dashed lines split the class into observation / transition / beam-search parts),
and `n_nodes` = N+1, read from the state's shape.

<!-- quote: svrp/env/svrp_env.py -->
```python
    def observed_demand(self) -> torch.Tensor:
        """Estimate for customers not visited yet, true remaining demand for those already visited."""
        return torch.where(self.visited, self.remaining, self.batch.dem_hat)
```

**The reveal rule in one line.** Where `visited` is True show the truth (`remaining`), else show the
belief (`dem_hat`). This is the *only* way true demand reaches a policy.

<!-- quote: svrp/env/svrp_env.py -->
```python
    def node_features(self) -> torch.Tensor:
        """(B, N+1, M + 1 + N+1): the per-node input of the paper's network."""
        b = self.batch
        B, N1 = self.remaining.shape
        w = b.w[:, None, :].expand(B, N1, b.w.shape[1])
        if not self.weather_input:
            w = torch.zeros_like(w)
        return torch.cat([w, (self.observed_demand() / self.Q)[..., None], b.cost_hat], dim=-1)
```

The paper's per-customer input (`part 5.md`, "The Inputs"): weather + current demand + the row of travel
costs. Shape (B, N1, M + 1 + N1):

- `w` (B, M) is copied to every node: (B, N1, M). Every customer "sees" today's weather.
- The observed demand divided by Q (so it's around 0–1, a friendly range for neural networks), with a new
  last dimension of size 1.
- `cost_hat` row i = the estimated cost from node i to every node: N1 numbers.
- `weather_input=False` zeroes the weather. The experiment "does the network need to see the weather?"
  flips this switch.

### Lines 60–67: the mask (legal moves)

<!-- quote: svrp/env/svrp_env.py -->
```python
    def mask(self) -> torch.Tensor:
        """(B, N+1) bool, True = allowed. Never all-False."""
        open_ = self.remaining > EPS
        open_[:, 0] = False
        all_served = ~open_.any(1)
        m = open_ & (self.load > EPS)[:, None]           # an empty truck cannot serve anybody
        m[:, 0] = (self.pos != 0) | all_served           # no depot -> depot loops (unless finished)
        return m
```

The rules, in plain words:

1. A customer is allowed if they still need something **and** the truck isn't empty.
2. The depot is allowed if the truck is **not** at the depot (no pointless depot → depot), or if
   everything is served (then "stay home" is the only thing left to do).

**Doesn't the mask leak the truth?** It uses `remaining`, the true value. For an *unvisited* customer the
true demand is always ≥ 1 (clipped in lesson 04), so "still needs something" is always True for them.
Using it reveals nothing. For visited customers, the truth is already revealed. (This exact question was
checked when the results looked "too good"; see `experiments/LOG.md`, E2.)

**"Never all-False" is a promise**, and the test checks it. If the truck is at the depot, it is full, so
some open customer is allowed. If it's elsewhere, the depot is allowed. If everything is served, the
depot is allowed. A softmax over all −∞ would give NaN and silently ruin training.

**The env trusts the policy to obey the mask.** `step` doesn't check legality. The rollout code (lesson
10) always masks before choosing. If you call `step` by hand with an illegal action, the env just pays
for the road and changes nothing useful.

### Lines 69–73: how long can a day take?

<!-- quote: svrp/env/svrp_env.py -->
```python
    def max_steps(self) -> int:
        """Upper bound on episode length for ANY legal policy (see the counting argument in SPEC S3)."""
        N = self.n_nodes - 1
        trips_full = int(torch.ceil(self.batch.demand.sum(1).max() / self.Q).item())
        return 3 * N + 2 * trips_full + 2
```

The rollout loop needs a maximum number of steps. Too small cuts days short; infinite risks hanging. The
**counting argument** (partial delivery) goes like this:

- Every customer visit either **finishes** that customer, or **empties** the truck.
- A trip (depot → … → depot) can empty the truck at most once, because then only "go home" is allowed.
- A trip that finishes nobody must have emptied a *full* truck at its first stop, so it delivered Q. There
  are at most ⌈total demand / Q⌉ such trips. All other trips finish ≥ 1 customer, so there are at most N
  of those.
- So trips ≤ N + ⌈D/Q⌉, and visits ≤ N (finishing visits) + trips (one emptying visit per trip). Steps =
  visits + trips ≤ 3N + 2⌈D/Q⌉ + 1.

The code adds a little slack (+2) and uses the largest day in the batch. This uses the *true* demand,
but only to bound a loop, never to make a decision. Full delivery is shorter (each visit finishes a
customer), so the same bound covers it.

### Lines 75–90: one step, the transition

<!-- quote: svrp/env/svrp_env.py -->
```python
    # ------------------------------------------------------------------ transition
    def step(self, action: torch.Tensor) -> torch.Tensor:
        """Drive every unfinished day to its chosen node. Returns the cost paid this step (B,)."""
        ar, b = self.ar, self.batch
        active = ~self.done
        step_cost = b.cost[ar, self.pos, action] * active

        is_cust = (action != 0) & active
        rem_a = self.remaining[ar, action]
        fail = is_cust & (self.load < rem_a - EPS)       # not enough on the truck = recourse event
        self.n_fail += fail.float()
```

- `active`: days not finished yet. Multiplying by `active` (True = 1, False = 0) makes finished days pay 0.
- `step_cost`: the **true** cost of the road from `pos` to `action`, for each day: the arange indexing
  trick.
- `is_cust`: this day is driving to a customer (not the depot) and is active.
- `rem_a`: how much that customer still needs (the truth, used here to *apply* the rules).
- `fail`: arriving with less than needed is a recourse event.

<!-- quote: svrp/env/svrp_env.py -->
```python
        if self.delivery == "partial":
            delivered = torch.minimum(self.load, rem_a) * is_cust
            self.remaining[ar, action] = (rem_a - delivered).clamp_min(0.0)
            self.load = (self.load - delivered).clamp_min(0.0)
```

**Partial delivery:** hand over min(load, needed). Both amounts go down by that. `clamp_min(0.0)` stops
float crumbs from becoming −0.0000001. The policy then decides what to do next. If the truck is empty,
the mask will force it home.

Compare with the executor (lesson 07): there, recourse was *automatic* (immediate shuttles). Here the
**policy itself decides** when and how to come back. That freedom is exactly what RL learns to use well.

### Lines 91–105: full delivery and the end of the step

<!-- quote: svrp/env/svrp_env.py -->
```python
        else:
            # indivisible order: drive back to the depot, refill, come back, then deliver everything
            round_trip = b.cost[ar, action, 0] + b.cost[ar, 0, action]
            step_cost = step_cost + fail * round_trip
            load = torch.where(fail, torch.full_like(self.load, self.Q), self.load)
            self.load = torch.where(is_cust, (load - rem_a).clamp_min(0.0), self.load)
            self.remaining[ar, action] = torch.where(is_cust, torch.zeros_like(rem_a), rem_a)
```

**Full delivery:** an order can't be split. On failure, the env adds the round trip to the depot
automatically and refills the truck (`load = Q`). Then the whole order is delivered (`remaining = 0`).
`torch.where` makes these changes only where the condition holds and leaves the other days alone.

<!-- quote: svrp/env/svrp_env.py -->
```python
        at_depot = (action == 0) & active
        self.load = torch.where(at_depot, torch.full_like(self.load, self.Q), self.load)
        self.pos = torch.where(active, action, self.pos)
        self.visited[ar, action] |= active
        self.total_cost += step_cost
        self.done = (self.pos == 0) & ~(self.remaining[:, 1:] > EPS).any(1)
        return step_cost
```

- Going to the depot refills the truck.
- Active trucks move. Finished ones stay put.
- Mark the node as visited, which reveals its demand from now on. `|=` is in-place "or".
- Add up the cost.
- **Done** = at the depot, and no customer has anything left.

### Lines 107–116: copying days (for beam search)

<!-- quote: svrp/env/svrp_env.py -->
```python
    # ------------------------------------------------------------------ beam search support
    def select(self, idx: torch.Tensor) -> "SVRPEnv":
        """New env whose day k is a copy of this env's day idx[k] (beam search re-uses this)."""
        new = SVRPEnv.__new__(SVRPEnv)
        new.batch = self.batch.index(idx)
        new.delivery, new.weather_input, new.Q = self.delivery, self.weather_input, self.Q
        new.ar = torch.arange(len(idx), device=idx.device)
        for name in ("pos", "load", "remaining", "visited", "total_cost", "n_fail", "done"):
            setattr(new, name, getattr(self, name)[idx].clone())
        return new
```

Beam search (lesson 10) must branch: "copy day 2 three times, then try three different next stops".
`select([2, 2, 2])` returns a new env holding three independent copies of day 2's *current* state.

- `SVRPEnv.__new__(SVRPEnv)` makes an empty object **without running `__init__`** (which would reset
  everything to the start of the day).
- The fields are then copied by hand. `.clone()` makes the copies independent: stepping one copy mustn't
  change another (the test checks this).

### `svrp/baselines/nn_online.py`: the first "policy"

<!-- quote: svrp/baselines/nn_online.py -->
```python
"""Nearest-feasible-neighbour, re-decided at every stop (an *online* heuristic, like a driver without a plan).

Rule: drive to the closest (by estimated cost) open customer whose believed demand fits on the truck;
if nobody fits, go back to the depot and refill. Any learned policy should at least beat this.
"""
from __future__ import annotations

import torch

from svrp.data import Batch
from svrp.env.svrp_env import EPS, SVRPEnv
```

The docstring gives the rule in one sentence and says why it exists: *any learned policy should at
least beat this*. It imports the env and its `EPS`, because it is a policy that plays the env's game.

<!-- quote: svrp/baselines/nn_online.py -->
```python
@torch.no_grad()
def nn_online(batch: Batch, delivery: str = "partial", split: bool = False) -> SVRPEnv:
    """split=True ("nn_split"): always drive to the nearest open customer and hand over whatever is on
    the truck; go home only when empty. A control for how much of RL's gain is just "split deliveries"."""
    env = SVRPEnv(batch, delivery)
    inf = torch.tensor(float("inf"), device=batch.cost.device)
    for _ in range(env.max_steps()):
        if env.done.all():
            break
        m = env.mask()
        cust = m.clone()
        cust[:, 0] = False
        c = batch.cost_hat[env.ar, env.pos]
        fits = cust if split else cust & (env.observed_demand() <= env.load[:, None] + EPS)
        nearest_fit = torch.where(fits, c, inf).argmin(1)
        nearest_any = torch.where(cust, c, inf).argmin(1)
        at_depot_open = (env.pos == 0) & cust.any(1)     # at the depot with a full truck: just go
        a = torch.where(fits.any(1), nearest_fit, torch.where(at_depot_open, nearest_any, 0))
        env.step(a)
    return env
```

This is a hand-written policy using the same environment. It's your first complete "agent", and the
pattern every later policy follows: **loop: look at the mask → choose → step**.

- `@torch.no_grad()`: no learning here, so don't record gradients.
- `cust`: the allowed customers (the mask without the depot).
- `c`: the estimated cost from each truck's position to every node (one row per day).
- `fits`: allowed customers whose *believed* demand fits on the truck. (`split=True` drops that
  condition: the `nn_split` control from `LOG.md` E2.)
- `torch.where(fits, c, inf).argmin(1)`: the nearest fitting customer. Non-fitting ones get cost ∞, so
  `argmin` never picks them.
- The decision, written as nested `where`s:
  - if someone fits, go to the nearest who fits;
  - else, if we're at the depot (full truck) and someone is open, go to the nearest anyway (their order
    must simply be bigger than a truck);
  - else go home (0).
- It returns the finished env: `env.total_cost` holds the result.

## Rebuild it yourself

1. `__init__` and `observed_demand`, then try the "play by hand" script with *your* env.
2. `mask`, then `step` (partial only). Re-run the script: your output must match the one above line by
   line.
3. Full delivery, `max_steps`, `node_features`, `select`.
4. `nn_online`.

## Verify

```bash
uv run python -m pytest tests/test_env.py -q          # copy it first (svrp -> mysvrp)
```

7 tests must pass:

- random legal policies always finish, for partial/full and fixed/flexible;
- the total cost equals the sum of the true roads driven (partial), and is larger exactly when there were
  recourse events (full);
- demand is revealed only after a visit;
- the split example (deliver 5 of 15, then only the depot is allowed);
- `select` copies are independent.

Now also copy `tests/test_baselines.py`. It imports `aco` (lesson 14): **in your copy**, delete the line
`from mysvrp.baselines.aco import aco` and the `"aco": ...` entry of the `PLANNERS` dict until then.

```bash
uv run python -m pytest tests/test_baselines.py -q
```

The key one is `test_deterministic_world_executor_equals_ortools_objective_and_env`. With all randomness
switched off, **three independent calculations of the same plan's cost must agree**: OR-Tools'
objective, your executor (lesson 07), and your environment replaying the plan step by step. If they
agree, the three pieces of code share one understanding of the rules.

Finally, the nearest-neighbour numbers on the 1,000 test days at N = 10 (`experiments/LOG.md`, E1/E2):

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from mysvrp.config import EnvConfig
from mysvrp.data import World
from mysvrp.baselines.nn_online import nn_online
b = World(EnvConfig(n_customers=10)).eval_set('test', 1000)
print('nn_online %.4f' % nn_online(b).total_cost.mean().item(), ' nn_split %.4f' % nn_online(b, split=True).total_cost.mean().item())
"
```

Expected: `nn_online 8.0282  nn_split 7.8668`. A one-line rule that re-decides at every stop already
matches OR-Tools' carefully optimised morning plan (8.02).

## Common mistakes

- **Forgetting `* active`**: finished days keep paying, and costs creep up.
- **Showing `remaining` instead of `observed_demand()`** to the policy: the AI sees the future. Results
  look amazing and mean nothing.
- **`remaining = batch.demand`** without `.clone()`: the env edits the batch, and the next method scored
  on the same batch sees a half-served world.
- **Mask without the depot rule**: depot → depot loops, so episodes never end, until `max_steps` cuts them.
- **`select` with `__init__`**: every copy resets to the start of the day.

## Check your understanding

1. Why is `visited[:, 0]` set to True at the start?
2. A truck is at customer 4 with load 3.0. Customer 6 is unvisited, believed 2.5, true 7.0. Is customer 6
   allowed? What happens under partial delivery if the policy goes there? And under full delivery?
3. Why does `step` multiply the cost by `active` rather than skip finished days with an `if`?
4. In the counting argument, why can a trip empty the truck only once?

**Answers** (try the questions first, then check here)

1. The depot has no demand to reveal. Marking it visited makes `observed_demand` show its true remaining
   (0) instead of `dem_hat`'s 0. The values are the same either way, but this is tidier and consistent.
2. Allowed (open, and the truck isn't empty). Partial: deliver 3.0, so 4.0 is left, the load is 0, it
   counts as a fail, and next the mask allows only the depot. Full: a fail; the env automatically adds
   6 → 0 → 6, refills to 20, and delivers 7.0, leaving load 13.0.
3. An `if` works per day in Python (slow, a loop over 512 days). Multiplying by a 0/1 tensor handles all
   days in one GPU operation.
4. Once the load is 0, the mask allows only the depot, so the trip ends there.

## Read more

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., free at incompleteideas.net/book/the-book-2nd.html),
  chapter 3: "Finite Markov Decision Processes".
- Nazari et al. (2018), "Reinforcement Learning for Solving the Vehicle Routing Problem" (arXiv 1802.04240),
  section 3: the same kind of environment for the deterministic and split-delivery VRP. This paper's network
  is the parent of PaperNet (next lesson).

**Next:** [09 — The policy network](09_policy_network.md)
