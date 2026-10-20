# 10 — Rollouts and inference: turning scores into routes (`svrp/inference.py`)

| | |
|---|---|
| **Layer** | 8 |
| **Files covered** | `svrp/inference.py` (117 lines) |
| **You build** | `mysvrp/inference.py` |
| **Time** | 3–4 hours |
| **Before this** | [08 — Environment](08_environment.md), [09 — Policy network](09_policy_network.md); `part 6.md` ("Inference strategy") |

## Why this layer exists

The network (lesson 09) gives scores for **one** step. A day needs ~20–40 steps. A **rollout** is the
loop that plays a whole day:

```
observe → network scores → mask → choose a node → env.step → repeat until done
```

There are several ways to *choose* (`part 6.md`): greedy, sampling, beam search. And there's a fourth
use: planning in your head on the estimates (a-priori). This file holds all of them.

It also holds one of the project's most important ideas about honesty: **some inference modes quietly
use information a real dispatcher can't have.** We label them.

## Concepts from zero

### A policy is a probability distribution

After masking and softmax, the network gives π(a | s), a probability for each legal next node. Two ways
to pick:

- **Greedy:** always take the most likely node (`argmax`). Deterministic.
- **Sampling:** roll a weighted die (`torch.multinomial`). Different every time. **Training needs
  sampling**: to learn which choices are good, the AI must sometimes try different ones (*exploration*).

### The probability of a whole route = the product of step probabilities

A route a₁, a₂, …, a_T has probability π(a₁)·π(a₂)·…·π(a_T). Multiplying 30 numbers like 0.4 gives a
tiny number (0.4³⁰ ≈ 10⁻¹²) that computers handle badly. So we use **logarithms**: log of a product = sum
of logs. The rollout keeps

```
logp_sum = Σ_t log π(a_t | s_t)
```

A value like −25.9 means "this route had probability e^(−25.9)". REINFORCE (lesson 11) needs exactly
this sum, and its gradient.

### Beam search, with numbers

Greedy commits to the best *single* step. Beam search keeps the W best *partial routes* (beams), scored
by their total log-probability. Example with W = 2:

Step 1 from the depot: A = −0.2, B = −1.8, C = −2.5 → keep **A (−0.2)** and **B (−1.8)**.

Step 2, all continuations:

| beam | next | log p of step | total |
|---|---|---|---|
| A | B | −0.5 | **−0.7** ✓ |
| A | C | −1.2 | **−1.4** ✓ |
| B | A | −0.1 | −1.9 |
| B | C | −2.0 | −3.8 |

Keep A→B and A→C. Beam "B" died, even though its next step was very confident. Beam search looks a little
further than greedy, "like a chess player" in your notes.

### Deployable, lookahead, hindsight: an honesty label

On a real morning you drive **once**, and you learn a customer's demand **only when you arrive**. Check
each mode against that:

| mode | what it does | can a real dispatcher do it? |
|---|---|---|
| greedy (online) | one drive, decisions use only revealed info | **yes** |
| a-priori | plan the whole day on the *estimates*, then drive it with recourse | **yes** |
| beam (online) | W beams "drive" on the real day at once. Beams that visited customer 7 have *seen its true demand*, and that affects which beams survive | **no: lookahead.** It peeks at hidden demand |
| best-of-n | drive n different routes on the real day and keep the cheapest | **no: hindsight.** You can't drive 16 times and keep the best |

The paper's headline RL number uses beam search. This project reports it, but labeled, and treats
**greedy** as the honest headline. The verify section below shows why this matters: even an *untrained*
network looks great with best-of-16.

## Code walkthrough

### Lines 1–19: the header

<!-- quote: svrp/inference.py -->
```python
"""Running a policy on days: greedy / sampling rollouts, beam search, and a-priori planning.

Deployability (SPEC S8):
    greedy online        deployable     uses only what has been revealed so far
    apriori              deployable     plan on the estimates, then drive it with recourse
    beam (online)        LOOKAHEAD      beams "visit" customers on the real day = peeks at hidden demand
    best-of-n sampling   HINDSIGHT      chooses the rollout with the best realized cost
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from svrp.data import Batch
from svrp.env.executor import execute_many
from svrp.env.svrp_env import SVRPEnv
```

The docstring repeats the honesty table, so nobody can use `beam_search` without seeing the word
LOOKAHEAD. `torch.nn.functional` (as `F`) provides `log_softmax`. NumPy and `execute_many` are only needed
by `apriori`, which hands its plans to the executor.

### Lines 22–41: the result type and two helpers

<!-- quote: svrp/inference.py -->
```python
@dataclass
class Rollout:
    cost: torch.Tensor      # (B,) realized total cost
    logp: torch.Tensor      # (B,) Σ_t log π(a_t)  (carries gradients when sampling in training)
    n_fail: torch.Tensor    # (B,) recourse events
    actions: torch.Tensor   # (B, T)
```

Everything a rollout produces, in one object: costs, log-probs, recourse events, and the chosen nodes
(one row per day).

<!-- quote: svrp/inference.py -->
```python
def index_state(state, idx):
    if isinstance(state, torch.Tensor):
        return state[idx]
    if isinstance(state, (tuple, list)):
        return type(state)(index_state(s, idx) for s in state)
    if isinstance(state, dict):
        return {k: index_state(v, idx) for k, v in state.items()}
    return state
```

Beam search must reorder the policy's memory along with the env (`env.select`). PaperNet's state is a
tuple (h, c). GraphNet's (lesson 13) is a dict. This function indexes **any** nesting of tensors,
recursively: a tensor gets indexed, and tuples, lists and dicts get processed element by element. Then
beam search doesn't need to know which network it's running.

<!-- quote: svrp/inference.py -->
```python
def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
```

Mask, then log-softmax (lesson 02 section 6). `log_softmax` is computed in a numerically stable way.
Never write `torch.log(torch.softmax(x))`, which can give −∞ for small probabilities.

### Lines 44–65: `rollout`, the core loop

<!-- quote: svrp/inference.py -->
```python
def rollout(policy, batch: Batch, delivery: str = "partial", weather_input: bool = True,
            decode: str = "greedy") -> Rollout:
    env = SVRPEnv(batch, delivery, weather_input)
    state = policy.init_state(env)
    logp_sum = torch.zeros(batch.size, device=batch.demand.device)
    actions = []
    for _ in range(env.max_steps()):
        if env.done.all():
            break
        logits, state = policy(env, state)
        logp = masked_log_probs(logits, env.mask())
        if decode == "greedy":
            a = logp.argmax(-1)
        elif decode == "sample":
            a = torch.multinomial(logp.exp(), 1).squeeze(1)
        else:
            raise ValueError(decode)
        logp_sum = logp_sum + logp.gather(1, a[:, None]).squeeze(1) * (~env.done)
        env.step(a)
        actions.append(a)
    assert env.done.all(), "episode did not finish within max_steps"
    return Rollout(env.total_cost, logp_sum, env.n_fail, torch.stack(actions, 1))
```

- A fresh env per rollout, and the policy's initial memory.
- Loop at most `max_steps` times, and stop early when every day is done.
- `policy(env, state)` returns logits for every node of every day.
- **The mask is applied here, always.** No decoding mode can pick an illegal node.
- Greedy: `argmax`. Sample: `torch.multinomial(probs, 1)` draws one index per row in proportion to the
  probabilities, shape (B, 1), so `.squeeze(1)` gives (B,).
- **Accumulate log π(chosen node)** with `gather` (lesson 02 section 9). `* (~env.done)` is a safety
  multiply: days that were already finished contribute nothing. (For them only the depot is legal, so its
  log-prob is log 1 = 0 anyway.)
- Step all days. Record the actions.
- The `assert` turns a bug ("an episode never ends") into a loud error instead of a wrong number.
- `torch.stack(actions, 1)`: a list of T tensors of shape (B,) becomes (B, T).

This one function is used everywhere: training (`decode="sample"`, with gradients), validation and
testing (`"greedy"`), and inside a-priori planning.

### Lines 68–76: best-of-n (labeled hindsight)

<!-- quote: svrp/inference.py -->
```python
@torch.no_grad()
def best_of_n(policy, batch: Batch, n: int, delivery="partial", weather_input=True, chunk=4096):
    """HINDSIGHT: sample n rollouts per day on the real day and keep the cheapest."""
    costs = []
    for b0 in range(0, batch.size, max(1, chunk // n)):
        sub = batch.index(slice(b0, b0 + max(1, chunk // n))).repeat_interleave(n)
        r = rollout(policy, sub, delivery, weather_input, decode="sample")
        costs.append(r.cost.view(-1, n).min(1).values)
    return torch.cat(costs)
```

- Process the days in chunks so that at most `chunk` = 4,096 rollouts run at once (memory). With n = 16,
  that's 256 days per chunk.
- `repeat_interleave(n)`: each day n times in a row (rows 0–15 = day 0, …).
- Sample all of them, then `view(-1, n)` regroups into (days, n), and `.min(1)` keeps each day's cheapest.
- The docstring says **HINDSIGHT** in capitals, and the evaluation table prints that label next to the
  number.

### Lines 79–106: beam search

<!-- quote: svrp/inference.py -->
```python
@torch.no_grad()
def beam_search(policy, batch: Batch, width: int, delivery="partial", weather_input=True):
    """Keeps the `width` most probable partial routes. Returns (cost of most-probable beam, cheapest beam)."""
    B, N1, W = batch.size, batch.n_nodes, width
    dev = batch.demand.device
    env = SVRPEnv(batch, delivery, weather_input).select(torch.arange(B, device=dev).repeat_interleave(W))
    state = policy.init_state(env)
    score = torch.full((B, W), float("-inf"), device=dev)
    score[:, 0] = 0.0                                       # start with a single live beam per day
    rows = torch.arange(B, device=dev)[:, None]
```

- Make W copies of every day: B·W envs (rows 0…W−1 = day 0's beams, …).
- `score[b, k]`: the total log-prob of beam k of day b. At the start all W beams are identical, so only
  beam 0 is "alive" (score 0) and the others are −∞. Otherwise the first step would pick the same
  continuation W times, from W identical copies.

<!-- quote: svrp/inference.py -->
```python
    for _ in range(env.max_steps()):
        if env.done.all():
            break
        logits, state = policy(env, state)
        logp = masked_log_probs(logits, env.mask())        # finished beams: depot has log p = 0
        cand = (score.view(-1, 1) + logp).view(B, W * N1)
        top, ti = cand.topk(W, dim=1)
        parent, action = ti // N1, ti % N1
```

The heart, in four lines:

1. Every beam's next-step log-probs, shape (B·W, N1).
2. **Candidate score** = beam score + step log-prob, for every (beam, next node) pair. `.view(B, W*N1)`
   lays all W·N1 candidates of a day in one row.
3. `topk(W)`: the W best candidates per day.
4. Decode the flat candidate index: `parent = ti // N1` (which beam it extends) and `action = ti % N1`
   (which node). Example with N1 = 3: candidate 4 → beam 1, node 1.

A finished beam has only one legal continuation (the depot, log-prob 0), so its score carries over
unchanged. Finished beams compete fairly with unfinished ones.

<!-- quote: svrp/inference.py -->
```python
        dead = ~torch.isfinite(top)                         # fewer than W legal continuations
        parent = torch.where(dead, parent[:, :1], parent)
        action = torch.where(dead, action[:, :1], action)
        flat = (rows * W + parent).view(-1)
        env, state = env.select(flat), index_state(state, flat)
        env.step(action.view(-1))
        score = top
```

- Sometimes a day has fewer than W legal candidates in total. Then `topk` returns some −∞ "candidates"
  whose action may be illegal. Those *dead* slots are filled with a copy of the best beam (column 0). They
  keep score −∞, so they never win.
- `flat = rows*W + parent`: the row number of each chosen parent among the B·W envs.
- `env.select(flat)` copies those parents (lesson 08), and the policy's memory is copied the same way.
- Step every beam with its chosen node, and update the scores.

<!-- quote: svrp/inference.py -->
```python
    cost = env.total_cost.view(B, W)
    most_probable = cost[torch.arange(B, device=dev), score.argmax(1)]
    return most_probable, cost.min(1).values
```

Two answers: the cost of the **most probable** beam (what beam search "chooses", the paper's way), and
the **cheapest** beam (pure hindsight, for curiosity). The evaluation reports the first, labeled
"lookahead".

### Lines 109–117: a-priori RL, plan in your head and then drive

<!-- quote: svrp/inference.py -->
```python
@torch.no_grad()
def apriori(policy, batch: Batch, delivery="partial", weather_input=True, decode="greedy"):
    """DEPLOYABLE plan-then-drive: roll out in the *believed* world, then execute that plan for real."""
    r = rollout(policy, batch.believed(), delivery, weather_input, decode=decode)
    acts = r.actions.cpu().numpy()
    plans = [np.concatenate([[0], a]) for a in acts]
    cost, fails = execute_many(plans, batch.demand.cpu().double().numpy(), batch.cost.cpu().double().numpy(),
                               batch.capacity, delivery)
    return cost, fails
```

- `batch.believed()` (lesson 05): a world where the estimates *are* the truth. Rolling out there is the
  policy imagining its day in the morning, with no surprises.
- The recorded actions become plans: prefix the start at the depot. The trailing zeros after the episode
  ended are harmless, because the executor ignores depot → depot.
- Then drive those plans on the **real** day with the executor's recourse rules (lesson 07). This is
  exactly how the classical planners are scored, so a-priori RL vs OR-Tools is a fair, apples-to-apples
  comparison. It's also the paper's "a priori vs reoptimisation" experiment (finding 7).

## Rebuild it yourself

1. `Rollout`, `index_state`, `masked_log_probs`.
2. `rollout`. Test it with your `PaperNet` from lesson 09.
3. `best_of_n`, `apriori`.
4. `beam_search`, last. Check that width 1 gives exactly the greedy cost.

## Verify

From `learning/workspace`, on the CPU, with an **untrained** network (seeded):

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
import torch
from mysvrp.config import EnvConfig
from mysvrp.data import World
from mysvrp.models.paper_net import PaperNet
from mysvrp.inference import rollout, beam_search, best_of_n, apriori
torch.manual_seed(0)
b = World(EnvConfig(n_customers=10)).eval_set('test', 100)
net = PaperNet(11, 3, 128)
with torch.no_grad():
    g = rollout(net, b)
    s = rollout(net, b, decode='sample')
print('greedy  %.4f  steps %d  first route %s' % (g.cost.mean().item(), g.actions.shape[1], g.actions[0].tolist()))
print('sample  %.4f  logp of day 0 route %.3f' % (s.cost.mean().item(), s.logp[0].item()))
mp1, ch1 = beam_search(net, b, 1); print('beam1   %.4f (must equal greedy)' % mp1.mean().item())
mp3, ch3 = beam_search(net, b, 3); print('beam3   most-probable %.4f  cheapest %.4f' % (mp3.mean().item(), ch3.mean().item()))
print('best16  %.4f' % best_of_n(net, b, 16).mean().item())
c, f = apriori(net, b); print('apriori %.4f  fails %.2f' % (c.mean(), f.mean()))
"
```

Reference output:

```
greedy  9.4989  steps 29  first route [1, 5, 7, 0, 7, 0, 4, 0, 4, 0, 10, 0, 9, 0, 8, 0, 3, 0, 6, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]
sample  10.1815  logp of day 0 route -25.898
beam1   9.4989 (must equal greedy)
beam3   most-probable 9.7404  cheapest 9.5157
best16  7.5122
apriori 9.3923  fails 1.87
```

What to notice:

- **The untrained network is bad** (9.50 vs Clarke–Wright's ~8.0). Training will fix that.
- **The trailing zeros** in the route: day 0 finished early, and it waits at the depot while the other 99
  days finish.
- **beam1 = greedy**, to every digit. (The test `test_beam_width_one_equals_greedy` checks this.)
- **best16 = 7.51 for a *random* network**, already better than OR-Tools' 8.02. That's the power of
  hindsight: choosing the luckiest of 16 drives on a day you already know. It's exactly why such numbers
  must never be presented as a method's real performance.
- With sampling, your numbers match only if your code consumes the random stream in the same order. If
  greedy, beam and apriori match, you're fine.

## Common mistakes

- **Sampling without the mask**: illegal moves, and the env silently does nonsense.
- **`torch.log(torch.softmax(...))`** instead of `log_softmax`: `-inf` and NaN in training.
- **Forgetting `index_state` in beam search**: the beams' memories get mixed up, and beam1 ≠ greedy.
- **All beams alive at the start**: W identical beams pick W copies of the same node, and beam ≈ greedy.
- **Calling `rollout` for evaluation without `torch.no_grad()`**: it works, but stores gradient
  information for nothing (slow, uses lots of memory).

## Check your understanding

1. A route has step probabilities 0.5, 0.8, 0.25. What is its `logp_sum`?
2. Why must training use `decode="sample"` rather than greedy?
3. In the beam table above, which beams survive with W = 3?
4. Why is a-priori RL a fairer comparison to OR-Tools than online RL? And why is online RL still the more
   *useful* method?

**Answers** (try the questions first, then check here)

1. ln 0.5 + ln 0.8 + ln 0.25 = −0.693 − 0.223 − 1.386 = **−2.303** (= ln 0.1).
2. Greedy always does the same thing, so the AI never discovers whether another choice would have been
   cheaper. REINFORCE learns from comparing *different* sampled outcomes.
3. A→B (−0.7), A→C (−1.4) and B→A (−1.9).
4. Both a-priori methods plan from the same estimates and pay the same recourse rules, so the only
   difference is plan quality. Online RL is more useful because a real driver *can* re-decide after each
   stop. Using revealed information is allowed, and it's exactly where RL wins.

## Read more

- `part 6.md` ("Inference strategy") and the paper's section 4.3.
- Kool, van Hoof & Welling (2019), "Attention, Learn to Solve Routing Problems!" (arXiv 1803.08475),
  section 4: greedy vs sampling 1,280 solutions. In *deterministic* routing best-of-n is legitimate. Ask
  yourself why it isn't here.

**Next:** [11 — Training with REINFORCE](11_training_reinforce.md)
