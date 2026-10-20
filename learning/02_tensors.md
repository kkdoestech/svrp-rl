# 02 — Tensors crash course (with the project's own lines)

| | |
|---|---|
| **Layer** | 0 |
| **You build** | nothing permanent; you run small experiments in a Python shell |
| **Time** | 3–5 hours. Don't rush this lesson; every later lesson depends on it |
| **Before this** | [01 — Setup](01_setup_and_tools.md) |

## Why this lesson exists

The whole project is written with **tensors**: PyTorch's multi-dimensional arrays. Instead of looping
over 512 days one by one, the code handles all 512 at once, in one line. That's why training is fast.
It's also why the code can look cryptic at first.

This lesson teaches each tensor trick the project uses. For each one you get a tiny example with its
**real printed output** (all were run with the project's PyTorch), then the line in the project where
you'll meet it.

Open a Python shell to try things as you read:

```bash
uv run python
>>> import torch
```

## 1. The shape convention: every letter means something

A tensor has a **shape**, a tuple of sizes, one per dimension. In this project the letters are always
the same:

| letter | meaning | typical value |
|---|---|---|
| **B** | batch size: how many independent *days* are simulated together | 512 in training, 1000 in tests |
| **N** | number of customers | 10, 20, 50 |
| **N1 = N+1** | number of **nodes** (node 0 = the depot, nodes 1…N = customers) | 11, 21 |
| **M** | number of weather variables | 3 |
| **K = M²** | number of weather *products* w_m·w_n | 9 |
| **P = N1²** | number of edges (i → j pairs) | 121, 441 |
| **C** | number of cities (1 if the city is fixed) | 1 or B |
| **H** | days in the history bank | 1000 |
| **D** | embedding size of the neural network | 128 |
| **W** | beam width | 3 |
| **T** | steps in an episode | ~30–40 |

So when a comment says `cost (B, N+1, N+1)`, it means: for each of B days, a full matrix of travel
costs between all N+1 nodes. `cost[5, 0, 3]` is "on day 5, the cost of driving from the depot to
customer 3".

**The single most useful habit:** when confused, print `.shape`.

## 2. Creating tensors

| code | gives | used for |
|---|---|---|
| `torch.zeros(B)` | B zeros | total cost per day, starts at 0 |
| `torch.full((B,), 20.0)` | B copies of 20.0 | every truck starts full (load = Q) |
| `torch.arange(B)` | 0, 1, …, B−1 | the "day index", see section 5 |
| `torch.rand(B, N1, 2)` | uniform in [0, 1) | customer positions |
| `torch.randn(...)` | normal N(0, 1) | noise ε, raw weather weights α |
| `torch.empty(...)` | uninitialised memory (fast); you must fill it | k-NN output buffer |
| `torch.tensor([1., 2.])` | from a Python list | tests and examples |

Every creation function accepts `device="cuda"` (create directly on the GPU) and `generator=g` (use a
specific random-number stream, section 12).

## 3. Indexing: `[:, 0]`, `[..., 0]`, negative dimensions

```python
coords[:, 0] = 0.5          # for every city (all of dim 0), node 0 (the depot) gets x = y = 0.5
d[..., 0] = 0.0             # "..." = "all leading dimensions, however many": the last index 0 = depot
```

`...` (ellipsis) is handy when the same function must work for shapes (B, N1) *and* (C, H, N1). Negative
dimensions count from the end: `dim=-1` is the last dimension, `dim=-2` the one before.

Where you'll meet it (`svrp/env/generator.py`):

<!-- quote: svrp/env/generator.py -->
```python
d[..., 0] = 0.0                     # the depot has no demand
```

## 4. Broadcasting: making shapes fit automatically

When you combine two tensors with `+`, `*`, `&`, `<` and so on, PyTorch lines the shapes up **from the
right**. A dimension of size **1** is *stretched* to match the other tensor. A missing dimension counts
as size 1.

`x[:, None]` (or `x.unsqueeze(1)`) inserts a new dimension of size 1. Example: is the truck empty, per day?

```python
>>> load = torch.tensor([5.0, 0.0])               # 2 days: truck has 5 units / 0 units
>>> (load > 1e-6)[:, None]                         # shape (2,) -> (2, 1)
tensor([[ True],
        [False]])
>>> remaining = torch.tensor([[0., 3., 0.], [0., 4., 2.]])   # (2 days, 3 nodes)
>>> open_ = remaining > 1e-6
>>> open_ & (load > 1e-6)[:, None]                 # (2, 3) & (2, 1) -> (2, 3)
tensor([[False,  True, False],
        [False, False, False]])
```

The (2, 1) column was stretched across the 3 nodes: "customer still has demand **and** this day's truck
is not empty". Day 2's truck is empty, so no customer is allowed. This exact line is in the
environment's mask (`svrp/env/svrp_env.py`):

<!-- quote: svrp/env/svrp_env.py -->
```python
m = open_ & (self.load > EPS)[:, None]           # an empty truck cannot serve anybody
```

**Outer product by broadcasting.** (M, 1) × (1, M) → (M, M), i.e. all products w_m·w_n:

```python
>>> w = torch.tensor([1., 2., 3.])
>>> w[:, None] * w[None, :]
tensor([[1., 2., 3.],
        [2., 4., 6.],
        [3., 6., 9.]])
>>> (w[..., :, None] * w[..., None, :]).flatten(-2)       # flattened to K = 9 numbers
tensor([1., 2., 3., 2., 4., 6., 3., 6., 9.])
```

That is how the generator gets all the w_m·w_n for the formula Σ α_mn w_m w_n:

<!-- quote: svrp/env/generator.py -->
```python
return (w[..., :, None] * w[..., None, :]).flatten(-2)      # (..., M²): all w_m · w_n
```

## 5. Indexing with `arange`: "for each day, pick *its own* element"

This is the trick that confuses everybody once, and then is used everywhere.

Problem: you have the cost matrices of B days, shape (B, N1, N1). Each day's truck is at its own
position `pos[b]` and drives to its own `action[b]`. You want B numbers: `cost[b, pos[b], action[b]]`
for each b.

Solution: index with *three tensors of the same length*. PyTorch pairs them element by element.

```python
>>> cost = torch.tensor([[[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]],      # day 0
...                      [[0., 5., 6.], [5., 0., 7.], [6., 7., 0.]]])     # day 1
>>> ar = torch.arange(2)               # tensor([0, 1])  the day index
>>> pos = torch.tensor([0, 2])         # day 0 at node 0, day 1 at node 2
>>> action = torch.tensor([1, 1])      # both drive to node 1
>>> cost[ar, pos, action]              # = [cost[0,0,1], cost[1,2,1]]
tensor([1., 7.])
```

In the environment (`svrp/env/svrp_env.py`), `self.ar = torch.arange(B)` is created once and reused:

<!-- quote: svrp/env/svrp_env.py -->
```python
step_cost = b.cost[ar, self.pos, action] * active
```

and in the network (`svrp/models/paper_net.py`), "the embedding of the node each truck stands on":

<!-- quote: svrp/models/paper_net.py -->
```python
e_pos = e[env.ar, env.pos]
```

`e` has shape (B, N1, D). `e[ar, pos]` picks, per day, one row of D numbers, giving shape (B, D).

Writing works too: `self.remaining[ar, action] = new_values` changes one entry per day.

## 6. Boolean masks and `masked_fill`

Comparisons give **bool tensors**. Combine them with `&` (and), `|` (or), `~` (not). Don't use Python's
`and`/`or`/`not` on tensors. Reduce them with `.any(dim)` and `.all(dim)`.

The most important use is **forbidding illegal moves** before the softmax: set their scores to −∞, so
their probability becomes exactly 0.

```python
>>> logits = torch.tensor([[2.0, 1.0, 0.5], [0.3, 0.2, 0.1]])
>>> mask = torch.tensor([[False, True, True], [True, False, False]])     # True = allowed
>>> x = logits.masked_fill(~mask, float("-inf"))
>>> x
tensor([[  -inf, 1.0000, 0.5000],
        [0.3000,   -inf,   -inf]])
>>> torch.log_softmax(x, dim=-1).exp()          # probabilities
tensor([[0.0000, 0.6225, 0.3775],
        [1.0000, 0.0000, 0.0000]])
```

Row 1: node 0 is forbidden, so the probability is shared between nodes 1 and 2. Row 2: only node 0 is
allowed, so it gets probability 1. The project does this in `svrp/inference.py`:

<!-- quote: svrp/inference.py -->
```python
return F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
```

## 7. `torch.where(cond, a, b)`: an if/else for every element

```python
>>> torch.where(mask, logits, torch.zeros_like(logits))
tensor([[0.0000, 1.0000, 0.5000],
        [0.3000, 0.0000, 0.0000]])
```

Where `cond` is True take `a`, else `b`. Shapes broadcast. In the environment (refill the truck on days
that drove to the depot, keep the old load on the others):

<!-- quote: svrp/env/svrp_env.py -->
```python
self.load = torch.where(at_depot, torch.full_like(self.load, self.Q), self.load)
```

## 8. Reductions: `sum`, `mean`, `min`, `argmax`, `keepdim`, `topk`

```python
>>> c = torch.tensor([[3., 1., 2.], [0., 5., 4.]])
>>> c.sum(1), c.mean(0), c.min(1).values, c.argmin(1)
(tensor([6., 9.]), tensor([1.5000, 3.0000, 3.0000]), tensor([1., 0.]), tensor([1, 0]))
>>> c.sum(1, keepdim=True).shape          # keep the summed dim as size 1 (handy for broadcasting back)
torch.Size([2, 1])
>>> c.topk(2, dim=1, largest=False)       # the 2 smallest per row, with their positions
torch.return_types.topk(
values=tensor([[1., 2.],
        [0., 4.]]),
indices=tensor([[1, 2],
        [0, 2]]))
```

`dim=1` means "collapse dimension 1": you get one answer per row. `min` and `max` along a dimension return
`(values, indices)`. `argmin`/`argmax` return just the indices. Where you'll meet them:

- `logp.argmax(-1)`: greedy decoding picks the most likely node (lesson 10).
- `d.topk(k, dim=-1, largest=False)`: the k most similar past days in k-NN (lesson 05).
- `c.mean(1, keepdim=True)`: the shared baseline, the average cost of M rollouts of the same day (lesson 11).

## 9. `gather`: pick one value per row by index

After choosing action `a[b]` for each day, you need its log-probability `logp[b, a[b]]`:

```python
>>> lp = torch.log_softmax(x, dim=-1)
>>> a = lp.argmax(-1)                  # tensor([1, 0])
>>> lp.gather(1, a[:, None]).squeeze(1)
tensor([-0.4741,  0.0000])
```

`gather(1, idx)` reads along dim 1 at positions `idx` (shape (B, 1)). `squeeze(1)` removes the size-1
dimension again. (`lp[ar, a]` gives the same result. Both styles appear in practice.)

<!-- quote: svrp/inference.py -->
```python
logp_sum = logp_sum + logp.gather(1, a[:, None]).squeeze(1) * (~env.done)
```

## 10. `cdist`, `triu`: distances and symmetric matrices

```python
>>> pts = torch.tensor([[[0.5, 0.5], [0.5, 1.0], [1.0, 1.0]]])   # (1 city, 3 nodes, xy)
>>> torch.cdist(pts, pts)                                          # all pairwise distances
tensor([[[0.0000, 0.5000, 0.7071],
         [0.5000, 0.0000, 0.5000],
         [0.7071, 0.5000, 0.0000]]])
```

To make *symmetric* random noise (the same noise for i→j and j→i), keep the part above the diagonal and
mirror it:

```python
>>> e = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
>>> u = torch.triu(e, diagonal=1); u + u.transpose(-1, -2)
tensor([[0., 2., 3.],
        [2., 0., 6.],
        [3., 6., 0.]])
```

<!-- quote: svrp/env/generator.py -->
```python
eps_cost = torch.triu(torch.randn(Cq, H, N1, N1, generator=gen), diagonal=1)
eps_cost = (eps_cost + eps_cost.transpose(-1, -2)) * city.sigma_cost[:, None]
```

## 11. Reshaping: `view`, `reshape`, `flatten`, `transpose`, `expand`, `repeat_interleave`

```python
>>> d = torch.arange(6).view(2, 3)        # same 6 numbers, seen as 2 rows × 3 cols
>>> d.view(3, 2)                          # tensor([[0, 1], [2, 3], [4, 5]])
>>> d.flatten()                           # tensor([0, 1, 2, 3, 4, 5])
>>> d.transpose(0, 1)                     # tensor([[0, 3], [1, 4], [2, 5]])
```

- `view` / `reshape` change the shape without changing the order of the numbers. `view` needs the data
  to be laid out contiguously in memory; `reshape` copies if it must. After a `transpose` you usually need
  `reshape`, not `view`.
- `flatten(-2)` merges the last two dims: (…, M, M) → (…, M²).

Two ways to make copies along a batch dimension:

```python
>>> torch.tensor([[1, 2, 3]]).expand(2, 3)         # a VIEW: no memory copied
tensor([[1, 2, 3],
        [1, 2, 3]])
>>> torch.tensor([[1, 2], [3, 4]]).repeat_interleave(2, dim=0)    # real copies, each row repeated
tensor([[1, 2],
        [1, 2],
        [3, 4],
        [3, 4]])
```

`expand` is free but *read-only in spirit*: all rows share the same memory, so writing into one row
writes into all. That's why `data.py` writes `t.expand(...).clone()` before anything could modify it.
`repeat_interleave(M)` is used for the shared baseline: each day is repeated M times *next to each
other* (rows 0–7 = day 0, rows 8–15 = day 1, …), so `.view(-1, M)` later groups them back.

## 12. Matrix multiplication with batch dimensions

`A @ B` multiplies the **last two** dimensions and broadcasts the others:

```python
>>> ww = torch.randn(1, 5, 9); alpha = torch.randn(1, 11, 9)        # (C'=1, H=5, K) and (C=1, N1=11, K)
>>> (ww @ alpha.transpose(-1, -2)).shape                             # (1, 5, 9) @ (1, 9, 11)
torch.Size([1, 5, 11])
>>> ww2 = torch.randn(4, 1, 9); alpha2 = torch.randn(4, 11, 9)      # 4 different cities, 1 day each
>>> (ww2 @ alpha2.transpose(-1, -2)).shape
torch.Size([4, 1, 11])
```

The same line serves both "one fixed city, many days" and "many cities, one day each". That's exactly
how the generator computes Σ_mn α_imn w_m w_n for every customer and day at once:

<!-- quote: svrp/env/generator.py -->
```python
weather_dem = ww @ city.alpha_dem.transpose(-1, -2) - city.shift_dem[:, None, :]        # (C', H, N+1)
```

## 13. Leaving the tensor world: `.item()`, `.tolist()`, `.cpu().numpy()`

- `x.item()`: one-element tensor → Python number (`env.total_cost.mean().item()`).
- `x.tolist()`: tensor → nested Python lists.
- `x.cpu().numpy()`: GPU tensor → NumPy array. The classical solvers are NumPy/OR-Tools code, so data
  crosses over here (`runner.py`, `evaluate.py`).
- `x.to(device)`: move to GPU/CPU. Operations need all their tensors on the **same device**.

## 14. Gradients in one paragraph (more in lesson 11)

Tensors produced by a neural network remember how they were computed, so that `loss.backward()` can
compute gradients. Three tools control this:

- `x.detach()`: "treat this as a constant" (the baseline in REINFORCE must not receive gradients).
- `with torch.no_grad():` / `@torch.no_grad()`: don't record anything. Faster and uses less memory; used for
  evaluation and the heuristics.
- **In-place edits** (`x += 1`, `x[idx] = v`) on tensors that need gradients can break `backward()`. The
  environment only edits its *own* state tensors (load, remaining, …), which never need gradients, so
  that's safe.

## 15. Reproducible randomness: `torch.Generator`

```python
>>> g = torch.Generator(); g.manual_seed(0); torch.rand(3, generator=g)
tensor([0.4963, 0.7682, 0.0885])
>>> g = torch.Generator(); g.manual_seed(0); torch.rand(3, generator=g)
tensor([0.4963, 0.7682, 0.0885])
```

Same seed → same numbers, every time, on every PC. The project gives each *role* its own seed (the city,
the history, the validation days, the test days, the training stream), so test days are identical for
every method and every run. Lesson 05 shows why that makes comparisons fair.

---

## Exercises (predict first, then run to check)

1. `x` has shape (512, 21, 21). What is the shape of `x[:, 0]`? Of `x[..., 0]`? Of `x.flatten(-2)`?
2. `pos` has shape (512,), `e` has shape (512, 21, 128), `ar = torch.arange(512)`. Shape of `e[ar, pos]`?
3. `a` has shape (8, 1), `b` has shape (1, 5). Shape of `a + b`? What about shapes (8,) and (5,)?
4. Write one line that, given `remaining` (B, N1), returns a (B,) bool tensor "all customers are served"
   (ignore the depot column 0; use a small EPS).
5. `logits` (B, N1), `mask` (B, N1). Write the line giving the greedy action per day.
6. A batch of 64 days is `repeat_interleave(8)`'d into 512 rows. You get `cost` (512,). How do you get the
   mean cost of the 8 copies of each day, as shape (64, 1)?
7. Why would `d.transpose(0, 1).view(6)` fail, while `.reshape(6)` works?

**Answers** (try the questions first, then check here)

1. (512, 21): the first *row* of every matrix (from node 0). (512, 21): the first *column* (to node 0).
   (512, 441).
2. (512, 128).
3. (8, 5). Shapes (8,) and (5,) cannot broadcast: aligned from the right, 8 ≠ 5 and neither is 1.
4. `~(remaining[:, 1:] > EPS).any(1)`, which is the same as `(remaining[:, 1:] <= EPS).all(1)`. The
   environment uses the first form.
5. `logits.masked_fill(~mask, float("-inf")).argmax(-1)`.
6. `cost.view(-1, 8).mean(1, keepdim=True)` (the `-1` means "work out this size": 512 / 8 = 64).
7. After `transpose`, the numbers are no longer laid out row by row in memory, so `view` refuses.
   `reshape` copies them into a new layout when it must.


## Read more

- PyTorch tutorial, *Learn the Basics → Tensors*:
  [pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- NumPy's broadcasting rules (PyTorch's are identical):
  [numpy.org/doc/stable/user/basics.broadcasting.html](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- PyTorch docs for `gather`, `topk`, `masked_fill`, `where`: search "torch.gather" etc. on pytorch.org.

**Next:** [03 — Config](03_config.md)
