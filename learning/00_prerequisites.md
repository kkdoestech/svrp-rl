# 00 — Prerequisites: the Python, probability and matrix ideas this project uses

| | |
|---|---|
| **Layer** | 0 (foundation) |
| **You build** | nothing yet; you learn the "alphabet" |
| **Time** | 3–6 hours (skim what you already know) |
| **Before this** | nothing |

## Why this lesson exists

The project code is short (about 1,500 lines), but it is *dense*. One line like

```python
s = q @ k.transpose(-1, -2) / math.sqrt(d)
```

uses a Python operator, a tensor method and a piece of math at once. If any of the three is unfamiliar,
the line looks like magic. This lesson gives you the three alphabets, **using examples taken from the
project itself**, so later lessons can focus on the ideas.

If you already know a section, skip to the self-test at the end and check yourself.

---

## Part A — Python features used in the project

### A1. Modules, packages and imports

A **module** is one `.py` file. A **package** is a folder of modules that contains an `__init__.py` file
(it may be empty, and in this project it always is).

```
svrp/                 <- package "svrp"
├── __init__.py       <- makes the folder importable
├── config.py         <- module "svrp.config"
└── env/              <- sub-package "svrp.env"
    ├── __init__.py
    └── generator.py  <- module "svrp.env.generator"
```

So `from svrp.config import EnvConfig` means "open `svrp/config.py` and give me the name `EnvConfig`".
`from svrp.env import generator as G` means "give me the whole module, but call it `G` here", which is
why `data.py` can write `G.make_cities(...)`.

### A2. Functions, default arguments, keyword arguments, `**kwargs`

```python
def ortools_cvrp(dem, cost, capacity, time_limit=1.0, metaheuristic="gls", return_objective=False):
```

- `time_limit=1.0` is a **default**. If the caller doesn't pass it, it is 1.0.
- Callers can pass arguments **by name**: `ortools_cvrp(d, c, 20, metaheuristic="tabu")`.
- `**kw` in a call *spreads a dictionary into keyword arguments*: if `kw = {"time_limit": 0.5}`, then
  `f(x, **kw)` is the same as `f(x, time_limit=0.5)`. `runner.py` uses this to pass solver settings along.

### A3. Type hints

```python
def clip_cost(c: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
```

`c: torch.Tensor` says "c should be a tensor". `-> torch.Tensor` says "this returns a tensor". Python
**does not check** these; they are notes for humans and for your editor (VS Code shows them on hover).
`int | None` means "an int or None". The line `from __future__ import annotations` at the top of many
files lets you write hints that mention classes not yet defined. Treat it as standard boilerplate.

### A4. Classes, `self`, methods, `@property`

```python
class SVRPEnv:
    def __init__(self, batch, delivery="partial", weather_input=True):
        self.batch = batch          # store data on the object
        ...
    def mask(self):                 # a method: a function that belongs to the object
        ...
```

`env = SVRPEnv(batch)` calls `__init__`. Inside methods, `self` is "this particular environment".
`@property` turns a method into something you read like a field, without brackets:

```python
@property
def capacity(self) -> float:
    return self.mean_demand / self.fill_rate

cfg.capacity      # not cfg.capacity()
```

### A5. `@dataclass`: classes that are mostly data

```python
@dataclass
class EnvConfig:
    n_customers: int = 20
    fill_rate: float = 0.5
```

`@dataclass` writes the boring parts for you: `__init__` (so `EnvConfig(n_customers=10)` works), a
readable print, and `==` comparison. Helpers from the `dataclasses` module that the project uses:

| helper | what it does | used in |
|---|---|---|
| `fields(cls)` | list of the declared fields (name, default, …) | `config.py` builds one CLI flag per field |
| `asdict(obj)` | dataclass → plain `dict` | saving configs to JSON |
| `replace(obj, a=1)` | copy of `obj` with field `a` changed | `Batch.believed()` |

### A6. f-strings and format specs

```python
print(f"epoch {epoch:3d}  {minutes:6.1f} min  val {v:.4f}")
```

Inside `{ }`, after the colon: `3d` = integer, width 3. `6.1f` = float, width 6, 1 decimal. `.4f` = 4
decimals. `:g` = "general" (drops useless zeros: `0.5` not `0.500000`). `{v!r}` = the `repr` (strings
get quotes).

### A7. Comprehensions, `zip`, `enumerate`, `lambda`

```python
plans = [p for p, _ in out]                                 # list comprehension
{f.name: getattr(args, f.name) for f in fields(cls)}        # dict comprehension
for p, L in zip(plans, lens): ...                           # walk two lists together
for i, h in enumerate(hlines): ...                          # index + item
flat = lambda t: t.reshape(n, *t.shape[2:])                 # a tiny unnamed function
```

`dict.fromkeys(list)` is a trick used in `evaluate.py`: it removes duplicates **but keeps the order**.

### A8. `if __name__ == "__main__":`

When you run `python scripts/evaluate.py`, Python sets `__name__ = "__main__"` for that file. When
another file *imports* it, `__name__` is the module name instead. So

```python
if __name__ == "__main__":
    main()
```

means "only run `main()` when this file is the program, not when it is imported". On **Windows this is
required** for anything that starts worker processes (lesson 07 explains why).

### A9. Small standard-library tools

| tool | example from the project | meaning |
|---|---|---|
| `argparse` | `p.add_argument("--n_days", type=int, default=1000)` | read command-line flags |
| `pathlib.Path` | `Path(__file__).resolve().parents[1]` | "the folder above the one this file is in" |
| `with open(...) as fh:` | writing `log.csv` | the file closes automatically, even on errors |
| `csv`, `json` | `json.dumps(cfg, indent=2)` | save tables and configs as text |
| `time.perf_counter()` | timing a solve | a precise stopwatch |
| `raise ValueError(...)` | unknown option names | stop with a clear error message |
| `assert cond, "msg"` | `assert env.done.all()` | crash loudly if something "impossible" happens |

**Resource:** the official [Python tutorial](https://docs.python.org/3/tutorial/), chapters 4–9, covers
all of Part A.

---

## Part B — Probability (just what the paper needs)

### B1. Random variables and distributions

A **random variable** is a number whose value is decided by chance, like tomorrow's demand ξ_i of
customer i. Its **distribution** says how likely each value is.

Two distributions are used everywhere in this project:

**Uniform U(a, b)**: every value between a and b is equally likely.

- The weather variables are `w ~ U(−1, 1)` (temperature, pressure, humidity, scaled to [−1, 1]).
- Customer positions are `U[0, 1]²` (a random point in the unit square).
- Mean demands θ̄_i are `U[5, 15]`.

**Normal N(μ, σ²)** (the "bell curve"): values near μ are most likely, and σ (the *standard deviation*)
is the width. About 68% of values fall within μ ± σ and 95% within μ ± 2σ. The noise ε in the paper's
formula is normal with mean 0.

In PyTorch: `torch.rand(...)` draws U(0, 1), and `torch.randn(...)` draws N(0, 1). To get U(−1, 1), the
code does `2 * torch.rand(...) - 1`. To get N(0, σ²), it does `σ * torch.randn(...)`.

### B2. Expectation (the long-run average)

The **expectation** E[X] is the average value of X if you could repeat the experiment forever.

Key rules (you will use them in lesson 04):

- **Linearity:** E[aX + bY + c] = a·E[X] + b·E[Y] + c, always (even when X and Y are dependent).
- **Independent product:** if X and Y are independent, E[XY] = E[X]·E[Y].

Worked example: for w ~ U(−1, 1), E[w] = 0 (symmetric around 0). And

E[w²] = ∫₋₁¹ w² · ½ dw = ½ · [w³/3]₋₁¹ = ½ · (1/3 + 1/3) = **1/3**,
E[w⁴] = ∫₋₁¹ w⁴ · ½ dw = ½ · (2/5) = **1/5**.

(The ½ is the density of U(−1, 1): total length 2, so height ½.) These two numbers, **1/3 and 1/5**,
appear in `generator.py` as `_E_W2` and `_VAR_W2 = 1/5 − 1/9`.

### B3. Variance (how spread out)

Var(X) = E[(X − E[X])²] = E[X²] − (E[X])².

Rules:

- Var(c·X) = c²·Var(X): scaling by c scales the spread by |c|, the variance by c².
- Var(X + c) = Var(X): shifting doesn't change spread.
- If X and Y are **independent**: Var(X + Y) = Var(X) + Var(Y). (More generally, when they are only
  *uncorrelated*, meaning covariance 0, the same holds.)

Worked example: Var(w²) for w ~ U(−1, 1) is E[w⁴] − (E[w²])² = 1/5 − 1/9 = **4/45**.

The paper's **signal ratio** (see `b.md`) splits the total "size" of a random quantity into three
buckets: base² : Var(weather part) : Var(noise). Lesson 04 makes those buckets exactly 0.6 : 0.2 : 0.2.

### B4. Samples, Monte Carlo and the standard error

Instead of computing E[X] with an integral, you can **simulate**: draw 100,000 samples and average them.
That is the *Monte Carlo* method, and the project's tests use it to check the formulas.

The average of n independent samples wobbles around the true mean by about σ/√n. This is the
**standard error**: with 4× more samples, the wobble halves. It is why the evaluation uses 1,000 test
days, and why "RL is 0.3% better" on 10 days means nothing.

### B5. Percentiles and confidence intervals

The 2.5th percentile is the value below which 2.5% of the data falls. A **95% confidence interval**
[lo, hi] is a range that would contain the true value 95% of the time if you repeated the whole
experiment. Lesson 12 computes these with the *bootstrap* (resample the 1,000 days with replacement,
many times).

**Resource:** Khan Academy, *Statistics and probability* (units on random variables, and sampling
distributions); or the first chapters of *Seeing Theory* (seeing-theory.brown.edu), which is visual
and free.

---

## Part C — Vectors and matrices

### C1. Shapes

A **vector** is a list of numbers, like the weather w = (w₁, w₂, w₃): shape (3,). A **matrix** is a table:
the distance matrix `dist` for 1 depot + 10 customers has shape (11, 11), and `dist[i, j]` is the distance
from node i to node j. It is **symmetric**: `dist[i, j] == dist[j, i]`.

### C2. Dot product and matrix multiplication

Dot product: (1, 2, 3) · (4, 5, 6) = 1·4 + 2·5 + 3·6 = 32.

Matrix product A @ B: entry (i, j) = (row i of A) · (column j of B). Shapes must fit:
(m × k) @ (k × n) → (m × n). In Python the operator is `@`.

### C3. Transpose

Aᵀ swaps rows and columns: (Aᵀ)[i, j] = A[j, i]. A matrix is symmetric when A = Aᵀ. In PyTorch,
`a.transpose(-1, -2)` swaps the last two dimensions.

### C4. The quadratic form wᵀ α w (the heart of the weather effect)

For a vector w and a square matrix α:

wᵀ α w = Σ_m Σ_n α_mn · w_m · w_n.

Worked example with 2 weather variables, w = (1, 2) and α = [[3, 1], [0, 2]]:

= α₁₁·1·1 + α₁₂·1·2 + α₂₁·2·1 + α₂₂·2·2 = 3 + 2 + 0 + 8 = **13**.

Notice that only α₁₂ + α₂₁ matters, never the two separately (both multiply w₁·w₂). So we may replace α
by its **symmetric part** a = (α + αᵀ)/2 without changing anything: a = [[3, 0.5], [0.5, 2]] gives
3 + 2·(0.5·2) + 8 = 13 again. `generator.py` does exactly this.

### C5. Euclidean distance

The distance between points (x₁, y₁) and (x₂, y₂) is √((x₁−x₂)² + (y₁−y₂)²). `torch.cdist(A, B)` computes
it for every pair of points in A and B at once.

**Resource:** 3Blue1Brown, *Essence of Linear Algebra* (YouTube), episodes 1–4 and 9.

---

## Self-test (answers below; try first!)

1. In `from svrp.env import generator as G`, what is `G`?
2. What does `f"{0.123456:.2f}"` print? And `f"{0.5:g}"`?
3. `cfg = EnvConfig(fill_rate=0.25)`, where mean demand is 10. What is `cfg.capacity`, and why no brackets?
4. Why does `evaluate.py` need `if __name__ == "__main__":`?
5. For w ~ U(−1, 1), what are E[w], E[w²] and Var(w)?
6. X has variance 4. What is Var(3X + 7)?
7. You average 100 samples with σ = 10. Roughly how far is your average from the true mean? With 10,000 samples?
8. Compute wᵀ α w for w = (1, −1) and α = [[2, 4], [0, 1]]. Then compute it with the symmetric part of α.
9. Shapes: (5 × 3) @ (3 × 2) = ? Is (3 × 2) @ (5 × 3) allowed?

**Answers** (try the questions first, then check here)

1. The module `svrp/env/generator.py` itself, under the short name `G` (so you write `G.realize(...)`).
2. `0.12` and `0.5`.
3. 10 / 0.25 = **40**. It is a `@property`, so you read it like a field.
4. On Windows, worker processes re-import the script. Without the guard, each worker would start the
   whole evaluation again (lesson 07).
5. 0, 1/3, and Var(w) = E[w²] − E[w]² = 1/3.
6. 3² · 4 = **36** (the +7 doesn't change spread).
7. About σ/√n = 10/10 = **1**. With 10,000 samples: 10/100 = **0.1**.
8. 2·1·1 + 4·1·(−1) + 0·(−1)·1 + 1·(−1)·(−1) = 2 − 4 + 0 + 1 = **−1**. The symmetric part is
   [[2, 2], [2, 1]]: 2 − 2 − 2 + 1 = −1. Same answer.
9. (5 × 2). No: the inner sizes (2 and 5) don't match.


If you got 7 or more right, move on. Otherwise re-read the matching part. These ideas come back in
every lesson.

**Next:** [01 — Setup and tools](01_setup_and_tools.md)
