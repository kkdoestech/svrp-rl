# 03 — Config: describing one experiment completely (`svrp/config.py`)

| | |
|---|---|
| **Layer** | 1 |
| **Files covered** | `svrp/config.py` (105 lines) |
| **You build** | `learning/workspace/mysvrp/config.py` |
| **Time** | ~1 hour |
| **Before this** | [00](00_prerequisites.md) (dataclasses, `@property`, argparse) |

## Why this layer exists

The paper tests its AI under many "difficulty settings" (`part 6.md`): signal ratio, fill rate, fixed or
flexible positions, partial or full delivery, constant or k-NN estimates. Every experiment is one
combination of these settings.

If the settings were scattered through the code as loose numbers, two runs could silently differ and
you'd never know why their results differ. So the project puts **every setting in one place**, two
small dataclasses:

- `EnvConfig`, the **world**: how many customers, how noisy, which delivery rules.
- `TrainConfig`, the **learner**: which model, learning rate, batch size, time budget.

Both are saved as JSON next to every trained model, so any number in `LOG.md` can be traced back to its
exact settings.

## Concepts from zero

- **Configuration object**: one object holding all the knobs. Functions receive `cfg` and read
  `cfg.n_customers` instead of taking 20 separate arguments.
- **Derived value**: something computed from other settings, like the truck capacity Q = mean demand / Φ.
  Make it a `@property`, so it can never get out of sync with the settings it comes from.
- **Tag**: a short unique text name for a world, e.g. `n10_phi0.5_r0.6-0.2-0.2_d1c1_fixed_partial_knn_k10_s0`.
  Results are stored in folders named after the tag, so results from different worlds never mix.

## Code walkthrough

### Lines 1–10: docstring and imports

<!-- quote: svrp/config.py -->
```python
"""Configuration dataclasses + a tiny helper that turns them into command-line flags.

Every experiment is fully described by an EnvConfig (the "world") and a TrainConfig (the "learner").
Both are saved next to every checkpoint, so any result can be traced back to its exact settings.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
```

Only standard-library imports: this file is the bottom layer and depends on nothing else in the project.

### Lines 12–31: the world's settings

<!-- quote: svrp/config.py -->
```python
@dataclass
class EnvConfig:
    n_customers: int = 20
    fill_rate: float = 0.5           # Φ = E[ξ_i] / Q  ->  Q = mean demand / Φ
    a_ratio: float = 0.6             # A: share of the constant (base) part
    b_ratio: float = 0.2             # B: share explained by weather
    gamma_ratio: float = 0.2         # Γ: share that is pure noise
    stochastic_demand: bool = True
    stochastic_cost: bool = True
    positions: str = "fixed"         # fixed (same city every day, B2B) | flexible (new city every day, B2C)
    delivery: str = "partial"        # partial (split allowed) | full (indivisible orders)
    estimate: str = "knn"            # constant | knn | oracle_mean
    weather_input: bool = True       # does the network see w?
    knn_k: int = 10
    history_days: int = 1000         # history bank size for a fixed city
    history_days_flexible: int = 100 # per-instance history bank when positions are flexible
    n_weather: int = 3
    demand_low: float = 5.0
    demand_high: float = 15.0
    city_seed: int = 0
```

Each line is `name: type = default`. Here is each setting and where it comes from in the paper:

| field | meaning | paper link |
|---|---|---|
| `n_customers` | N, the number of customers (plus 1 depot) | Table 2 uses 10, 20, 50, 100 |
| `fill_rate` | Φ: how much of the truck one customer's order fills on average. 0.5 → Q = 20 → ~2 customers per trip | "fill rate" in `part 6.md` |
| `a_ratio`, `b_ratio`, `gamma_ratio` | A, B, Γ: shares of base / weather / noise | `b.md` (signal ratio) |
| `stochastic_demand`, `stochastic_cost` | switch the randomness of demand / travel cost on or off | finding #2 in `part 6.md` |
| `positions` | `fixed`: same customers every day (B2B). `flexible`: new random customers every day (B2C) | finding #4 |
| `delivery` | `partial`: hand over what you have, come back later. `full`: orders are indivisible | finding #5 |
| `estimate` | how the dispatcher guesses demand before visiting: constant average, k-NN on weather, or a perfect "oracle" | finding #3 |
| `weather_input` | is the weather vector given to the network at all? | used in the "weather value" experiment |
| `knn_k` | k in k-NN (the paper doesn't say; we use 10) | Section 6 |
| `history_days*` | how many past days the k-NN can look up | — |
| `n_weather` | 3: temperature, pressure, humidity | `a.md` |
| `demand_low/high` | θ̄_i ~ U[5, 15]: mean demand of each customer | — |
| `city_seed` | which random city (change it to test on a different map) | — |

The comment on `fill_rate` notes a deviation: the paper prints Φ = Q / E[ξ]. Taken literally, Φ = 0.5
would mean the truck carries half of *one* order, which is impossible, so the project reads it the
other way round (`SPEC.md` explains this).

### Lines 32–39: the extra "event" variable (improvement I4)

<!-- quote: svrp/config.py -->
```python
    # I4 (Part 7, limitation #2): a second external variable besides weather. On an "event" day
    # (concert, football match, ...) customers within event_radius of a fixed venue order
    # event_boost × θ̄_i extra. event_input says whether the event flag is visible to the agent
    # and to the k-NN lookup. event_prob = 0 switches the feature off (all earlier worlds unchanged).
    event_prob: float = 0.0
    event_boost: float = 1.0
    event_radius: float = 0.3
    event_input: bool = True
```

These four fields were added later, for lesson 13. **When you first write your config, you may leave
them out** and add them in lesson 13. Notice the design rule in the comment: the new feature defaults
to *off*, so every earlier experiment stays exactly the same.

### Lines 41–52: derived values (`@property`)

<!-- quote: svrp/config.py -->
```python
    @property
    def n_external(self) -> int:
        """Size of the observable external-variable vector: weather (+ the event flag)."""
        return self.n_weather + (1 if self.event_prob > 0 else 0)

    @property
    def mean_demand(self) -> float:
        return 0.5 * (self.demand_low + self.demand_high)

    @property
    def capacity(self) -> float:
        return self.mean_demand / self.fill_rate
```

- `n_external`: the length of the "outside information" vector the network sees: 3 weather numbers,
  plus 1 event flag when events are on. The networks use it to size their input layer.
- `mean_demand`: the middle of U[5, 15] = 10.
- `capacity`: **Q = 10 / Φ**. Default Φ = 0.5 gives Q = 20; Φ = 0.2 gives Q = 50 (big trucks, ~5
  customers per trip).

Why is Q computed from the *nominal* mean (10) and not from the actual sampled θ̄ of the city? So that Q
is the same number for every city and every seed. Comparing worlds is simpler when the truck doesn't
change size.

### Lines 54–62: the world's name tag

<!-- quote: svrp/config.py -->
```python
    def tag(self) -> str:
        """Short unique name for this world, used for result/caching folders."""
        r = f"{self.a_ratio:g}-{self.b_ratio:g}-{self.gamma_ratio:g}"
        sto = f"d{int(self.stochastic_demand)}c{int(self.stochastic_cost)}"
        ev = ""
        if self.event_prob > 0:
            ev = f"_ev{self.event_prob:g}x{self.event_boost:g}r{self.event_radius:g}" + ("" if self.event_input else "h")
        return (f"n{self.n_customers}_phi{self.fill_rate:g}_r{r}_{sto}_{self.positions}_"
                f"{self.delivery}_{self.estimate}_k{self.knn_k}_s{self.city_seed}{ev}")
```

- `r` → `"0.6-0.2-0.2"` (`:g` drops trailing zeros).
- `sto` → `"d1c1"`: demand stochastic = 1, cost stochastic = 1. `int(True)` is 1.
- `ev` is empty unless events are on, so the old tags don't change (the same "don't break the past" rule).
- The result for the defaults with N = 10 is `n10_phi0.5_r0.6-0.2-0.2_d1c1_fixed_partial_knn_k10_s0`.
  Look in `results/eval/`: you'll find folders with exactly these names.

Why isn't `weather_input` in the tag? The tag names the **days and the beliefs** (what the classical
planners see). Hiding the weather from the *network* doesn't change the days or the planners' results,
so both runs can share one cached set of baseline results.

### Lines 65–81: the learner's settings

<!-- quote: svrp/config.py -->
```python
@dataclass
class TrainConfig:
    model: str = "paper"             # paper | graph
    baseline: str = "critic"         # critic | shared
    n_shared: int = 8                # M rollouts per instance for the shared baseline
    embed_dim: int = 128
    batch_size: int = 256
    epochs: int = 50
    epoch_size: int = 10240
    lr: float = 1e-4
    critic_lr: float = 1e-4
    grad_clip: float = 1.0
    val_size: int = 1000
    seed: int = 1
    device: str = "auto"             # auto | cuda | cpu
    time_limit_min: float = 60.0     # stop training after this many minutes (the 1-hour budget)
    run_name: str = ""
```

You'll understand each of these in lesson 11. For now: `model` picks the network, `baseline` picks the
REINFORCE baseline (`part 5.md`'s b_φ), `lr` is the learning rate, `epoch_size = 10240` days ≈ the paper's
10,000 training scenarios, and `time_limit_min` enforces the "≤ 1 hour per run" budget.

The defaults (`lr = 1e-4`, `batch_size = 256`) are *not* what the experiments ended up using. The
experiments pass `--lr 1e-3 --batch_size 512` (see `experiments/queue_1.sh`), because the learning-rate
experiment E2 showed 1e-4 is far too slow. The defaults were left alone so that old runs keep their
meaning.

### Lines 84–89: reading true/false from the command line

<!-- quote: svrp/config.py -->
```python
def _str2bool(v: str) -> bool:
    if v.lower() in ("1", "true", "yes", "y"):
        return True
    if v.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")
```

**A classic Python trap:** `argparse` with `type=bool` turns *any* non-empty string into True, because
`bool("False")` is True. So `--weather_input false` would silently mean True. This helper parses the text
properly and refuses nonsense like `--weather_input maybe`.

### Lines 92–105: one flag per field, automatically

<!-- quote: svrp/config.py -->
```python
def add_args(parser: argparse.ArgumentParser, cls) -> None:
    """Add one --flag per dataclass field (e.g. --n_customers 10 --delivery full)."""
    for f in fields(cls):
        default = f.default
        kind = bool if isinstance(default, bool) else type(default)
        parser.add_argument(f"--{f.name}", type=_str2bool if kind is bool else kind, default=default)


def from_args(args: argparse.Namespace, cls):
    return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


def to_dict(cfg) -> dict:
    return asdict(cfg)
```

- `add_args` loops over the dataclass fields. For each, it reads the default value, takes its type
  (`type(20)` is `int`, `type("fixed")` is `str`), and registers `--name` with that type. Bools get the
  safe parser. Result: **every field is automatically a command-line flag**. Add a field, and you get a
  flag for free.
- `from_args` does the reverse: it collects the parsed values back into a dataclass.
  `getattr(args, "n_customers")` is the same as `args.n_customers`. The dict comprehension builds
  `{"n_customers": 10, ...}`, and `cls(**that_dict)` calls `EnvConfig(n_customers=10, ...)`.
- `to_dict` turns a config into a plain dict, for `json.dumps`.

`scripts/train.py` (lines 14–16) shows the payoff. It needs only these three lines to accept ~35 flags:

<!-- quote: scripts/train.py -->
```python
p = argparse.ArgumentParser()
add_args(p, EnvConfig)
add_args(p, TrainConfig)
```

Notice: `EnvConfig` and `TrainConfig` must never have a field with the same name, or argparse would
complain about a duplicate flag.

## Rebuild it yourself

1. Create `learning/workspace/mysvrp/config.py`.
2. Write `EnvConfig` with the fields of lines 13–31 (skip the event fields for now). Try to remember what
   each one is for. Write the comments too: they are part of understanding.
3. Add the properties `mean_demand` and `capacity`. Add `n_external` returning just `self.n_weather` for
   now.
4. Write `tag()` without the event part.
5. Write `TrainConfig`, `_str2bool`, `add_args`, `from_args`, `to_dict`.

## Verify

There is no test file for config alone. Run this check from `learning/workspace`:

```bash
uv run python -c "
import sys, argparse; sys.path.insert(0, '.')
from mysvrp.config import EnvConfig, TrainConfig, add_args, from_args
c = EnvConfig(n_customers=10)
print(c.capacity, EnvConfig(fill_rate=0.2).capacity)
print(c.tag())
p = argparse.ArgumentParser(); add_args(p, EnvConfig); add_args(p, TrainConfig)
a = p.parse_args(['--n_customers', '7', '--weather_input', 'false', '--lr', '0.001'])
print(from_args(a, EnvConfig).n_customers, from_args(a, EnvConfig).weather_input, from_args(a, TrainConfig).lr)
"
```

Expected:

```
20.0 50.0
n10_phi0.5_r0.6-0.2-0.2_d1c1_fixed_partial_knn_k10_s0
7 False 0.001
```

If the tag differs, the result folders won't match the project's. Compare it character by character.

## Common mistakes

- `type=bool` in argparse: `--weather_input false` becomes True. Use `_str2bool`.
- Writing `capacity` as a normal field: then `EnvConfig(fill_rate=0.2)` keeps Q = 20. Wrong.
- A mutable default like `my_list: list = []` makes dataclasses raise an error. You'd need
  `field(default_factory=list)`. This project never needs one.

## Check your understanding

1. With Φ = 0.25, what is Q, and roughly how many customers fit in one trip?
2. Why is `weather_input` not part of `tag()`, while `estimate` is?
3. What does `from_args(args, EnvConfig)` return, and why does it need `fields(cls)`?
4. You add a field `traffic_level: float = 0.0` to `EnvConfig`. What else must change before
   `--traffic_level 0.5` works on the command line?

**Answers** (try the questions first, then check here)

1. Q = 10 / 0.25 = 40. About 4 customers of average demand 10.
2. The tag identifies the days and the planners' beliefs, which is what the cached baseline results
   depend on. `estimate` changes the planners' inputs. `weather_input` only changes what the *network*
   sees.
3. An `EnvConfig` filled with the parsed flag values. `fields(cls)` gives the list of field names to
   collect from `args`.
4. Nothing. `add_args` creates the flag automatically from the new field. (You'd also add it to `tag()`
   if it changes the days.)


## Read more

- Python docs, `dataclasses`: [docs.python.org/3/library/dataclasses.html](https://docs.python.org/3/library/dataclasses.html)
- Python docs, `argparse` tutorial: [docs.python.org/3/howto/argparse.html](https://docs.python.org/3/howto/argparse.html)

**Next:** [04 — The random world](04_generator.md)
