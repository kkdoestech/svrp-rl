# 16 — Helper scripts, line by line (`check_gpu.py`, `plot.py`, `inspect_policy.py`)

| | |
|---|---|
| **Layer** | 10 (tools around training and evaluation) |
| **Files covered** | `scripts/check_gpu.py` (the benchmark, lines 25–58), `scripts/plot.py` (72 lines), `scripts/inspect_policy.py` (95 lines) |
| **You build** | the same three scripts in `learning/workspace/scripts/` |
| **Time** | 2–3 hours |
| **Before this** | [11 — Training](11_training_reinforce.md), [12 — Evaluation](12_evaluation_and_experiments.md) |

## Why this lesson exists

These three scripts don't change any result, but each answers a question the project needed answered:

- **`check_gpu.py` (benchmark):** how fast is one training step, so how big a batch and how many epochs fit
  in the 1-hour budget? (`LOG.md`, E0.)
- **`plot.py`:** is training still improving, has it plateaued, which learning rate is better? (E2's plot.)
- **`inspect_policy.py`:** *what* did the policy learn, beyond "its cost is lower"? (E2's behaviour table.)

---

## `scripts/check_gpu.py`, lines 25–58: the benchmark

(Lines 1–22, the GPU check itself, are in [lesson 01](01_setup_and_tools.md).)

<!-- quote: scripts/check_gpu.py -->
```python
def bench(n, batch_size, device, steps=5):
    env = EnvConfig(n_customers=n)
    tr = TrainConfig(batch_size=batch_size)
    world, gen = World(env), train_generator(0)
    policy = make_policy(tr, env).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    times, gen_times = [], []
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 1):
        t0 = time.perf_counter()
        b = world.sample(batch_size, gen).to(device)
        t1 = time.perf_counter()
        r = rollout(policy, b, decode="sample")
        loss = (r.cost.detach() * r.logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        if i:                                   # skip warm-up
            times.append(time.perf_counter() - t0)
            gen_times.append(t1 - t0)
```

`bench(n, batch_size, device, steps=5)` times **one real training step**, exactly as `train()` does it, but
without the bookkeeping:

- Build a world with N customers, a training stream, an untrained PaperNet on the device, and an optimiser.
- `reset_peak_memory_stats()`: start measuring the GPU's *highest* memory use from now on.
- Repeat `steps + 1` times:
  - time the data generation (`t0` → `t1`), then a sampled rollout, a REINFORCE-style loss, backward and
    the optimiser step;
  - `torch.cuda.synchronize()` waits for the GPU to really finish before the clock is read (GPU work is
    asynchronous, lesson 12);
  - `if i:` skips the first repetition, the **warm-up**. The first CUDA call is slow, because it loads
    kernels and allocates memory, and would distort the average.
- `r.cost.detach() * r.logp` is a REINFORCE loss without a baseline. For *timing* it doesn't matter which
  baseline is used.

<!-- quote: scripts/check_gpu.py -->
```python
    per_step = sum(times) / steps
    mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
    epoch_s = per_step * 10240 / batch_size
    print(f"  N={n:3d} batch={batch_size:4d} {device:4s}: {per_step*1000:7.0f} ms/step "
          f"(data {sum(gen_times)/steps*1000:4.0f} ms)  peak {mem:6.0f} MB  ->  {epoch_s:5.1f} s/epoch, "
          f"{3600/epoch_s:5.0f} epochs/hour")


print("\nPaperNet REINFORCE step benchmark (1 epoch = 10,240 days):")
for n, bs in [(10, 512), (20, 256), (20, 512), (50, 128)]:
    bench(n, bs, "cuda")
bench(10, 512, "cpu")
```

- The average time per step, the peak memory in MB (`max_memory_allocated()` / 2²⁰), and the derived
  **seconds per epoch** (10,240 days / batch size steps) and **epochs per hour**.
- The last lines run four GPU settings and one CPU setting. The table in `LOG.md` (E0) came from here, and
  so did the decision "batch 512 for N ≤ 20": a step at 512 takes barely longer than at 256 (195 ms vs
  170 ms), so it processes twice the data for free.

---

## `scripts/plot.py`: learning curves

<!-- quote: scripts/plot.py -->
```python
"""Learning curves: validation cost vs training time, with optional baseline reference lines.

    uv run python scripts/plot.py --runs runs/A runs/B --labels "critic" "shared" \
        --hline "OR-Tools GLS=16.12" --hline "ACO=16.30" --out n20_baselines
Writes results/plots/<out>.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def read_log(run):
    with open(Path(run) / "log.csv") as fh:
        rows = list(csv.DictReader(fh))
    return [float(r["minutes"]) for r in rows], [float(r["val_cost"]) for r in rows]
```

- The docstring: an example command, and where the picture goes.
- `matplotlib.use("Agg")` must come **before** `import matplotlib.pyplot`. "Agg" draws into memory and
  files without opening a window. That's needed when the script runs in a terminal or a background job.
- `ROOT`: the project folder (two levels up from this file), so the output always lands in `results/plots/`.
- **Colours as named constants.** `SERIES` is an ordered list of line colours (series 1 is always blue, 2
  orange, …). `INK`, `INK2`, `MUTED`, `GRID`, `AXIS` and `SURFACE` are the text, axis and background
  colours. Naming them once keeps every plot consistent, and they're easy to change in one place.
- `read_log(run)`: read a run's `log.csv` with `csv.DictReader` (each row becomes a dict keyed by the header
  names) and return two lists, minutes and validation cost.

<!-- quote: scripts/plot.py -->
```python
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="*")
    p.add_argument("--hline", action="append", default=[], help="label=value")
    p.add_argument("--title", default="Validation cost during training (lower is better)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    labels = a.labels or [Path(r).name for r in a.runs]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
```

- Flags: one or more `--runs`, optional `--labels` (otherwise the folder names are used), any number of
  `--hline "name=value"` (`action="append"` collects repeated flags into a list), a title, and the output name.
- `plt.subplots(figsize=(8, 4.5), dpi=150)`: one chart, 8 × 4.5 inches at 150 dots per inch (1200 × 675
  pixels). The figure and the plotting area both get the surface colour.

<!-- quote: scripts/plot.py -->
```python
    for i, (run, lab) in enumerate(zip(a.runs, labels)):
        x, y = read_log(run)
        ax.plot(x, y, color=SERIES[i], lw=2, label=lab, solid_capstyle="round")
        best = min(range(len(y)), key=y.__getitem__)
        ax.plot(x[best], y[best], "o", ms=6, color=SERIES[i], mec=SURFACE, mew=2)
        ax.annotate(f"{lab}  {y[best]:.3f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    for i, h in enumerate(sorted(a.hline, key=lambda s: -float(s.rsplit("=", 1)[1]))):
        name, val = h.rsplit("=", 1)
        ax.axhline(float(val), color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.annotate(f"{name}  {float(val):.3f}", (0, float(val)), xytext=(4, 3 if i % 2 == 0 else -11),
                    textcoords="offset points", fontsize=8, color=INK2, xycoords=("axes fraction", "data"))
```

- **One line per run**, 2 px wide, with rounded ends. `zip(a.runs, labels)` walks the runs and their labels
  together, and `enumerate` gives the index for the colour.
- **The best epoch** gets a dot with a surface-coloured ring (`mec`/`mew`), so it stands out from the line.
  `min(range(len(y)), key=y.__getitem__)` = "the index of the smallest y".
- **A direct label** at the end of each line ("lr 1e-3  6.069"), so the reader doesn't need to look up the
  legend.
- **Baseline lines:** dashed and muted grey (they're references, not data), sorted from highest to lowest.
  The labels go alternately above and below the line (`3 if i % 2 == 0 else -11`), so two close values
  don't print on top of each other. That was a real problem in the first version of E2's plot.
  `xycoords=("axes fraction", "data")` means x is measured as a fraction of the axis width (0 = the left
  edge) and y in data units (the baseline's cost).

<!-- quote: scripts/plot.py -->
```python
    ax.set_xlabel("training time (minutes)", color=INK2, fontsize=9)
    ax.set_ylabel("mean cost on validation days", color=INK2, fontsize=9)
    ax.set_title(a.title, color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    if len(a.runs) >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    out = ROOT / "results" / "plots" / f"{a.out}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
```

- Axis labels and a left-aligned title.
- Light horizontal grid lines *behind* the data (`set_axisbelow`). The top and right borders are removed,
  and the remaining ones are muted. Less ink on decoration, more attention on the curves.
- A legend only when there are ≥ 2 runs (one line needs no legend; its direct label is enough).
- `tight_layout()` fits everything inside the figure. Save into `results/plots/<out>.png` (the folder is
  created if missing) and print the path.

---

## `scripts/inspect_policy.py`: what did the policy learn?

<!-- quote: scripts/inspect_policy.py -->
```python
"""What did the policy learn?  Behaviour statistics of a trained policy vs the heuristics, plus one example day.

    uv run python scripts/inspect_policy.py --run runs/E2_n10_lr1e-3 --n_days 1000
"""
import argparse

import torch

from svrp.data import World
from svrp.env.svrp_env import EPS, SVRPEnv
from svrp.inference import masked_log_probs
from svrp.train.reinforce import load_policy, resolve_device
```

Run it on a trained run folder. It needs the env (to drive) and `EPS`, `masked_log_probs` (to turn the
network's logits into a greedy choice), and `load_policy`.

<!-- quote: scripts/inspect_policy.py -->
```python
@torch.no_grad()
def drive(choose, batch, delivery, weather_input=True):
    """Runs a decision rule and records behaviour: trips, splits, load left when going home."""
    env = SVRPEnv(batch, delivery, weather_input)
    state = choose.init(env) if hasattr(choose, "init") else None
    B = batch.size
    trips = torch.zeros(B, device=env.pos.device)
    early_home = torch.zeros(B, device=env.pos.device)       # went home with load still on the truck
    left_when_home = torch.zeros(B, device=env.pos.device)
    routes = [[0] for _ in range(min(B, 1))]
    for _ in range(env.max_steps()):
        if env.done.all():
            break
        a, state = choose(env, state)
        going_home = (a == 0) & (env.pos != 0) & ~env.done
        trips += going_home
        early_home += going_home & (env.load > EPS)
        left_when_home += torch.where(going_home, env.load, 0.0)
        if not env.done[0]:
            routes[0].append(int(a[0]))
        env.step(a)
    return env, {"cost": env.total_cost.mean().item(), "trips": trips.mean().item(),
                 "splits (load ran out at a customer)": env.n_fail.mean().item(),
                 "trips ending with load left": (early_home.sum() / trips.sum()).item(),
                 "avg load left when going home": (left_when_home.sum() / trips.sum().clamp_min(1)).item()}, routes[0]
```

`drive(choose, batch, …)` is `rollout` from lesson 10 with **behaviour counters** added. `choose` can be
any decision rule: the RL policy or a heuristic.

- `state = choose.init(env) if hasattr(choose, "init") else None`: rules that have memory (the network)
  provide an `init`, and rules without memory (the heuristics) don't. `hasattr` checks for the method.
- **Counters, one per day:**
  - `trips`: how many times the truck drove home;
  - `early_home`: drove home *with load still on the truck*;
  - `left_when_home`: how much load was left.
- `going_home`: the chosen action is the depot, the truck is *not* already there, and the day isn't finished.
- `routes[0]` records day 0's route (for printing an example). `min(B, 1)` makes it work even for a batch of 0.
- The returned statistics:
  - mean cost;
  - trips per day;
  - recourse events (here called *splits*, because under partial delivery that's what they are);
  - the fraction of trips that ended early, and the average load left. `clamp_min(1)` avoids dividing by
    zero.

<!-- quote: scripts/inspect_policy.py -->
```python
class Policy:
    def __init__(self, net):
        self.net = net

    def init(self, env):
        return self.net.init_state(env)

    def __call__(self, env, state):
        logits, state = self.net(env, state)
        return masked_log_probs(logits, env.mask()).argmax(-1), state


def nearest(split):
    def choose(env, state):
        m = env.mask()
        cust = m.clone()
        cust[:, 0] = False
        c = env.batch.cost_hat[env.ar, env.pos]
        fits = cust if split else cust & (env.observed_demand() <= env.load[:, None] + EPS)
        inf = torch.tensor(float("inf"), device=c.device)
        nf, na = torch.where(fits, c, inf).argmin(1), torch.where(cust, c, inf).argmin(1)
        a = torch.where(fits.any(1), nf, torch.where((env.pos == 0) & cust.any(1), na, 0))
        return a, state
    return choose
```

Two ways to make "decision rules" with the same calling convention, `choose(env, state) → (action, state)`:

- **`Policy`, a class with `__call__`.** An object can be called like a function if it defines
  `__call__`. It wraps the network: logits → masked log-probs → `argmax` (greedy).
- **`nearest(split)`, a function that returns a function** (a *closure*). The inner `choose` remembers the
  `split` value it was created with. It's the same rule as `nn_online` (lesson 08), rewritten to fit
  `drive`'s convention. So `nearest(False)` is nn_online and `nearest(True)` is nn_split.

This "same interface, different implementations" idea is what lets one `drive` function measure all three.

<!-- quote: scripts/inspect_policy.py -->
```python
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n_days", type=int, default=1000)
    a = p.parse_args()
    device = resolve_device("auto")
    net, env_cfg, _, _ = load_policy(a.run, device)
    batch = World(env_cfg).eval_set("test", a.n_days).to(device)
    Q = env_cfg.capacity
    print(f"world {env_cfg.tag()}   Q = {Q:g}\n")
    rows = {}
    for name, rule in [("nn_online", nearest(False)), ("nn_split", nearest(True)), ("RL greedy", Policy(net))]:
        _, stats, route = drive(rule, batch, env_cfg.delivery, env_cfg.weather_input)
        rows[name] = (stats, route)
    keys = list(next(iter(rows.values()))[0])
    print(f"{'':38s}" + "".join(f"{n:>12s}" for n in rows))
    for k in keys:
        print(f"{k:38s}" + "".join(f"{rows[n][0][k]:12.3f}" for n in rows))

    b = batch.index(slice(0, 1))
    print("\nExample day 0 — customer: estimate -> true demand")
    print("  " + "  ".join(f"{i}: {b.dem_hat[0, i]:.1f}->{b.demand[0, i]:.1f}" for i in range(1, b.n_nodes)))
    for n, (_, route) in rows.items():
        print(f"  {n:10s} route: {' '.join(map(str, route))}")


if __name__ == "__main__":
    main()
```

- Load the run. Its saved config gives the world, so the script needs no world flags. Use the **test** days
  (this is a report, not a choice).
- Drive all three rules on *the same* days, then print a table: one row per statistic, one column per
  rule. `next(iter(rows.values()))[0]` = "the stats dict of the first rule", used only to get the row names.
- **The example day:** each customer's belief → truth (e.g. `7: 10.0->20.4`), then each rule's route
  through day 0. Reading three routes for the same day side by side is the fastest way to *see* the
  difference between greedy and learned behaviour.

The project's output for the N = 10 model is the behaviour table in `LOG.md` (E2) and in
[lesson 12](12_evaluation_and_experiments.md).

## Rebuild it yourself

1. `check_gpu.py`'s `bench`: run it and fill in your own E0 table.
2. `plot.py`: start with one plain line per run, then add the best-dot, the direct labels, the baselines and
   the styling.
3. `inspect_policy.py`: `drive` with only the cost first, then the counters, then the `Policy`/`nearest`
   rules and the report.

## Verify

```bash
uv run python -m scripts.check_gpu
uv run python -m scripts.plot --runs runs/my_E2_lr1e-3 --hline "Clarke-Wright (val)=8.007" --out my_E2
uv run python -m scripts.inspect_policy --run runs/my_E2_lr1e-3
```

- `check_gpu`: compare your step times with E0 (they must be in the same ballpark on the same GPU, if
  nothing else is running).
- `plot`: open `learning/workspace/results/plots/my_E2.png` and compare it with `results/plots/E2_n10_lr.png`.
- `inspect_policy`: the nn_online and nn_split columns must match E2 exactly (8.028 and 7.867; they don't
  depend on training). The RL column will be close to E2's.

## Check your understanding

1. Why does the benchmark skip the first repetition?
2. Why must `matplotlib.use("Agg")` come before `import matplotlib.pyplot`?
3. What does `nearest(True)` return, and why is it written as a function returning a function?

**Answers** (try the questions first, then check here)

1. The first CUDA operations include one-time setup (loading kernels, allocating memory), which would make
   the average step look slower than it really is.
2. `pyplot` picks its drawing backend when it's imported. Choosing "Agg" afterwards may be too late.
3. A decision rule (`choose`) that always splits. Returning a function lets one definition produce both
   variants with the same `choose(env, state)` interface that `drive` expects.

**Next:** [17 — Experiment scripts and project documents](17_experiment_scripts_and_docs.md)
