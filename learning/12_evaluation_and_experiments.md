# 12 — Evaluation and experiments: being honest with numbers

| | |
|---|---|
| **Layer** | 10 |
| **Files covered** | `scripts/evaluate.py` (163 lines), `scripts/plot.py` (72), `scripts/inspect_policy.py` (95), `experiments/queue_1.sh`, `experiments/queue_2.sh`, `experiments/baselines_E4.sh`, `experiments/LOG.md` |
| **You build** | `learning/workspace/scripts/evaluate.py`, `plot.py`, `inspect_policy.py`, and your own lab notebook |
| **Time** | 4–6 hours |
| **Before this** | [11 — Training](11_training_reinforce.md) (a trained run in `runs/`) |

## Why this layer exists

You now have a trained policy and a pile of baselines. The tempting next step is to print one average
and say "RL is 24% better!". This lesson is about everything that has to be true *before* you're allowed
to say that:

- every method was scored on **the same days**;
- the difference is **bigger than the noise**;
- the method **didn't peek** at information it can't have;
- the gain isn't explained by something trivial (a **control** experiment);
- you understand **what** the policy learned, not just that its number is lower;
- every experiment, including the failures, is **written down**.

This is the difference between a project and a piece of science. The paper's Part 6 is exactly this
kind of evaluation, and `experiments/LOG.md` is this project's version of it.

## Concepts from zero

### Paired differences and the bootstrap, with numbers

Five test days. Method X and the reference R (OR-Tools):

| day | X | R | X − R |
|---|---|---|---|
| 1 | 8.0 | 8.5 | −0.5 |
| 2 | 9.0 | 9.5 | −0.5 |
| 3 | 7.0 | 7.2 | −0.2 |
| 4 | 10.0 | 10.1 | −0.1 |
| 5 | 6.0 | 6.6 | −0.6 |

The means are 8.00 vs 8.38. **Unpaired**, you'd say "the days vary from 6 to 10, so a difference of 0.38
could be luck". **Paired**, X won on *every single day*. The day-to-day variation cancels in X − R. The
gap is mean(X − R)/mean(R) = −0.38/8.38 = **−4.5%**.

How sure are we? The **bootstrap**: pretend the 1,000 test days are the whole world, and draw 1,000 days
*with replacement* (some days twice, some not at all). Recompute the gap. Repeat 2,000 times. The middle
95% of those 2,000 gaps is the **95% confidence interval**. If it doesn't include 0%, the difference is
real at that confidence level.

### Deployable / lookahead / hindsight (from lesson 10)

Every row of the results table carries this label. A number labeled "hindsight" is never compared with
a "yes" as if they were the same kind of thing.

### Controls and leak checks

When a result looks *too good*, assume a bug until proven otherwise. In this project, RL at N = 10 came out
24% better than OR-Tools, and the paper reports 3.4%. Before believing it, these checks were run
(`LOG.md`, E2):

1. **Leak check:** does the policy see anything it shouldn't? The observation was re-read line by line,
   and the mask argument from lesson 08 confirmed it reveals nothing.
2. **Accounting check:** does the env charge exactly the roads driven? `tests/test_env.py` checks this.
3. **Control experiment:** maybe RL only wins because partial delivery allows splitting, which the CVRP
   planners never plan for. A *dumb* rule that splits (`nn_split`) got only −1.9%. So splitting alone
   doesn't explain −24%.
4. **Behaviour analysis:** *what* is the policy doing differently? (`inspect_policy.py`, below.)

### One change at a time, seeds, and the keep-or-reject rule

- An experiment changes **one thing** compared with a reference run (learning rate, baseline, model,
  world setting). Change two things and you can't tell which one mattered.
- Training is random. Seed 1 and seed 2 of the same config differ (6.07 vs 5.96 at N = 10). A difference
  smaller than the seed-to-seed spread means nothing. So headline comparisons use **3 seeds**.
- **SPEC.md rule S10:** keep a change only if it improves the mean validation cost across seeds by more
  than 2 standard errors, at no larger training budget. Otherwise reject it, and still log it.
- Decide on **validation**; report on **test**.

## Code walkthrough: `scripts/evaluate.py`

### Lines 1–26: the header

<!-- quote: scripts/evaluate.py -->
```python
"""Score methods on the SAME days and print a paired comparison table.  Examples:

    uv run python scripts/evaluate.py --n_customers 20 --methods nn_online,cw,ortools_gls,ortools_tabu,aco,hindsight
    uv run python scripts/evaluate.py --n_customers 20 --runs runs/paper_critic_... --rl_modes greedy,beam3,apriori

Per-day costs are cached in results/eval/<world>/<split>/<method>.npz, so re-running only computes
what is new. The table compares every method to --ref on a per-day basis (paired bootstrap CI).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from svrp.baselines.nn_online import nn_online
from svrp.baselines.runner import PLANNERS, run_planner
from svrp.config import EnvConfig, add_args, from_args
from svrp.data import World
from svrp.inference import apriori, beam_search, best_of_n, rollout
from svrp.train.reinforce import load_policy, resolve_device
```

- The docstring gives two example commands (baselines only; RL runs with their inference modes) and the
  two key behaviours: per-day **caching** and **paired** comparison.
- `datetime` timestamps each row of `results.csv`. `time` measures speed.
- It imports from almost every layer: the heuristic, the runner (and its `PLANNERS` list), config, data,
  the four inference functions, and `load_policy` / `resolve_device`. An evaluation script sits at the
  very top of the layer map.

### Lines 28–47: labels and the paired bootstrap

<!-- quote: scripts/evaluate.py -->
```python
ROOT = Path(__file__).resolve().parents[1]
DEPLOY = {"nn_online": "yes", "nn_split": "yes", "cw": "yes", "ortools_gls": "yes", "ortools_tabu": "yes", "aco": "yes",
          "hindsight": "reference", "greedy": "yes", "apriori": "yes", "beam": "lookahead", "best": "hindsight"}


def label(method: str) -> str:
    key = method.split(":")[-1]
    for k, v in DEPLOY.items():
        if key.startswith(k):
            return v
    return "?"
```

Every method name maps to its honesty label. RL methods are named `rl:<run>:<mode>` (e.g.
`rl:E2_n10_lr1e-3:beam3`), so the label is found from the last part (`beam3` starts with `beam`, giving
"lookahead").

<!-- quote: scripts/evaluate.py -->
```python
def paired_ci(x: np.ndarray, ref: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Mean relative gap (x − ref)/mean(ref) in %, with a 95% bootstrap CI over days."""
    rng = np.random.default_rng(seed)
    d = x - ref
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1) / ref[idx].mean(1) * 100
    return d.mean() / ref.mean() * 100, np.percentile(boots, 2.5), np.percentile(boots, 97.5)
```

- `d = x − ref`: the per-day differences (paired!).
- `idx` has shape (2000, n_days): 2,000 bootstrap samples, each a list of day indices drawn with replacement.
- `d[idx].mean(1) / ref[idx].mean(1)`: the relative gap within each resample, vectorised (no Python loop).
- Return the actual gap and the 2.5th and 97.5th percentiles: the 95% CI. The fixed `seed=0` makes the
  CI itself reproducible.

### Lines 50–87: running one method

<!-- quote: scripts/evaluate.py -->
```python
def run_method(method: str, batch, env_cfg: EnvConfig, device, policies, workers=None) -> dict:
    t0 = time.perf_counter()
    if method in ("nn_online", "nn_split"):
        env = nn_online(batch.to(device), env_cfg.delivery, split=method == "nn_split")
        out = {"cost": env.total_cost.cpu().numpy(), "n_fail": env.n_fail.cpu().numpy()}
    elif method in PLANNERS:
        kw = {}
        if method == "aco":
            kw = json.loads((ROOT / "results" / "aco_params.json").read_text()) \
                if (ROOT / "results" / "aco_params.json").exists() else {}
        out = run_planner(method, batch, env_cfg.delivery, workers=workers, **kw)
        out["sec_total"] = time.perf_counter() - t0
        return out
```

A dispatcher over method kinds:

- `workers` is passed through to the runner (the `--workers` flag).
- `t0` starts the stopwatch.
- **Heuristics** (`nn_online`, `nn_split`) run in the env. `split=method == "nn_split"` is a compact way to
  pass True only for the control.
- **Planners** go through `run_planner` (lesson 07). ACO reads its tuned settings from
  `results/aco_params.json` (lesson 14) if the file exists, and an empty dict (the defaults) otherwise.
- Planners `return` early: `run_planner` already measured `sec_per_instance` itself (pure solve time per
  day, inside the workers). That's more accurate than wall-clock divided by days when 6 processes run at once.

<!-- quote: scripts/evaluate.py -->
```python
    else:                                                    # rl:<run_name>:<mode>
        _, run, mode = method.split(":")
        policy = policies[run]
        b = batch.to(device)
        d, wi = env_cfg.delivery, env_cfg.weather_input
        with torch.no_grad():
            if mode == "greedy":
                r = rollout(policy, b, d, wi)
                out = {"cost": r.cost.cpu().numpy(), "n_fail": r.n_fail.cpu().numpy()}
            elif mode.startswith("beam"):
                most_probable, _ = beam_search(policy, b, int(mode[4:]), d, wi)
                out = {"cost": most_probable.cpu().numpy(), "n_fail": np.full(b.size, np.nan)}
            elif mode.startswith("best"):
                out = {"cost": best_of_n(policy, b, int(mode[4:]), d, wi).cpu().numpy(),
                       "n_fail": np.full(b.size, np.nan)}
            elif mode == "apriori":
                c, f = apriori(policy, b, d, wi)
                out = {"cost": c, "n_fail": f}
            else:
                raise ValueError(mode)
```

Anything else is an RL method named `rl:<run>:<mode>`. `method.split(":")` unpacks it into three parts
(the first, `"rl"`, is thrown away into `_`), and the run's loaded policy is looked up by name. Each
mode calls one function from lesson 10, all under `torch.no_grad()`:

- `greedy` → `rollout`.
- `beamW` → `beam_search` with width W, parsed from the name: `"beam3"[4:]` is `"3"`. Only the
  most-probable beam's cost is kept.
- `bestN` → `best_of_n` with N samples (`"best16"[4:]` is `"16"`).
- `apriori` → plan on the beliefs, drive with the executor.

Modes that don't track recourse events store NaN ("not available"), never a fake 0. An unknown mode raises
an error instead of silently doing something else.

<!-- quote: scripts/evaluate.py -->
```python
    if device.type == "cuda":
        torch.cuda.synchronize()
    out["sec_total"] = time.perf_counter() - t0
    out["sec_per_instance"] = out["sec_total"] / batch.size
    return out
```

GPU work is *asynchronous*: Python continues before the GPU finishes. `synchronize()` waits, so the
timing is real (the paper's speed comparison, table 10, is about exactly this number).

### Lines 90–118: the command line, and loading trained runs

<!-- quote: scripts/evaluate.py -->
```python
def main():
    p = argparse.ArgumentParser()
    add_args(p, EnvConfig)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--n_days", type=int, default=1000)
    p.add_argument("--methods", default="")
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--rl_modes", default="greedy")
    p.add_argument("--ref", default="ortools_gls")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=None, help="planner processes (default: half the CPU threads)")
    p.add_argument("--world_from_run", action="store_true", help="evaluate in the world the first --runs was trained in")
    a = p.parse_args()
    env_cfg = from_args(a, EnvConfig)
    device = resolve_device(a.device)
    if a.world_from_run:
        env_cfg = load_policy(a.runs[0], "cpu")[1]

    methods = [m for m in a.methods.split(",") if m]
    policies = {}
    for run in a.runs:
        policy, run_env, _, _ = load_policy(run, device)
        if run_env.tag() != env_cfg.tag() or run_env.weather_input != env_cfg.weather_input:
            print(f"note: {Path(run).name} was trained on {run_env.tag()} (wi={run_env.weather_input}); "
                  f"evaluating with its own weather_input on {env_cfg.tag()}")
        env_cfg.weather_input = run_env.weather_input
        policies[Path(run).name] = policy
        methods += [f"rl:{Path(run).name}:{m}" for m in a.rl_modes.split(",")]
```

- **The flags.**
  - Every `EnvConfig` field is a flag (lesson 03), so you evaluate in any world with e.g. `--delivery full`.
  - `--split` val/test and `--n_days` choose the days.
  - `--methods` is a comma-separated list of baselines.
  - `--runs` takes one or more run folders, with `--rl_modes` saying how to run each.
  - `--ref` picks the method everything is compared to (default OR-Tools GLS), and `--force` ignores the cache.
  - `--workers` sets the planner processes.
  - `--world_from_run` takes the world settings from the first run's saved `config.json`. That's handy for
    the Phase 4 runs, each trained in a different world.
- **Loading runs.** Each run is loaded with `load_policy` (lesson 11). If it was trained in a different
  world than the one being evaluated, a note is printed (deliberate generalisation tests are allowed, but
  never by accident). The run's own `weather_input` is used, because a network trained without weather must
  be evaluated without it. Each run adds one method name per requested mode, e.g.
  `rl:my_E2_lr1e-3:greedy`, `rl:my_E2_lr1e-3:beam3`.

### Lines 120–134: caching per-day results

<!-- quote: scripts/evaluate.py -->
```python
    cache = ROOT / "results" / "eval" / env_cfg.tag() / f"{a.split}{a.n_days}"
    cache.mkdir(parents=True, exist_ok=True)
    batch = None
    results = {}
    for m in dict.fromkeys(methods + [a.ref, "hindsight"]):
        f = cache / (m.replace(":", "__") + ".npz")
        if f.exists() and not a.force:
            results[m] = dict(np.load(f))
            continue
        if batch is None:
            batch = World(env_cfg).eval_set(a.split, a.n_days)
        print(f"running {m} ...", flush=True)
        out = run_method(m, batch, env_cfg, device, policies, a.workers)
        np.savez(f, **{k: np.asarray(v) for k, v in out.items()})
        results[m] = out
```

- **Per-day costs are saved**, one `.npz` file per method, in a folder named after the world's tag
  (lesson 03). OR-Tools on 1,000 days takes minutes, so each method is computed once and reused for every
  later comparison. `--force` recomputes.
- The reference and `hindsight` are always included (`dict.fromkeys` removes duplicates and keeps order).
- The days are generated only if something actually needs computing (`batch = None` until then).
- `:` isn't allowed in Windows file names, hence `replace(":", "__")`.

Because these are *per-day* arrays and not averages, any two methods can be paired later, even ones
computed weeks apart. That works only because `eval_set("test")` always produces the same days (lesson 05).

### Lines 136–163: the table and the permanent record

<!-- quote: scripts/evaluate.py -->
```python
    ref, hind = results[a.ref]["cost"], results["hindsight"]["cost"]
    print(f"\nworld {env_cfg.tag()}  split={a.split} ({a.n_days} days)  ref={a.ref}")
    print(f"{'method':48s} {'mean cost':>10s} {'gap vs ref [95% CI]':>24s} {'vs hindsight':>13s} "
          f"{'fails':>6s} {'ms/day':>8s}  deployable")
    rows = []
    for m in dict.fromkeys(methods + [a.ref, "hindsight"]):
        r = results[m]
        g, lo, hi = paired_ci(r["cost"], ref)
        gh = (r["cost"].mean() / hind.mean() - 1) * 100
        fails = np.nanmean(r["n_fail"]) if np.isfinite(r["n_fail"]).any() else float("nan")
        ms = float(r.get("sec_per_instance", np.nan)) * 1000
        print(f"{m:48s} {r['cost'].mean():10.4f} {g:+7.2f}% [{lo:+6.2f},{hi:+6.2f}] {gh:+12.2f}% "
              f"{fails:6.2f} {ms:8.2f}  {label(m)}")
        rows.append([datetime.now().isoformat(timespec="seconds"), env_cfg.tag(), a.split, a.n_days, m,
                     f"{r['cost'].mean():.4f}", f"{g:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{gh:.3f}",
                     f"{fails:.3f}", f"{ms:.3f}", label(m)])
    out_csv = ROOT / "results" / "results.csv"
    new = not out_csv.exists()
    with open(out_csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "world", "split", "n_days", "method", "mean_cost", "gap_ref_pct", "ci_lo", "ci_hi",
                        "gap_hindsight_pct", "fails", "ms_per_day", "deployable"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
```

- The reference's and hindsight's per-day costs are taken out first. Every row is compared to them.
- The header line uses fixed column widths (`:48s` = a 48-character text column, `:>10s` = right-aligned
  10 wide), so the table lines up in a terminal.
- **Per method:**
  - `paired_ci` gives the gap vs the reference with its 95% CI.
  - `gh` is the gap vs hindsight in %.
  - `fails` is the mean recourse events per day, using `nanmean` to ignore NaN. If *all* values are NaN, the
    column shows `nan` instead of crashing.
  - `ms` is milliseconds per day. `r.get("sec_per_instance", np.nan)` is a dict lookup with a default,
    because older cached files might lack the key.
  - The row is printed with its honesty label, and collected for the CSV.
- **`results/results.csv`:** opened in **append** mode (`"a"`). The header is written only if the file is
  new. Every evaluation ever run adds rows with a timestamp, so nothing is overwritten. It's a permanent,
  greppable record.
- The `__main__` guard is required: `run_planner` starts worker processes (lesson 07).

## `scripts/plot.py`: learning curves

(The idea only. **Every line of `plot.py` and `inspect_policy.py` is explained in
[lesson 16](16_helper_scripts.md)**; rebuild them from there.)

The idea, in three parts:

- `read_log(run)` reads `minutes` and `val_cost` from a run's `log.csv`.
- One line per run, plus a dot at the best epoch, labeled with its value at the right end.
- Dashed horizontal lines for baselines (`--hline "Clarke-Wright (val)=8.007"`), with labels staggered
  up/down so that close values don't overlap. (The first version overlapped; see `results/plots/E2_n10_lr.png`.)

<!-- quote: scripts/plot.py -->
```python
    for i, (run, lab) in enumerate(zip(a.runs, labels)):
        x, y = read_log(run)
        ax.plot(x, y, color=SERIES[i], lw=2, label=lab, solid_capstyle="round")
        best = min(range(len(y)), key=y.__getitem__)
        ax.plot(x[best], y[best], "o", ms=6, color=SERIES[i], mec=SURFACE, mew=2)
```

`min(range(len(y)), key=y.__getitem__)` is "the index of the smallest value" in plain Python (an argmin
without NumPy). The colours (`SERIES`, `INK`, …) are defined once at the top. Recessive grey axes and
gridlines keep the eye on the data. `matplotlib.use("Agg")` draws to a file without opening a window.

## `scripts/inspect_policy.py`: what did it learn?

<!-- quote: scripts/inspect_policy.py -->
```python
        a, state = choose(env, state)
        going_home = (a == 0) & (env.pos != 0) & ~env.done
        trips += going_home
        early_home += going_home & (env.load > EPS)
        left_when_home += torch.where(going_home, env.load, 0.0)
```

The same driving loop as `rollout`, but it records **behaviour**: how many trips, how often the truck goes
home with load still on board, and how much. It runs the RL policy *and* the two heuristics on the same
days. The result (`LOG.md`, E2):

| | nn_online | nn_split | RL greedy |
|---|---|---|---|
| cost | 8.03 | 7.87 | **6.06** |
| trips per day | 6.78 | 5.81 | 6.50 |
| cost per trip | 1.18 | 1.35 | **0.93** |
| trips ending with load left | 68% | 17% | 44% |

RL makes *more* trips than `nn_split`, but each is 30% cheaper. It learned which customers belong in the
same trip, and when it's worth going home early. A greedy "nearest next" rule can't express that. **This
table is the real finding**; the −24% is just its consequence.

## The experiment queue: `experiments/queue_1.sh`

(The core idea. **All three shell scripts are explained line by line in [lesson 17](17_experiment_scripts_and_docs.md).**)

<!-- quote: experiments/queue_1.sh -->
```bash
cd "$(dirname "$0")/.."
COMMON="--batch_size 512 --epochs 100000 --lr 1e-3 --critic_lr 1e-3"

run() {
  name=$1; shift
  if [ -f "runs/$name/DONE" ]; then echo "skip $name"; return; fi
  echo "$(date +%H:%M) start $name"
  mkdir -p "runs/$name"
  uv run python -u scripts/train.py $COMMON --run_name "$name" "$@" > "runs/$name/stdout.log" 2>&1
  echo "$(date +%H:%M) end   $name: $(tail -1 runs/$name/stdout.log)"
}
```

- `cd "$(dirname "$0")/.."`: run from the project root, wherever the script is called from.
- `COMMON`: the settings every run shares (`epochs 100000` = "until the time budget stops it").
- `run name flags...`: **skip** the run if it already has `DONE` (so the queue is *resumable*: if your PC
  restarts, run the script again and it continues where it stopped). Otherwise train, sending all
  output to the run's `stdout.log`. `-u` = unbuffered, so the log fills in real time.
- `"$@"` passes the remaining arguments (the run's own flags) through unchanged.

Then the experiments are just one line each, which makes the queue itself a readable list of what was
tested:

<!-- quote: experiments/queue_1.sh -->
```bash
run E4_n10_const_noweather $T --estimate constant --weather_input false
run E4_n10_const_weather   $T --estimate constant --weather_input true
run E4_n10_oracle          $T --estimate oracle_mean
```

**Why a queue that runs one job at a time?** Measured in E3: two training jobs on the GTX 1060 at once
ran at 9.5 s per epoch *each*, against 3.9 s alone. That's slower in total. So jobs run one after another.
`experiments/baselines_E4.sh` computes the classical baselines on the CPU *at the same time* with
`--workers 5`, leaving a core free for the training process.

## How to write a lab-notebook entry (`experiments/LOG.md`)

Every entry in the project's log has the same parts. Use them for yours:

```
## E<number> — <what was tested>
**Hypothesis:** what you expect, and why.
**Setup:** what is different from the reference run (ONE thing), seeds, budget, which days (val/test).
**Result:** a table: method, mean cost, gap with CI, label.
**What this tells us:** 2–4 sentences in plain words. Include surprises and caveats.
**Decision:** kept / rejected / needs more seeds.
```

Read E0–E2 in `experiments/LOG.md` as models: E2 shows a result that looked too good, the checks that were
run, and the explanation.

## Rebuild it yourself

1. `learning/workspace/scripts/evaluate.py`. Start with greedy + planners only, add caching, then the
   bootstrap, then the other RL modes. Keep `main()` behind the `__main__` guard (it starts worker
   processes).
2. `plot.py` (simple first: one line per run, then the extras).
3. `inspect_policy.py`.
4. Start `learning/workspace/LOG.md` with your own E1 (baselines) and E2 (your first training run).

## Verify: reproduce the E2 table

With your trained run from lesson 11 (`runs/my_E2_lr1e-3`), from `learning/workspace`:

```bash
uv run python -m scripts.evaluate --n_customers 10 --methods nn_online,nn_split,cw \
    --runs runs/my_E2_lr1e-3 --rl_modes greedy,apriori,beam3,best16
uv run python -m scripts.inspect_policy --run runs/my_E2_lr1e-3
```

The project's numbers (`LOG.md`, E2, test days):

```
nn_split                     7.8668   -1.91%   yes
cw                           8.0010   -0.23%   yes
rl:...:greedy                6.0603  -24.43%   yes
rl:...:apriori               7.0615  -11.95%   yes
rl:...:beam3                 6.0443  -24.63%   lookahead
rl:...:best16                5.6122  -30.02%   hindsight
ortools_gls                  8.0198   +0.00%   yes
hindsight                    6.0364  -24.73%   reference
```

Your RL rows will differ a little (a different training run). The *pattern* should hold:

- RL greedy is far below OR-Tools, with a CI that doesn't touch 0;
- a-priori RL is in between;
- beam3 ≈ greedy;
- best16 is lower still, but it's hindsight.

**Two questions to answer in your own log** (the project's answers are in `LOG.md`):

1. Why can RL *greedy* match "hindsight" here? (Hint: what does hindsight's OR-Tools *not* allow?)
2. Why is a-priori RL so much better than OR-Tools' a-priori plan, when both plan from the same estimates?

## Common mistakes

- **Comparing averages computed on different days**: always use the same `eval_set`.
- **Tuning on test**: use `--split val` for decisions.
- **Reporting best-of-n or beam as "the method's result"** without the label.
- **No `synchronize()` before timing the GPU**: impossibly fast ms/day.
- **Only logging successes**: the rejected ideas (lr 1e-4, the overlapping plot, two GPU jobs at once) are
  as informative as the kept ones.

## Check your understanding

1. In the 5-day toy table, would the bootstrap CI include 0? Why?
2. Why does `evaluate.py` save per-day arrays instead of just the mean?
3. `nn_split` was added *after* seeing RL's result. Is that cheating? What is it for?
4. Your new idea improves validation cost by 0.4% with one seed. What do you do before calling it an
   improvement?

**Answers** (try the questions first, then check here)

1. No. All five differences are negative, so every resample has a negative mean difference. The CI lies
   entirely below 0.
2. So that any two methods can be compared *paired*, per day, later, and the bootstrap can resample days.
   A mean alone allows neither.
3. Not cheating. It's a **control**: it tests a specific boring explanation for the result. Adding controls
   after a surprising result is good practice. Changing the *test* until the result looks good would be
   cheating.
4. Run more seeds (3+) for both the new and the reference config, compare the mean difference with 2
   standard errors (rule S10), and only then check test.

## Read more

- Efron & Tibshirani, *An Introduction to the Bootstrap* (1993), chapters 1–2. Or the Wikipedia page
  "Bootstrapping (statistics)".
- Henderson et al. (2018), "Deep Reinforcement Learning that Matters" (arXiv 1709.06560): why RL results
  need several seeds and honest reporting.
- The paper's Part 6 (`part 6.md`), now that you can see what each of its tables would take to reproduce.

**Next:** [13 — Improvements](13_improvements.md)
