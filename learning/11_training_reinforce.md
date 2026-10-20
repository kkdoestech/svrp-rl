# 11 — Training with REINFORCE (`svrp/train/reinforce.py`, `scripts/train.py`)

| | |
|---|---|
| **Layer** | 9 |
| **Files covered** | `svrp/train/reinforce.py` (145 lines), `scripts/train.py` (18), `tests/test_models.py` |
| **You build** | `mysvrp/train/reinforce.py`, `learning/workspace/scripts/train.py` |
| **Time** | 4–6 hours + 15 minutes of GPU time for the milestone |
| **Before this** | [10 — Rollouts](10_rollout_and_inference.md); **re-read `part 5.md` part 3 (Training)**, slowly |

## Why this layer exists

The network from lesson 09 is random: 9.50 at N = 10, worse than Clarke–Wright's 8.0. Training turns its
170,000 knobs so that cheap routes become likely. The paper uses **REINFORCE with a baseline**, the
formula in your notes:

```
∇_Θ J  ≈  mean over days  [ (C − b_φ) · ∇_Θ Σ_t log π_Θ(a_t | s_t) ]
```

This lesson turns that line into ~40 lines of PyTorch. At the end, **your own RL beats Clarke–Wright**
after 15 minutes of training on your GTX 1060.

## Concepts from zero

### Gradient descent: turning knobs downhill

Imagine a number L (the *loss*) that depends on all the knobs Θ. The **gradient** ∇L says, for every knob,
"turning this knob up by a tiny bit changes L by this much". Turning every knob a little *against* its
gradient lowers L:

```
Θ  ←  Θ − lr · ∇L          (lr = learning rate: how big each step is)
```

PyTorch does the calculus for you (*autograd*):

1. compute `loss` from tensors that came out of the network;
2. `loss.backward()` fills `p.grad` for every parameter p;
3. `optimizer.step()` updates the parameters;
4. `optimizer.zero_grad()` clears the old gradients (PyTorch *adds* new gradients to old ones otherwise).

**Adam** is a smarter "step" that adapts the step size per knob. It's the default choice almost everywhere.

### The REINFORCE trick: you can't differentiate a road

Our true goal is *small expected cost*. But the cost comes out of the environment (roads, demands,
`if`s). It isn't a differentiable function of the knobs, so `cost.backward()` is impossible. REINFORCE
(Williams, 1992) works around it with a fact from probability:

```
∇ E[C]  =  E[ C · ∇ log π(route) ]
```

In words: *sample routes; for each one, push its log-probability up or down, in proportion to its
cost.* Since we **minimise** cost, we do gradient descent on the surrogate loss

```
loss = mean[ (C − b) · logp_sum ]
```

The sign logic, from your notes' "surprise factor":

- C > b (worse than expected): (C − b) > 0. Descent lowers `logp_sum`, so **that route becomes less likely**.
- C < b (better than expected): (C − b) < 0. Descent raises `logp_sum`, so **that route becomes more likely**.

### Why the baseline b matters (the "class average")

Without b, every cost is positive (~8), so *every* sampled route would be pushed down, just some more
than others. Learning still works on average, but it's very noisy. Subtracting a baseline b ≈ "the
expected cost of this day" makes good routes positive news and bad routes negative news. Your notes'
exam analogy: 90% means little until you know the class average.

Worked example. On one day the baseline predicts b = 8.0. Two sampled routes cost 7.0 and 9.5, with log
probabilities −20 and −22:

| route | C − b | effect of descent on (C − b)·logp |
|---|---|---|
| 1 | −1.0 | raise log p of route 1 (cheaper than expected) |
| 2 | +1.5 | lower log p of route 2 (dearer than expected) |

Two baselines are implemented:

- **critic** (the paper): a second network b_φ(observation) that learns to predict C by minimising
  (b_φ − C)². It's trained alongside the policy.
- **shared** (improvement I1, from POMO, Kwon et al. 2020): sample **M routes on the same day** and use
  their *average cost* as b. No second network is needed, and the baseline is exactly "the average on
  this day". Lesson 13 tests whether it helps.

### Epochs, validation, checkpoints, budget

- **Step** = one batch of 512 sampled days → one update.
- **Epoch** = 10,240 days (20 steps at batch 512), roughly the paper's "10,000 training scenarios". Fresh
  days are generated for every step, so there's no fixed training set to memorise.
- After every epoch: **validate** (greedy on the 1,000 validation days). If it's the best so far, save the
  weights as `best.pt`. Training is noisy, so the last epoch isn't always the best.
- Stop after `time_limit_min` minutes (the "≤ 1 hour per run" rule).

### Gradient clipping

Occasionally one batch produces a huge gradient (a freak day), and one giant step can wreck everything
learned so far. `clip_grad_norm_(params, 1.0)` rescales the whole gradient whenever its length exceeds 1.0.

## Code walkthrough

### Lines 1–28: the formula, and where runs go

<!-- quote: svrp/train/reinforce.py -->
```python
"""REINFORCE training (part 5.md).

    loss_actor  = mean[ (C − b) · Σ_t log π(a_t | s_t) ]      C = realized cost of the sampled route
    loss_critic = mean[ (b_φ − C)² ]                           (only for baseline="critic")

Minimizing loss_actor makes routes that were cheaper than expected (C < b) more likely, and routes that
were more expensive than expected less likely. The baseline b is the "class average" from the notes:
    critic   b = b_φ(initial observation), a separate network           (the paper)
    shared   run M sampled routes on the SAME day, b = their mean cost   (POMO-style, improvement I1)
"""
```

then the imports:

<!-- quote: svrp/train/reinforce.py -->
```python
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from svrp.config import EnvConfig, TrainConfig, to_dict
from svrp.data import World, train_generator
from svrp.env.svrp_env import SVRPEnv
from svrp.inference import rollout
from svrp.models.paper_net import Critic, PaperNet
```

`csv` and `json` write the run's log and configs, `time` measures the budget, and `Path` builds folder paths.
From the project it needs config, data, the env (for the critic's morning features), `rollout`, and the
two networks. This is the top of the layer map: it uses almost everything below it.

And line 28:

<!-- quote: svrp/train/reinforce.py -->
```python
RUNS = Path(__file__).resolve().parents[2] / "runs"
```

`parents[2]` of `svrp/train/reinforce.py` is the project root, so runs are saved in `runs/<name>/`. For
your `mysvrp/train/reinforce.py` inside `learning/workspace`, the same line gives `learning/workspace/runs/`,
your own runs folder.

### Lines 31–56: helpers

<!-- quote: svrp/train/reinforce.py -->
```python
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_policy(tr: TrainConfig, env: EnvConfig) -> nn.Module:
    if tr.model == "paper":
        return PaperNet(env.n_customers + 1, env.n_external, tr.embed_dim)
    if tr.model == "graph":
        from svrp.models.graph_net import GraphNet
        return GraphNet(env.n_external, tr.embed_dim)
    raise ValueError(tr.model)
```

- `"auto"`: use the GPU if there is one.
- `make_policy`: config → network. GraphNet is imported *inside* the branch, so this file works before
  `graph_net.py` exists (that's lesson 13; leave the branch out for now).

<!-- quote: svrp/train/reinforce.py -->
```python
def default_run_name(env: EnvConfig, tr: TrainConfig) -> str:
    return f"{tr.model}_{tr.baseline}_{env.tag()}_seed{tr.seed}"
```

If you don't pass `--run_name`, the folder name is built from the model, the baseline, the world's tag and
the seed, e.g. `paper_critic_n10_phi0.5_…_s0_seed1`. Long, but it says exactly what was trained.

<!-- quote: svrp/train/reinforce.py -->
```python
@torch.no_grad()
def validate(policy, batch, env: EnvConfig, chunk: int = 1000) -> float:
    policy.eval()
    costs = [rollout(policy, batch.index(slice(i, i + chunk)), env.delivery, env.weather_input).cost
             for i in range(0, batch.size, chunk)]
    policy.train()
    return torch.cat(costs).mean().item()
```

Greedy rollouts on the validation days, without gradients, in chunks of 1,000. `policy.eval()` /
`policy.train()` switch modes (only some layers, such as dropout, care; it's good habit to switch anyway).

### Lines 59–80: setting up a run

<!-- quote: svrp/train/reinforce.py -->
```python
def train(env_cfg: EnvConfig, tr: TrainConfig, log=print) -> Path:
    device = resolve_device(tr.device)
    torch.manual_seed(tr.seed)
    run_dir = RUNS / (tr.run_name or default_run_name(env_cfg, tr))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"env": to_dict(env_cfg), "train": to_dict(tr)}, indent=2))

    world = World(env_cfg)
    val = world.eval_set("val", tr.val_size).to(device)
    gen = train_generator(tr.seed)
```

- `log=print`: the function used for progress messages. It's `print` by default; the tests pass
  `log=lambda *_: None` to keep their output quiet. (Passing a function as an argument is a handy trick.)
- `torch.manual_seed(tr.seed)` fixes the network's initial knobs and the sampling, so seed 1 and seed 2
  are two different, repeatable "students".
- The run folder gets **both configs as JSON first**. Even a crashed run documents its settings.
- The world, the fixed validation days, and the training stream (lesson 05).

<!-- quote: svrp/train/reinforce.py -->
```python
    policy = make_policy(tr, env_cfg).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=tr.lr)
    critic = critic_opt = None
    if tr.baseline == "critic":
        critic = Critic(env_cfg.n_customers + 1, env_cfg.n_external, tr.embed_dim).to(device)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=tr.critic_lr)
    elif tr.baseline != "shared":
        raise ValueError(tr.baseline)
    M = tr.n_shared if tr.baseline == "shared" else 1
    if tr.batch_size % M:
        raise ValueError("batch_size must be divisible by n_shared")
```

- Two separate optimisers: one for the policy, one for the critic. The critic's loss must never move the
  policy's knobs.
- M = how many rollouts share one day: 8 for the shared baseline, 1 otherwise. A batch of 512 with M = 8
  means 64 different days × 8 tries each.

### Lines 82–85: the starting point

<!-- quote: svrp/train/reinforce.py -->
```python
    best = validate(policy, val, env_cfg)
    log(f"[{run_dir.name}] device={device}  params={sum(p.numel() for p in policy.parameters()):,}  "
        f"untrained val cost={best:.4f}")
    torch.save({"policy": policy.state_dict(), "epoch": 0, "val": best}, run_dir / "best.pt")
```

Validate the untrained network and save it as the first "best". So `best.pt` always exists, and the log
shows where learning started. `state_dict()` is a dict of all parameter tensors, which is what gets saved.
`{:,}` prints 167,680 with thousands separators.

### Lines 87–99: the training loop, sample days and drive them

<!-- quote: svrp/train/reinforce.py -->
```python
    steps = math.ceil(tr.epoch_size / tr.batch_size)
    t0 = time.time()
    with open(run_dir / "log.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "minutes", "train_cost", "val_cost", "loss", "critic_mse", "grad_norm"])
        for epoch in range(1, tr.epochs + 1):
            sums = torch.zeros(4, device=device)
            for _ in range(steps):
                batch = world.sample(tr.batch_size // M, gen)
                if M > 1:
                    batch = batch.repeat_interleave(M)
                batch = batch.to(device)
                r = rollout(policy, batch, env_cfg.delivery, env_cfg.weather_input, decode="sample")
```

- `log.csv` gets one row per epoch. `scripts/plot.py` (lesson 12) draws the learning curves from it.
- `sums` accumulates 4 statistics on the GPU, which avoids slow GPU→CPU transfers every step.
- Each step: **fresh days** from the training stream (64 days × 8 copies for shared, or 512 days), moved to
  the GPU, then one **sampled** rollout with gradients on. `r.logp` is connected to every knob that
  influenced every decision.

### Lines 100–112: the baseline

<!-- quote: svrp/train/reinforce.py -->
```python
                if critic is not None:
                    feats = SVRPEnv(batch, env_cfg.delivery, env_cfg.weather_input).node_features()
                    b = critic(feats)
                    critic_loss = (b - r.cost).pow(2).mean()
                    critic_opt.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), tr.grad_clip)
                    critic_opt.step()
                    adv = r.cost - b.detach()
                else:
                    c = r.cost.view(-1, M)
                    adv = (c - c.mean(1, keepdim=True)).view(-1)
                    critic_loss = torch.zeros((), device=device)
```

**Critic path:**

- `feats`: the *morning* observation (a fresh env at the start of the day). The critic guesses the cost
  from what is known before driving.
- `critic_loss` = mean squared error between guess and actual cost: `part 5.md`'s ‖b_φ − C‖². One Adam
  step on the critic.
- `adv = C − b.detach()` is the **advantage** ("surprise"). In REINFORCE the baseline must act as a
  **constant**: the math only stays unbiased if no gradient flows through b. `.detach()` says so
  explicitly. Here it changes no numbers, because the critic is a separate network whose input doesn't
  depend on the policy. But it skips computing useless gradients into the critic. And it protects you in
  designs where the baseline *shares layers* with the policy (a common design). There, forgetting it
  really would let the policy's loss reshape the baseline.
- Notice that `r.cost` carries no gradient (it comes from the env's `if`s and arithmetic on non-learned
  tensors), so `critic_loss.backward()` only reaches the critic.

**Shared path:**

- `view(-1, M)` regroups the costs: each row is one day's M tries. Each try's advantage is its cost minus
  that day's mean. The M advantages of a day sum to zero: half the tries get pushed up, half down.

### Lines 113–118: the REINFORCE update

<!-- quote: svrp/train/reinforce.py -->
```python
                loss = (adv * r.logp).mean()
                opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(policy.parameters(), tr.grad_clip)
                opt.step()
                sums += torch.stack([r.cost.mean().detach(), loss.detach(), critic_loss.detach(), gn.detach()])
```

**The paper's formula is the first line.** `loss.backward()` computes ∇ of mean[(C − b)·Σ log π] through
every step of every rollout. That is the "steering wheel" ∇_Θ log π_Θ from your notes. Then clip and step.

`clip_grad_norm_` returns the gradient's length *before* clipping (`gn`). It's logged, because a gradient
norm that explodes is an early warning sign.

The loss value itself is **not** a meaningful score (it can be negative, and it jumps around). Watch
`train_cost` and `val_cost` instead.

### Lines 119–135: end of epoch, validate, save, log, budget

<!-- quote: svrp/train/reinforce.py -->
```python
            sums = (sums / steps).tolist()
            v = validate(policy, val, env_cfg)
            minutes = (time.time() - t0) / 60
            w.writerow([epoch, f"{minutes:.2f}", f"{sums[0]:.4f}", f"{v:.4f}", f"{sums[1]:.4f}",
                        f"{sums[2]:.4f}", f"{sums[3]:.3f}"])
            fh.flush()
            flag = ""
            if v < best:
                best, flag = v, " *"
                torch.save({"policy": policy.state_dict(), "epoch": epoch, "val": v}, run_dir / "best.pt")
            log(f"epoch {epoch:3d}  {minutes:6.1f} min  train {sums[0]:.4f}  val {v:.4f}{flag}")
            if minutes > tr.time_limit_min:
                log(f"time budget of {tr.time_limit_min} min reached")
                break
    log(f"best val {best:.4f}  ->  {run_dir}")
    (run_dir / "DONE").write_text(f"{best:.6f}\n")
    return run_dir
```

- Average the epoch's statistics, validate greedily, and write a CSV row. `fh.flush()` forces it to disk
  now, so you can watch `log.csv` grow while training runs.
- A new best is saved and marked with `*` in the log.
- Time budget reached: stop.
- Write a `DONE` marker. The experiment queue scripts (lesson 12) use it to skip finished runs.

### Lines 138–145: loading a trained policy

<!-- quote: svrp/train/reinforce.py -->
```python
def load_policy(run_dir: Path, device="cpu"):
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    env_cfg, tr = EnvConfig(**cfg["env"]), TrainConfig(**cfg["train"])
    policy = make_policy(tr, env_cfg).to(device)
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, env_cfg, tr, ckpt
```

Rebuild the exact configs from JSON, create the same network, and load the saved knobs. `map_location`
lets a GPU-trained model load on a CPU-only PC. Old runs that predate a new config field still load,
because the missing field takes its default (that's why new fields always get "off" defaults).

### `scripts/train.py`: the command-line entry point

<!-- quote: scripts/train.py -->
```python
"""Train a policy.  Examples:

    uv run python scripts/train.py --n_customers 10 --epochs 1                      # smoke test
    uv run python scripts/train.py --n_customers 20 --batch_size 256 --time_limit_min 60
    uv run python scripts/train.py --n_customers 20 --baseline shared --n_shared 8

Every EnvConfig / TrainConfig field is a flag (see svrp/config.py).
"""
import argparse

from svrp.config import EnvConfig, TrainConfig, add_args, from_args
from svrp.train.reinforce import train

p = argparse.ArgumentParser()
add_args(p, EnvConfig)
add_args(p, TrainConfig)
a = p.parse_args()
train(from_args(a, EnvConfig), from_args(a, TrainConfig))
```

Lesson 03's auto-flags at work: every config field is a `--flag`. (No `__main__` guard is needed: no
worker processes are started here.)

## Rebuild it yourself

1. `resolve_device`, `make_policy` (paper only), `default_run_name`, `validate`.
2. `train` with the **critic** path only. Run a smoke test (below).
3. Add the shared path.
4. `load_policy`. Then `learning/workspace/scripts/train.py`, importing from `mysvrp`.

## Verify

**Tests.** Copy `tests/test_models.py`. Its GraphNet test needs lesson 13, so skip it for now:

```bash
uv run python -m pytest tests/test_models.py -q -k "not graphnet"
```

The tests check that sampled rollouts have gradients, beam1 = greedy, beam/best-of-n/a-priori are valid,
and that `train()` runs with both baselines, saves `best.pt`, writes 3 lines of `log.csv` (header + 2
epochs), and reloads the same configs.

**Smoke run** (seconds). Before anything long, check that learning happens at all:

```bash
uv run python -m scripts.train --n_customers 10 --epochs 3 --batch_size 512 --run_name smoke_n10
```

You should see the validation cost drop. The project's own smoke run printed:

```
[smoke_n10] device=cuda  params=167,680  untrained val cost=10.2995
epoch   1     0.0 min  train 10.1226  val 9.6486 *
epoch   2     0.1 min  train 9.8426  val 8.9372 *
epoch   3     0.1 min  train 9.3058  val 8.6856 *
```

(Exact numbers on your run can differ slightly: GPU arithmetic isn't bit-for-bit repeatable.)

**Milestone: beat Clarke–Wright** (15 minutes of GPU). Make sure nothing else is training on the GPU
(Task Manager → Performance → GPU). Two jobs at once are each much slower (`LOG.md` E3 found that out
the hard way).

```bash
uv run python -m scripts.train --n_customers 10 --batch_size 512 --epochs 100000 \
    --time_limit_min 15 --lr 1e-3 --critic_lr 1e-3 --run_name my_E2_lr1e-3
```

Then plot it. You'll write your own plot script in lesson 12; for now use the project's original one. It
reads your `runs/` folder and writes the picture to the project's `results/plots/my_E2.png`:

```bash
uv run python ../../scripts/plot.py --runs runs/my_E2_lr1e-3 --hline "Clarke-Wright (val)=8.007" --out my_E2
```

The project's run reached **val 6.069** (vs Clarke–Wright 8.007 on the same validation days), with a
sharp drop around minute 2. Then try `--lr 1e-4`: it sits on a plateau near 8.25 for ~8 minutes and ends
at 7.6. That's the learning-rate experiment E2 in `experiments/LOG.md`, and the reason every later run
uses 1e-3. (`results/plots/E2_n10_lr.png` shows both curves.)

## Common mistakes

- **No `.detach()` on the baseline**: harmless in this exact code (separate critic, and its gradients are
  zeroed before its own step), but a real bug as soon as the baseline shares layers with the policy.
  Always detach it.
- **Forgetting `zero_grad()`**: gradients pile up from step to step, giving effectively huge random steps.
- **Wrong sign** (`loss = -(adv * logp).mean()`): the AI learns to make routes *more expensive*. Training
  cost climbs. That symptom is unmistakable.
- **Greedy decoding in training**: no exploration, so the loss is ~0 and nothing improves.
- **Judging by the loss value**: watch `val_cost`.
- **Two GPU jobs at once**: each runs at less than half speed, which silently breaks a time budget.

## Check your understanding

1. In the worked example, why does route 2 get pushed down even though its log-probability (−22) was
   already lower than route 1's?
2. What exactly would go wrong without `b.detach()`?
3. With the shared baseline and M = 8, what is the sum of the 8 advantages of one day? Why is that nice?
4. Why save `best.pt` by *validation* cost instead of keeping the last epoch?
5. `lr = 1e-3` learned far faster than `1e-4`. Why not use `1e-1`?

**Answers** (try the questions first, then check here)

1. The push depends on (C − b), not on log p. Route 2 cost more than expected, so *whatever* its
   probability, it's made less likely.
2. In *this* code, nothing visible. The policy's gradient doesn't depend on b's gradient, because b comes
   from a separate network whose input the policy doesn't touch. The stray gradients that `loss.backward()`
   puts into the critic are wiped by `critic_opt.zero_grad()` before the critic's next update. Only work is
   wasted. But if the critic were a "head" on the policy's own layers, the policy loss would push those
   shared layers to change b, so the "surprise" would no longer be an honest one. Detaching keeps the
   two learning problems separate in every design.
3. Zero (each is cost − mean of the 8). So the update only redistributes probability *among* that day's
   tries: better-than-average up, worse-than-average down.
4. Training is noisy, and the last epoch can be a dip. The validation days measure what we care about,
   and choosing on validation (not test) keeps the test score honest.
5. Steps that big jump past good settings and make training unstable. The right learning rate is an
   experiment (E2), not a guess.

## Read more

- Andrej Karpathy, "Deep Reinforcement Learning: Pong from Pixels": [karpathy.github.io/2016/05/31/rl](https://karpathy.github.io/2016/05/31/rl/).
  The friendliest explanation of policy gradients there is.
- Sutton & Barto, chapter 13 ("Policy Gradient Methods"), sections 13.1–13.4 (REINFORCE with baseline).
- Lilian Weng, "Policy Gradient Algorithms": [lilianweng.github.io/posts/2018-04-08-policy-gradient](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- Williams (1992), "Simple statistical gradient-following algorithms for connectionist reinforcement
  learning", *Machine Learning* 8. The original REINFORCE paper.
- Kwon et al. (2020), "POMO" (arXiv 2010.16011): the shared baseline idea.
- PyTorch tutorial, *Optimizing Model Parameters*: [pytorch.org/tutorials/beginner/basics/optimization_tutorial.html](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

**Next:** [12 — Evaluation and experiments](12_evaluation_and_experiments.md)
