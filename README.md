# svrp-rl

Reinforcement learning for the **Stochastic Vehicle Routing Problem**: a from-scratch reproduction of
Iklassov et al., *"Reinforcement Learning for Solving Stochastic Vehicle Routing Problem"* (ACML 2023),
plus experiments that try to improve it. Built to run on a GTX 1060 3GB.

- `SPEC.md` — what exactly is built and why (environment, baselines, models, evaluation rules).
- `experiments/LOG.md` — the lab notebook: every experiment, its result, and whether we kept it.
- `a.md`, `b.md`, `part 5.md`, `part 6.md` — study notes on the paper.

## Setup

```bash
uv sync                                   # installs torch 2.14 (CUDA 12.6 build: supports Pascal / sm_61), OR-Tools, ...
uv run python scripts/check_gpu.py        # GPU sanity check + speed benchmark
uv run pytest -q                          # tests: generator, env, executor, baselines, models
```

## Typical workflow

```bash
# 1. classical baselines on the 1000 test days (cached in results/eval/)
uv run python scripts/evaluate.py --n_customers 20 --methods nn_online,cw,ortools_gls,ortools_tabu,aco,hindsight

# 2. train the paper's model (stops after --time_limit_min minutes; best checkpoint in runs/<name>/best.pt)
uv run python scripts/train.py --n_customers 20 --batch_size 512 --epochs 1000 --time_limit_min 60

# 3. score the trained model on the same days, next to the baselines
uv run python scripts/evaluate.py --n_customers 20 --runs runs/<name> --rl_modes greedy,beam3,apriori,best16

# 4. learning curves
uv run python scripts/plot.py --runs runs/<name> --hline "OR-Tools GLS=..." --out my_plot
```

Every field of `EnvConfig` / `TrainConfig` in `svrp/config.py` is a command-line flag, e.g.
`--delivery full`, `--estimate constant`, `--weather_input false`, `--positions flexible`,
`--b_ratio 0.4`, `--fill_rate 0.2`, `--model graph`, `--baseline shared`.

## Layout

```
svrp/env/generator.py     the stochastic world: θ = θ̄ + wᵀαw + ε with the A/B/Γ signal ratio
svrp/env/estimates.py     k-NN estimates from a history bank of past days
svrp/data.py              Batch (truth + beliefs) and World (fixed seeds per split)
svrp/env/svrp_env.py      batched MDP: masking, partial/full delivery, recourse
svrp/env/executor.py      drive an a-priori plan on the real day with detour-to-depot recourse
svrp/baselines/           nearest-neighbour, Clarke-Wright, OR-Tools (GLS / Tabu), ACO, hindsight
svrp/models/              PaperNet (Conv1d + LSTM + attention), GraphNet (Transformer), Critic
svrp/train/reinforce.py   REINFORCE with critic or shared baseline
svrp/inference.py         greedy, sampling, beam search, a-priori planning
scripts/                  check_gpu, train, evaluate, tune_aco, plot
```
