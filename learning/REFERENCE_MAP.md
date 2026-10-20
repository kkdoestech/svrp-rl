# Reference map: every file in the project

Use this page to find where something lives, what a file needs, and which lesson explains it. Line counts
are for the current version.

## The package `svrp/` (what you rebuild as `mysvrp/`)

| file | lines | what it does | key names | needs | used by | lesson |
|---|---|---|---|---|---|---|
| `svrp/__init__.py` (and in `env/`, `baselines/`, `models/`, `train/`) | 0 | makes each folder an importable package | — | — | — | [01](01_setup_and_tools.md) |
| `svrp/config.py` | 105 | every setting of an experiment; auto CLI flags | `EnvConfig`, `TrainConfig`, `capacity`, `tag()`, `add_args`, `from_args` | — | everything | [03](03_config.md) |
| `svrp/env/generator.py` | 174 | the random world: θ = θ̄ + wᵀαw + ε, signal ratio, clipping, events | `weather_moments`, `City`, `make_cities`, `sample_weather`, `realize`, `clip_demand`, `clip_cost`, `sample_events` | config | data | [04](04_generator.md), [13](13_improvements.md) |
| `svrp/env/estimates.py` | 36 | k-NN estimate from a history of past days | `knn_estimate` | — | data | [05](05_estimates_and_data.md) |
| `svrp/data.py` | 143 | truth + belief in one `Batch`; fixed city, history bank, seeds per split | `Batch`, `World`, `eval_set`, `train_generator`, `_gen`, `EVENT_SCALE` | config, generator, estimates | env, baselines, train, scripts | [05](05_estimates_and_data.md) |
| `svrp/baselines/common.py` | 44 | plan ↔ routes, the planned cost, clipping orders to Q | `routes_to_plan`, `plan_to_routes`, `plan_cost`, `planning_demand` | numpy | CW, OR-Tools, ACO, tests | [06](06_plans_and_classical_solvers.md) |
| `svrp/baselines/clarke_wright.py` | 44 | savings heuristic | `clarke_wright` | common | runner | [06](06_plans_and_classical_solvers.md) |
| `svrp/baselines/ortools_solver.py` | 65 | OR-Tools CVRP (GLS / Tabu) on estimates | `ortools_cvrp`, `default_time_limit` | common, ortools | runner | [06](06_plans_and_classical_solvers.md) |
| `svrp/env/executor.py` | 66 | drive an a-priori plan on the real day with detour-to-depot recourse | `execute`, `execute_many` | numpy | runner, inference (`apriori`) | [07](07_executor_and_runner.md) |
| `svrp/baselines/runner.py` | 70 | plan from beliefs, score on truth, many days in parallel | `run_planner`, `solve`, `PLANNERS` | CW, OR-Tools, ACO, data, executor | evaluate, tune_aco | [07](07_executor_and_runner.md) |
| `svrp/env/svrp_env.py` | 116 | the batched MDP: observation, mask, step, recourse, `select` | `SVRPEnv`, `observed_demand`, `node_features`, `mask`, `step`, `max_steps`, `EPS` | data (`Batch`) | nn_online, models, inference, train | [08](08_environment.md) |
| `svrp/baselines/nn_online.py` | 33 | nearest-feasible-neighbour online rule (+ the `nn_split` control) | `nn_online` | data, env | evaluate | [08](08_environment.md) |
| `svrp/models/paper_net.py` | 53 | the paper's network and the critic | `PaperNet`, `Critic` | env | train | [09](09_policy_network.md) |
| `svrp/inference.py` | 117 | rollouts (greedy/sample), beam search, best-of-n, a-priori | `rollout`, `Rollout`, `masked_log_probs`, `beam_search`, `best_of_n`, `apriori`, `index_state` | data, env, executor | train, evaluate, scripts | [10](10_rollout_and_inference.md) |
| `svrp/train/reinforce.py` | 145 | REINFORCE with critic or shared baseline; validation, checkpoints | `train`, `validate`, `make_policy`, `load_policy`, `resolve_device`, `RUNS` | config, data, env, inference, models | train.py, evaluate, inspect, check_gpu | [11](11_training_reinforce.md) |
| `svrp/models/graph_net.py` | 105 | Transformer policy with cost-biased attention, any N | `GraphNet`, `MHA`, `EncoderLayer`, `SOFTPLUS_ONE` | env | train (`--model graph`) | [13](13_improvements.md) |
| `svrp/baselines/aco.py` | 101 | Ant System + elitist ant + 2-opt | `aco`, `two_opt`, `PARAM_GRID` | common | runner, tune_aco | [14](14_aco_optional.md) |

## Scripts `scripts/`

| file | lines | what it does | lesson |
|---|---|---|---|
| `scripts/check_gpu.py` | 58 | GPU sanity check + training-step benchmark | [01](01_setup_and_tools.md) (check), [16](16_helper_scripts.md) (benchmark) |
| `scripts/train.py` | 18 | train a policy from command-line flags | [11](11_training_reinforce.md) |
| `scripts/evaluate.py` | 163 | score methods on the same days; paired bootstrap; cache; results.csv | [12](12_evaluation_and_experiments.md) |
| `scripts/plot.py` | 72 | learning curves with baseline lines | [16](16_helper_scripts.md) (idea in [12](12_evaluation_and_experiments.md)) |
| `scripts/inspect_policy.py` | 95 | behaviour statistics: trips, splits, going home early | [16](16_helper_scripts.md) (idea in [12](12_evaluation_and_experiments.md)) |
| `scripts/tune_aco.py` | 39 | pick ACO settings on validation days | [14](14_aco_optional.md) |

## Tests `tests/` (copy them, replacing `svrp` → `mysvrp`)

| file | lines | checks | lesson |
|---|---|---|---|
| `tests/test_generator.py` | 110 | moment formula, signal ratio, symmetry, clipping, reproducibility, truth independent of estimator, k-NN quality, events | [15](15_tests.md) (written alongside 04 / 05 / 13) |
| `tests/test_env.py` | 70 | random policies finish, cost accounting, reveal-on-arrival, split rule, `select` | [15](15_tests.md) (alongside 08) |
| `tests/test_baselines.py` | 84 | plans valid, executor = OR-Tools objective = env replay, recourse math, runner | [15](15_tests.md) (alongside 07 / 08 / 14) |
| `tests/test_models.py` | 70 | gradients flow, beam1 = greedy, a-priori runs, GraphNet any-N, training loop saves/loads | [15](15_tests.md) (alongside 10 / 11 / 13) |

## Experiments `experiments/`

| file | what it is | lesson |
|---|---|---|
| `experiments/LOG.md` | the lab notebook (E0 setup, E1 baselines, E2 reproduction, …) | [12](12_evaluation_and_experiments.md), [17](17_experiment_scripts_and_docs.md) |
| `experiments/queue_1.sh` | GPU queue: N=20 headline, N=10 seeds, shared baseline, Phase 4 findings | [17](17_experiment_scripts_and_docs.md) |
| `experiments/queue_2.sh` | GPU queue: GraphNet and event experiments | [17](17_experiment_scripts_and_docs.md) |
| `experiments/baselines_E4.sh` | CPU: classical baselines for every Phase 4 world | [17](17_experiment_scripts_and_docs.md) |

## Project root

| file | what it is | lesson |
|---|---|---|
| `pyproject.toml` | dependencies (torch cu126 pin for the GTX 1060), package, pytest config | [01](01_setup_and_tools.md) |
| `uv.lock` | exact installed versions (generated by `uv`, never edited by hand) | [01](01_setup_and_tools.md) |
| `.gitignore` | files git ignores (`.venv`, `runs/`, `*.pt`, …) | [01](01_setup_and_tools.md) |
| `SPEC.md` | the project's rules and its deviations from the paper | [17](17_experiment_scripts_and_docs.md) (structure); read alongside 04–12 |
| `README.md` | how to run the original project | [17](17_experiment_scripts_and_docs.md) |
| `results/` | `eval/` per-day costs, `plots/`, `results.csv`, `aco_params.json` | [12](12_evaluation_and_experiments.md) |
| `runs/` (ignored by git) | one folder per training run: `config.json`, `log.csv`, `best.pt`, `stdout.log`, `DONE` | [11](11_training_reinforce.md) |

## Who imports whom

```
config ──► generator ──┐
   │       estimates ──┼──► data ──► svrp_env ──► nn_online
   │                   │      │         │
   │                   │      │         ├──► paper_net ──┐
   │                   │      │         ├──► graph_net ──┤
   │                   │      │         └──► inference ──┼──► reinforce ──► scripts/train.py
   │                   │      │                ▲         │
common ──► clarke_wright, ortools_solver, aco  │         │
             │                                 │         │
             └──► runner ◄── executor ─────────┘         │
                    │                                    │
                    └────────────► scripts/evaluate.py ◄─┘
```

## Build-order checklist

Tick these off in your own copy of this file as you go.

- [ ] 01 workspace created, `check_gpu` passes
- [ ] 03 `config.py`: the tag check prints `n10_phi0.5_r0.6-0.2-0.2_d1c1_fixed_partial_knn_k10_s0`
- [ ] 04 `env/generator.py`: shares 0.6 / 0.2 / 0.2
- [ ] 05 `env/estimates.py`, `data.py`: `test_generator.py -k "not event"` passes (9), fingerprint matches
- [ ] 06 `baselines/common.py`, `clarke_wright.py`, `ortools_solver.py`: toy CW = `[0, 1, 2, 0, 3, 0]`
- [ ] 07 `env/executor.py`, `baselines/runner.py`: baseline table (CW 8.0010, hindsight ≈ 6.04)
- [ ] 08 `env/svrp_env.py`, `baselines/nn_online.py`: `test_env.py` (7) + `test_baselines.py` pass; nn_online 8.0282
- [ ] 09 `models/paper_net.py`: 167,040 parameters at N = 5
- [ ] 10 `inference.py`: beam1 = greedy = 9.4989 (untrained, seeded)
- [ ] 11 `train/reinforce.py`, `scripts/train.py`: `test_models.py -k "not graphnet"` passes; your N = 10 run beats CW (val < 8.007)
- [ ] 12 `scripts/evaluate.py`, `plot.py`, `inspect_policy.py`: your E2 table; your own `LOG.md`
- [ ] 13 `models/graph_net.py` (743,817 parameters), events; all tests pass
- [ ] 14 (optional) `baselines/aco.py`, `scripts/tune_aco.py`
- [ ] 15 your own `tests/`: all 34 pass
- [ ] 16 `scripts/check_gpu.py` (benchmark), `plot.py`, `inspect_policy.py`: nn_online / nn_split columns = 8.028 / 7.867
- [ ] 17 a resumable queue script (second run prints only `skip`), your own `SPEC.md`, `README.md`, `LOG.md`
