# Rebuild svrp-rl yourself — a layer-by-layer course

You studied the paper (*Reinforcement Learning for Solving Stochastic Vehicle Routing Problem*, Iklassov
et al. 2023) in `a.md`, `b.md`, `part 5.md` and `part 6.md`. Then a working project was built from it:
the `svrp/` package, the `scripts/`, the `tests/`, and the experiment log `experiments/LOG.md`.

This folder teaches you that project **one layer at a time**, so that you can rebuild it with your own hands.

By the end you will have written, yourself:

- a random "delivery world" generator, where demand and travel time depend on the weather plus noise,
- the classical solvers (Clarke–Wright, OR-Tools, nearest neighbour) and a simulator that punishes bad plans,
- a reinforcement-learning environment (the "game" the AI plays),
- the paper's neural network (Conv1d + LSTM + attention), trained with REINFORCE,
- a fair evaluation pipeline, and the experiments that showed RL beating OR-Tools by ~24% at N = 10.

Nothing in this folder is code. Every lesson **quotes the real file** and explains it line by line.
You read it, understand it, and then write your own version.

---

## How to use one lesson (the loop)

```
 1. Read "Why this layer exists" and "Concepts from zero"          (understand the idea)
 2. Open the real file next to the lesson (VS Code split view, or GitHub)
 3. Read the code walkthrough, chunk by chunk, with the real file open
 4. CLOSE the real file
 5. Write your own version in learning/workspace/mysvrp/...         (type it, don't paste it)
 6. Run the test from the "Verify" section
 7. Stuck or failing? Re-open the real file, compare, fix, and write down what you learned
```

Step 4 matters most. Copying code in teaches your fingers, not your head. When you can write a
function from memory, and explain every line, you have learned it.

> **Line numbers** in the lessons refer to the files as they are *now* in this folder. If you read the
> code on GitHub, commit and push first (`git add -A && git commit -m "..." && git push`), so that
> GitHub shows the same version.

---

## Your workspace (where your own code goes)

You will write a copy of the package called **`mysvrp`**, with the same folders and file names as
`svrp/`. The original stays untouched, so you can always compare.

```
learning/
└── workspace/            <- you create this
    ├── mysvrp/           <- your package (mirror of svrp/)
    │   ├── __init__.py   <- empty file; it makes the folder a "package"
    │   ├── config.py
    │   ├── data.py
    │   ├── inference.py
    │   ├── env/          (__init__.py, generator.py, estimates.py, svrp_env.py, executor.py)
    │   ├── baselines/    (__init__.py, common.py, clarke_wright.py, ortools_solver.py, nn_online.py, runner.py, aco.py)
    │   ├── models/       (__init__.py, paper_net.py, graph_net.py)
    │   └── train/        (__init__.py, reinforce.py)
    ├── tests/            <- copies of the real tests, with "svrp" replaced by "mysvrp"
    └── scripts/          <- your train.py, evaluate.py, ...
```

You create each file when its lesson tells you to. You don't need them all on day one.

**How to run your code.** Always run from inside `learning/workspace`. `uv` walks up the folders, finds
the project's `pyproject.toml`, and reuses the same `.venv`, so nothing needs installing again:

```bash
cd learning/workspace
uv run python -m pytest tests -q                         # run your copies of the tests
uv run python -m pytest tests/test_generator.py -q       # just one test file
uv run python -m scripts.train --n_customers 10 --epochs 1 --device cpu   # run your script as a module
```

Why `python -m ...`? It adds the *current folder* (`learning/workspace`) to Python's search path, so
`import mysvrp` finds your package. This course's author ran exactly these commands on a copy of the
project, and they work.

> **The one trap to avoid.** The original `svrp` package is *installed* in the `.venv`. If one of your
> files says `from svrp.config import ...` instead of `from mysvrp.config import ...`, Python silently
> uses the *original* code, and your tests pass without testing your work. Check which file actually
> got imported:
> ```bash
> uv run python -c "import sys; sys.path.insert(0, '.'); import mysvrp.env.generator as m; print(m.__file__)"
> ```
> It must print a path inside `learning/workspace/mysvrp`.

**Copying a test** (every lesson has a "Verify" section that tells you which one):

```bash
cd learning/workspace
mkdir -p tests
sed 's/\bsvrp\./mysvrp./g; s/from svrp /from mysvrp /g' ../../tests/test_generator.py > tests/test_generator.py
```

(On Windows, run this in the Git Bash terminal; VS Code can open one for you.) You can also copy the
file by hand and use Find/Replace on `svrp.` → `mysvrp.`.

---

## The layers (what depends on what)

Each layer only uses the layers above it. Build top to bottom, and **finish each layer's test before
starting the next one**.

```
Layer 0   tools, Python, probability, tensors ................ 00, 01, 02
   │
Layer 1   svrp/config.py ...................................... 03   "the settings of one experiment"
   │
Layer 2   svrp/env/generator.py ............................... 04   θ = θ̄ + wᵀαw + ε  (random days)
   │
Layer 3   svrp/env/estimates.py, svrp/data.py ................. 05   truth vs belief, k-NN, Batch
   │
   ├───────────────────────────────┐
   │                               │
Layer 4   classical planners       Layer 6   the RL environment
   baselines/common.py             env/svrp_env.py ................ 08
   baselines/clarke_wright.py      baselines/nn_online.py
   baselines/ortools_solver.py     │
   ........................ 06     Layer 7   models/paper_net.py ....... 09
   │                               │
Layer 5   reality check            Layer 8   inference.py .............. 10
   env/executor.py                 │
   baselines/runner.py ..... 07    Layer 9   train/reinforce.py,
   │                               │         scripts/train.py .......... 11
   └───────────────┬───────────────┘
                   │
Layer 10  scripts/evaluate.py, plot.py, inspect_policy.py, experiments/ ... 12, 16, 17
Layer 11  models/graph_net.py, shared baseline, events ..................... 13
optional  baselines/aco.py, scripts/tune_aco.py ............................ 14
alongside tests/ (one file per layer group) ................................ 15
```

## The course

| #  | lesson | you build | time* | milestone when done |
|----|--------|-----------|-------|---------------------|
| 00 | [Prerequisites](00_prerequisites.md) | — | 3–6 h | you can read the Python, probability and matrix notation used everywhere |
| 01 | [Setup and tools](01_setup_and_tools.md) | your workspace | 1–2 h | `check_gpu.py` passes; you know why torch is pinned |
| 02 | [Tensors crash course](02_tensors.md) | small experiments | 3–5 h | you can predict tensor shapes before running code |
| 03 | [Config](03_config.md) | `mysvrp/config.py` | 1 h | every experiment setting becomes a CLI flag |
| 04 | [The random world](04_generator.md) | `mysvrp/env/generator.py` | 4–6 h | `test_generator.py`: signal ratio 0.6 / 0.2 / 0.2 checked |
| 05 | [Estimates and data](05_estimates_and_data.md) | `env/estimates.py`, `data.py` | 3–4 h | k-NN beats the constant guess when weather matters |
| 06 | [Plans and classical solvers](06_plans_and_classical_solvers.md) | `baselines/common.py`, `clarke_wright.py`, `ortools_solver.py` | 3–4 h | CW and OR-Tools produce valid plans |
| 07 | [Executor and runner](07_executor_and_runner.md) | `env/executor.py`, `baselines/runner.py` | 3 h | **you reproduce the baseline table** (planning from estimates ends up ~33% above hindsight) |
| 08 | [The environment (MDP)](08_environment.md) | `env/svrp_env.py`, `baselines/nn_online.py` | 4–6 h | `test_env.py` passes; a random policy always finishes |
| 09 | [The policy network](09_policy_network.md) | `models/paper_net.py` | 3–4 h | the network outputs one score per node |
| 10 | [Rollouts and inference](10_rollout_and_inference.md) | `inference.py` | 3–4 h | greedy / sampling / beam / a-priori all run |
| 11 | [Training with REINFORCE](11_training_reinforce.md) | `train/reinforce.py`, `scripts/train.py` | 4–6 h | **your RL beats Clarke–Wright at N = 10** |
| 12 | [Evaluation and experiments](12_evaluation_and_experiments.md) | `scripts/evaluate.py`, `plot.py`, `inspect_policy.py` | 4–6 h | **you reproduce the E2 table**: RL ≈ −24% vs OR-Tools |
| 13 | [Improvements](13_improvements.md) | `models/graph_net.py`, shared baseline, events | 5–8 h | a Transformer policy that works for any N |
| 14 | [ACO (optional)](14_aco_optional.md) | `baselines/aco.py`, `scripts/tune_aco.py` | 2–3 h | the paper's "champion" classical method, tuned |
| 15 | [The tests](15_tests.md) | all four files in `tests/` | 3–5 h, spread out | you can *write* the tests, not just copy them (34 pass) |
| 16 | [Helper scripts](16_helper_scripts.md) | `scripts/check_gpu.py`, `plot.py`, `inspect_policy.py` | 2–3 h | your own benchmark table, plots and behaviour report |
| 17 | [Experiment scripts and documents](17_experiment_scripts_and_docs.md) | `experiments/*.sh`, `SPEC.md`, `README.md`, `LOG.md` | 1–2 h | a resumable experiment queue and your own project docs |

**Every line of every code file is quoted and explained somewhere in lessons 01–17.** Lessons 15–17 aren't
a separate track: do each part when its layer is done (lesson 15 says which test belongs to which layer).

\* Rough times for a first read and rebuild. Going slower is fine. Understanding beats speed.

Reference pages, for any time:

- [REFERENCE_MAP.md](REFERENCE_MAP.md): every file in the project, what it does, what it depends on, and which lesson covers it.
- [GLOSSARY.md](GLOSSARY.md): every term in plain words (recourse, mask, logit, baseline, …).
- [READING_LIST.md](READING_LIST.md): papers and tutorials, per lesson, marked must / should / optional.

The project's own documents, worth reading along the way:

- [`../SPEC.md`](../SPEC.md): the exact rules of the project (what is built and why). Lessons 04–12 implement it.
- [`../experiments/LOG.md`](../experiments/LOG.md): the lab notebook, with every result and what it means.
- [`../README.md`](../README.md): how to run the original project.

---

## Golden rules

1. **Small first.** Test with N = 6 or 10 customers, a batch of 8, on the CPU. Bugs show up in seconds,
   not hours. Only go big when the small version is correct.
2. **One layer at a time.** Don't start lesson 05 until your `generator.py` passes its test. Every later
   layer trusts the earlier ones.
3. **Print shapes.** When something breaks in PyTorch, `print(x.shape)` finds it nine times out of ten.
4. **Ask "why this line?"** Every line in the reference exists for a reason, and the lessons tell you the
   reason. If you can't say it in your own words, re-read that part.
5. **Keep a journal.** Write down each bug you hit and how you fixed it. Those notes become your best
   teacher (and the start of your own `LOG.md`).
6. **Different is fine, if tested.** You may name things differently or write a loop where the
   reference uses a tensor trick. If the test passes and you understand why, that's a win.

Ready? Start with [00 — Prerequisites](00_prerequisites.md). If you already know Python dataclasses,
mean/variance and matrices, skim it and do the self-test at the end.
