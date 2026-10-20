# 17 — Experiment scripts and project documents (`experiments/*.sh`, `SPEC.md`, `README.md`, `LOG.md`)

| | |
|---|---|
| **Layer** | 10–11 (running experiments at scale, and writing them down) |
| **Files covered** | `experiments/queue_1.sh` (40 lines), `experiments/queue_2.sh` (30), `experiments/baselines_E4.sh` (11); the structure of `SPEC.md`, `README.md`, `experiments/LOG.md`, `results/` |
| **You build** | your own queue scripts, your own `SPEC.md`, `README.md` and `LOG.md` |
| **Time** | 1–2 hours |
| **Before this** | [12 — Evaluation and experiments](12_evaluation_and_experiments.md) |

## Why this lesson exists

Phase 4 of the project needed **ten** training runs of 15 minutes, plus ten sets of baselines. Typing
them one by one, and remembering which already ran, is how mistakes happen. The answer is a *script*:
the list of experiments becomes a file you can read, re-run and resume.

The project's documents are part of "the codebase" too. `SPEC.md` says what is built and why. `README.md`
says how to run it. `LOG.md` says what was found. To rebuild the project completely, you write your own
versions.

## Concepts from zero: just enough bash

These scripts run in **bash** (on Windows: the Git Bash terminal).

| bash | meaning |
|---|---|
| `#!/usr/bin/env bash` | first line: "run me with bash" |
| `# text` | a comment |
| `cd "$(dirname "$0")/.."` | `$0` is the script's own path, `dirname` its folder, `..` one up: go to the project root |
| `NAME="text"` / `$NAME` | set / use a variable (no spaces around `=`!) |
| `name() { ...; }` | define a function. Inside it, `$1` is the 1st argument, and `"$@"` is all arguments |
| `shift` | drop `$1`, so `"$@"` becomes the remaining arguments |
| `if [ -f path ]; then ...; fi` | if a file exists |
| `for s in 1 2 3; do ...; done` | a loop |
| `cmd > file 2>&1` | send normal output *and* errors into a file |
| `$(cmd)` | insert a command's output, e.g. `$(date +%H:%M)` |
| `${VAR:-default}` | the variable's value, or `default` if it's unset |
| `cmd1 \| cmd2` | pipe: cmd1's output becomes cmd2's input |

## `experiments/queue_1.sh`, complete

<!-- quote: experiments/queue_1.sh -->
```bash
#!/usr/bin/env bash
# GPU job queue, part 1 (runs strictly one job at a time: two jobs on the GTX 1060 are slower than one).
# Resumable: a run that already has runs/<name>/DONE is skipped.  Usage: bash experiments/queue_1.sh
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

- **Lines 2–3:** the header comment says *why* the queue runs one job at a time (measured in E3: two jobs
  on the GTX 1060 are slower than one), that it is resumable, and how to run it.
- **Line 4:** always work from the project root, wherever the script was started from.
- **Line 5:** `COMMON` holds the settings every run shares. `--epochs 100000` means "effectively forever", so
  each run stops at its time budget instead. lr 1e-3 is the winner of E2.
- **`run()`:**
  - `name=$1; shift` takes the run name and leaves its own flags in `"$@"`.
  - It skips the run if `runs/<name>/DONE` exists. `train()` writes `DONE` only at a normal finish (lesson
    11), so a crashed or interrupted run is re-done.
  - It prints a start time, trains with `COMMON` + the run's flags, and sends all output into the run's
    `stdout.log`. `-u` (unbuffered) makes the log fill in real time.
  - It prints the end time and the last log line (`tail -1`, the "best val …" line), so the queue's own
    log reads like a summary.

<!-- quote: experiments/queue_1.sh -->
```bash
# E3: headline N=20, paper model + critic (60 min)
run E3_n20_paper_critic_s1 --n_customers 20 --time_limit_min 60 --seed 1

# E2 seeds 2-3 (N=10 headline + arm A0 of improvement I1); seed 1 = runs/E2_n10_lr1e-3
run E2_n10_critic_s2 --n_customers 10 --time_limit_min 15 --seed 2
run E2_n10_critic_s3 --n_customers 10 --time_limit_min 15 --seed 3

# E5 / I1: shared (POMO-style) baseline instead of the critic, N=10, 3 seeds
for s in 1 2 3; do
  run E5_n10_paper_shared_s$s --n_customers 10 --time_limit_min 15 --seed $s --baseline shared --n_shared 8
done
```

- **E3:** the N = 20 headline run, 60 minutes.
- **E2 seeds 2–3:** the N = 10 headline needs 3 seeds. Seed 1 was the learning-rate run, so these two add
  seeds 2 and 3. They also form arm A0 (the critic) of improvement I1.
- **E5 (I1):** the shared baseline, 3 seeds. Same world, budget and lr as A0; only `--baseline` differs.
  `$s` inside the run name gives `E5_n10_paper_shared_s1`, `…_s2`, `…_s3`.

<!-- quote: experiments/queue_1.sh -->
```bash
# E4: replicate the paper's findings, N=10, paper model + critic, 15 min, seed 1
T="--n_customers 10 --time_limit_min 15 --seed 1"
run E4_n10_const_noweather $T --estimate constant --weather_input false
run E4_n10_const_weather   $T --estimate constant --weather_input true
run E4_n10_oracle          $T --estimate oracle_mean
run E4_n10_demand_only     $T --stochastic_cost false
run E4_n10_cost_only       $T --stochastic_demand false
run E4_n10_full            $T --delivery full
run E4_n10_flexible        $T --positions flexible
run E4_n10_B0              $T --b_ratio 0 --gamma_ratio 0.4
run E4_n10_B04             $T --b_ratio 0.4 --gamma_ratio 0
run E4_n10_phi02           $T --fill_rate 0.2
echo "$(date +%H:%M) queue_1 finished"
```

**E4: replicating the paper's findings**, one setting changed per run (`$T` holds the shared part). Each line
maps to one finding in `part 6.md`:

| run | changes | paper finding |
|---|---|---|
| `const_noweather` | constant estimate, weather hidden | #1, #3: is weather info worth anything? |
| `const_weather` | constant estimate, but the network sees the weather | can the network learn the weather effect itself? |
| `oracle` | perfect weather knowledge | the ceiling of #1 |
| `demand_only` / `cost_only` | only one kind of randomness | #2: which uncertainty hurts more? |
| `full` | indivisible orders | #5: partial vs full delivery |
| `flexible` | a new city every day | #4: fixed vs flexible positions |
| `B0` / `B04` | weather share 0 / 0.4 (A stays 0.6) | the signal-ratio experiment |
| `phi02` | bigger trucks (Q = 50) | the fill-rate experiment |

The last line prints when the whole queue finished.

## `experiments/queue_2.sh`, complete

<!-- quote: experiments/queue_2.sh -->
```bash
#!/usr/bin/env bash
# GPU job queue, part 2: Phase 5 improvements + N=20 seeds. Run after queue_1 has finished.
# BASELINE = whichever REINFORCE baseline won improvement I1 (critic or shared), see experiments/LOG.md.
# Usage: BASELINE=shared bash experiments/queue_2.sh
cd "$(dirname "$0")/.."
BASELINE=${BASELINE:-critic}
COMMON="--batch_size 512 --epochs 100000 --lr 1e-3 --critic_lr 1e-3"
BFLAGS="--baseline $BASELINE --n_shared 8"

run() {
  name=$1; shift
  if [ -f "runs/$name/DONE" ]; then echo "skip $name"; return; fi
  echo "$(date +%H:%M) start $name"
  mkdir -p "runs/$name"
  uv run python -u scripts/train.py $COMMON --run_name "$name" "$@" > "runs/$name/stdout.log" 2>&1
  echo "$(date +%H:%M) end   $name: $(tail -1 runs/$name/stdout.log)"
}
```

The same `run()` pattern, with one addition:

- `BASELINE=${BASELINE:-critic}`: which REINFORCE baseline to use is decided **after** E5 tells us which one
  won. The script reads it from an environment variable, so you run `BASELINE=shared bash
  experiments/queue_2.sh` without editing the file. If it's unset, the default is `critic`.
- `BFLAGS` turns that choice into flags for every run below.

<!-- quote: experiments/queue_2.sh -->
```bash
# I2: GraphNet (Transformer + cost-biased attention) on top of the I1 winner, N=10, 3 seeds
for s in 1 2 3; do
  run E6_n10_graph_${BASELINE}_s$s --n_customers 10 --time_limit_min 15 --seed $s --model graph $BFLAGS
done

# I4: an extra external variable (local events). Same days; does seeing the event flag help?
EV="--n_customers 10 --time_limit_min 15 --event_prob 0.3 --event_boost 1.5 $BFLAGS"
for s in 1 2; do
  run E7_n10_events_seen_s$s   $EV --seed $s --event_input true
  run E7_n10_events_hidden_s$s $EV --seed $s --event_input false
done
echo "$(date +%H:%M) queue_2 finished"
```

- **E6 (I2):** GraphNet on top of the I1 winner, 3 seeds. `${BASELINE}` inside the name keeps runs with
  different baselines apart.
- **E7 (I4):** events on 30% of days with a +150% boost near the venue. *Seen* vs *hidden* event flag, 2 seeds
  each. The days are identical (lesson 13), so only the information differs.

## `experiments/baselines_E4.sh`, complete

<!-- quote: experiments/baselines_E4.sh -->
```bash
#!/usr/bin/env bash
# Classical baselines for every Phase 4 world (CPU only; 5 planner processes so training keeps a core).
cd "$(dirname "$0")/.."
M="nn_online,nn_split,cw,ortools_gls,hindsight"
B="--n_customers 10 --methods $M --workers 5 --device cpu"
for flags in "--estimate constant" "--estimate oracle_mean" "--stochastic_cost false" "--stochastic_demand false" \
             "--delivery full" "--positions flexible" "--b_ratio 0 --gamma_ratio 0.4" "--b_ratio 0.4 --gamma_ratio 0" \
             "--fill_rate 0.2"; do
  echo "=== $flags"
  uv run python scripts/evaluate.py $B $flags 2>&1 | grep -vE "^running|^note"
done
```

The classical side of E4, on the **CPU**, at the same time as the GPU queue:

- `M`: the methods to compute in every world. `B`: the shared flags.
  - `--workers 5` leaves one of the six cores for the training process.
  - `--device cpu` keeps the heuristics off the busy GPU.
- The loop: one string of flags per world, the same nine worlds as the E4 runs. (`const_weather` shares
  the constant-estimate world: `weather_input` only changes the *network's* input, so it isn't part of the
  world's tag, lesson 03.)
- `grep -vE "^running|^note"` hides progress lines, so the log holds only the result tables. Every result
  is also cached per day (`results/eval/…`) and appended to `results/results.csv`.

## The project's documents

### `SPEC.md`: what is built, and why

Read the project's `SPEC.md` next to this structure:

1. **Goal**, and what "reproduce" means here (relative claims, not absolute numbers).
2. **S1–S10**, one numbered section per part of the system: generator, estimates, env, executor,
   baselines, models, training, inference, evaluation, keep-or-reject rule. Each is short, precise, and
   says how it will be checked.
3. **Deviations from the paper**, a table: what the paper says, what we do, and why.

Write yours *before* coding (the user asked for exactly that at the start of this project). It forces
decisions early, such as "Φ = E[ξ]/Q" or "α fixed per city", instead of discovering them halfway.

### `README.md`: how to run it

The project's `README.md` holds: one paragraph on what the project is, `uv sync` + check + tests, the typical
workflow (evaluate baselines → train → evaluate RL → plot), a note that every config field is a flag,
and the folder layout. A stranger (or you in six months) should be able to run everything from it alone.

### `experiments/LOG.md`: what was found

One entry per experiment (template in [lesson 12](12_evaluation_and_experiments.md)). Look at how the
project's entries are written:

- **E0** records the setup *and* the benchmark that decided the batch size.
- **E1** gives two tables, then "What this tells us": three plain-language conclusions, including an honest
  difference from the paper.
- **E2** is a result that looked too good, followed by the checks, the control, the behaviour analysis,
  and a comparison with the paper.

### `results/`: generated, but kept

| path | written by | contents |
|---|---|---|
| `results/eval/<world tag>/<split><n_days>/<method>.npz` | `evaluate.py` | per-day costs, recourse events, timing |
| `results/results.csv` | `evaluate.py` | one row per method per evaluation, append-only |
| `results/plots/*.png` | `plot.py` | learning curves |
| `results/aco_params.json` | `tune_aco.py` | ACO's tuned settings |

`runs/` (checkpoints) is ignored by git, but `results/` is kept: small files that document findings.
`uv.lock` is generated by `uv` and never edited by hand (lesson 01).

## Rebuild it yourself

1. Write `learning/workspace/experiments/queue_mine.sh` with the `run()` function, and put two of your own
   experiments in it (for example lr 3e-3, and batch 1024). Make it resumable.
2. Write your `SPEC.md` (even half a page), your `README.md`, and keep your `LOG.md` going.

## Verify

```bash
cd learning/workspace
bash experiments/queue_mine.sh        # runs your experiments
bash experiments/queue_mine.sh        # run it again: every finished run must print "skip <name>"
```

The second run printing only `skip …` lines proves the queue is resumable.

## Check your understanding

1. What happens if your PC restarts in the middle of `queue_1.sh`, and you start it again?
2. Why is the baseline choice an environment variable in `queue_2.sh`, not a hard-coded flag?
3. Why does `baselines_E4.sh` use `--workers 5`, not 6?

**Answers** (try the questions first, then check here)

1. Finished runs (those with `DONE`) are skipped. The interrupted run has no `DONE`, so it starts again
   from scratch. The remaining runs follow.
2. It depends on a result (E5) that didn't exist when the script was written. An environment variable
   lets you choose at launch time without editing the script.
3. Training runs at the same time and needs a CPU core for its Python loop. Starving it would slow the GPU
   job, which is time-budgeted, and so silently weaken that experiment.

**You've now seen every file in the project.** Check off the last items in [REFERENCE_MAP.md](REFERENCE_MAP.md).
