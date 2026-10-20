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

# E3: headline N=20, paper model + critic (60 min)
run E3_n20_paper_critic_s1 --n_customers 20 --time_limit_min 60 --seed 1

# E2 seeds 2-3 (N=10 headline + arm A0 of improvement I1); seed 1 = runs/E2_n10_lr1e-3
run E2_n10_critic_s2 --n_customers 10 --time_limit_min 15 --seed 2
run E2_n10_critic_s3 --n_customers 10 --time_limit_min 15 --seed 3

# E5 / I1: shared (POMO-style) baseline instead of the critic, N=10, 3 seeds
for s in 1 2 3; do
  run E5_n10_paper_shared_s$s --n_customers 10 --time_limit_min 15 --seed $s --baseline shared --n_shared 8
done

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
