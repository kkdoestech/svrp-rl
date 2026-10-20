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
