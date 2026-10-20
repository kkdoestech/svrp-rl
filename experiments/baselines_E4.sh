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
