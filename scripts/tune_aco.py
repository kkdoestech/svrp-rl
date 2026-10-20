"""Pick ACO's (alpha, beta, ants) on VALIDATION days so ACO is a fair, tuned opponent.

    uv run python scripts/tune_aco.py --n_customers 20 --n_days 100
Writes results/aco_params.json, which evaluate.py then uses.
"""
import argparse
import json
from pathlib import Path

from svrp.baselines.aco import PARAM_GRID
from svrp.baselines.runner import run_planner
from svrp.config import EnvConfig, add_args, from_args
from svrp.data import World


def main():
    p = argparse.ArgumentParser()
    add_args(p, EnvConfig)
    p.add_argument("--n_days", type=int, default=100)
    a = p.parse_args()
    cfg = from_args(a, EnvConfig)
    batch = World(cfg).eval_set("val", a.n_days)

    best = None
    for alpha, beta, ants in PARAM_GRID:
        kw = {"alpha": alpha, "beta": beta, "n_ants": ants}
        out = run_planner("aco", batch, cfg.delivery, **kw)
        cost = out["cost"].mean()
        print(f"alpha={alpha:2d} beta={beta:2d} ants={ants:2d}:  val cost {cost:.4f}  "
              f"({out['sec_per_instance']*1000:.0f} ms/day)")
        if best is None or cost < best[0]:
            best = (cost, kw)
    out = Path(__file__).resolve().parents[1] / "results" / "aco_params.json"
    out.write_text(json.dumps(best[1]))
    print(f"best {best[1]} -> {out}")


if __name__ == "__main__":     # required on Windows: worker processes re-import this file
    main()
