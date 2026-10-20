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

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = {"nn_online": "yes", "nn_split": "yes", "cw": "yes", "ortools_gls": "yes", "ortools_tabu": "yes", "aco": "yes",
          "hindsight": "reference", "greedy": "yes", "apriori": "yes", "beam": "lookahead", "best": "hindsight"}


def label(method: str) -> str:
    key = method.split(":")[-1]
    for k, v in DEPLOY.items():
        if key.startswith(k):
            return v
    return "?"


def paired_ci(x: np.ndarray, ref: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Mean relative gap (x − ref)/mean(ref) in %, with a 95% bootstrap CI over days."""
    rng = np.random.default_rng(seed)
    d = x - ref
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1) / ref[idx].mean(1) * 100
    return d.mean() / ref.mean() * 100, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


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
    if device.type == "cuda":
        torch.cuda.synchronize()
    out["sec_total"] = time.perf_counter() - t0
    out["sec_per_instance"] = out["sec_total"] / batch.size
    return out


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
