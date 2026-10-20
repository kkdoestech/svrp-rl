"""What did the policy learn?  Behaviour statistics of a trained policy vs the heuristics, plus one example day.

    uv run python scripts/inspect_policy.py --run runs/E2_n10_lr1e-3 --n_days 1000
"""
import argparse

import torch

from svrp.data import World
from svrp.env.svrp_env import EPS, SVRPEnv
from svrp.inference import masked_log_probs
from svrp.train.reinforce import load_policy, resolve_device


@torch.no_grad()
def drive(choose, batch, delivery, weather_input=True):
    """Runs a decision rule and records behaviour: trips, splits, load left when going home."""
    env = SVRPEnv(batch, delivery, weather_input)
    state = choose.init(env) if hasattr(choose, "init") else None
    B = batch.size
    trips = torch.zeros(B, device=env.pos.device)
    early_home = torch.zeros(B, device=env.pos.device)       # went home with load still on the truck
    left_when_home = torch.zeros(B, device=env.pos.device)
    routes = [[0] for _ in range(min(B, 1))]
    for _ in range(env.max_steps()):
        if env.done.all():
            break
        a, state = choose(env, state)
        going_home = (a == 0) & (env.pos != 0) & ~env.done
        trips += going_home
        early_home += going_home & (env.load > EPS)
        left_when_home += torch.where(going_home, env.load, 0.0)
        if not env.done[0]:
            routes[0].append(int(a[0]))
        env.step(a)
    return env, {"cost": env.total_cost.mean().item(), "trips": trips.mean().item(),
                 "splits (load ran out at a customer)": env.n_fail.mean().item(),
                 "trips ending with load left": (early_home.sum() / trips.sum()).item(),
                 "avg load left when going home": (left_when_home.sum() / trips.sum().clamp_min(1)).item()}, routes[0]


class Policy:
    def __init__(self, net):
        self.net = net

    def init(self, env):
        return self.net.init_state(env)

    def __call__(self, env, state):
        logits, state = self.net(env, state)
        return masked_log_probs(logits, env.mask()).argmax(-1), state


def nearest(split):
    def choose(env, state):
        m = env.mask()
        cust = m.clone()
        cust[:, 0] = False
        c = env.batch.cost_hat[env.ar, env.pos]
        fits = cust if split else cust & (env.observed_demand() <= env.load[:, None] + EPS)
        inf = torch.tensor(float("inf"), device=c.device)
        nf, na = torch.where(fits, c, inf).argmin(1), torch.where(cust, c, inf).argmin(1)
        a = torch.where(fits.any(1), nf, torch.where((env.pos == 0) & cust.any(1), na, 0))
        return a, state
    return choose


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n_days", type=int, default=1000)
    a = p.parse_args()
    device = resolve_device("auto")
    net, env_cfg, _, _ = load_policy(a.run, device)
    batch = World(env_cfg).eval_set("test", a.n_days).to(device)
    Q = env_cfg.capacity
    print(f"world {env_cfg.tag()}   Q = {Q:g}\n")
    rows = {}
    for name, rule in [("nn_online", nearest(False)), ("nn_split", nearest(True)), ("RL greedy", Policy(net))]:
        _, stats, route = drive(rule, batch, env_cfg.delivery, env_cfg.weather_input)
        rows[name] = (stats, route)
    keys = list(next(iter(rows.values()))[0])
    print(f"{'':38s}" + "".join(f"{n:>12s}" for n in rows))
    for k in keys:
        print(f"{k:38s}" + "".join(f"{rows[n][0][k]:12.3f}" for n in rows))

    b = batch.index(slice(0, 1))
    print("\nExample day 0 — customer: estimate -> true demand")
    print("  " + "  ".join(f"{i}: {b.dem_hat[0, i]:.1f}->{b.demand[0, i]:.1f}" for i in range(1, b.n_nodes)))
    for n, (_, route) in rows.items():
        print(f"  {n:10s} route: {' '.join(map(str, route))}")


if __name__ == "__main__":
    main()
