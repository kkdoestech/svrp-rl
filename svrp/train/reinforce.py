"""REINFORCE training (part 5.md).

    loss_actor  = mean[ (C − b) · Σ_t log π(a_t | s_t) ]      C = realized cost of the sampled route
    loss_critic = mean[ (b_φ − C)² ]                           (only for baseline="critic")

Minimizing loss_actor makes routes that were cheaper than expected (C < b) more likely, and routes that
were more expensive than expected less likely. The baseline b is the "class average" from the notes:
    critic   b = b_φ(initial observation), a separate network           (the paper)
    shared   run M sampled routes on the SAME day, b = their mean cost   (POMO-style, improvement I1)
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from svrp.config import EnvConfig, TrainConfig, to_dict
from svrp.data import World, train_generator
from svrp.env.svrp_env import SVRPEnv
from svrp.inference import rollout
from svrp.models.paper_net import Critic, PaperNet

RUNS = Path(__file__).resolve().parents[2] / "runs"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_policy(tr: TrainConfig, env: EnvConfig) -> nn.Module:
    if tr.model == "paper":
        return PaperNet(env.n_customers + 1, env.n_external, tr.embed_dim)
    if tr.model == "graph":
        from svrp.models.graph_net import GraphNet
        return GraphNet(env.n_external, tr.embed_dim)
    raise ValueError(tr.model)


def default_run_name(env: EnvConfig, tr: TrainConfig) -> str:
    return f"{tr.model}_{tr.baseline}_{env.tag()}_seed{tr.seed}"


@torch.no_grad()
def validate(policy, batch, env: EnvConfig, chunk: int = 1000) -> float:
    policy.eval()
    costs = [rollout(policy, batch.index(slice(i, i + chunk)), env.delivery, env.weather_input).cost
             for i in range(0, batch.size, chunk)]
    policy.train()
    return torch.cat(costs).mean().item()


def train(env_cfg: EnvConfig, tr: TrainConfig, log=print) -> Path:
    device = resolve_device(tr.device)
    torch.manual_seed(tr.seed)
    run_dir = RUNS / (tr.run_name or default_run_name(env_cfg, tr))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"env": to_dict(env_cfg), "train": to_dict(tr)}, indent=2))

    world = World(env_cfg)
    val = world.eval_set("val", tr.val_size).to(device)
    gen = train_generator(tr.seed)

    policy = make_policy(tr, env_cfg).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=tr.lr)
    critic = critic_opt = None
    if tr.baseline == "critic":
        critic = Critic(env_cfg.n_customers + 1, env_cfg.n_external, tr.embed_dim).to(device)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=tr.critic_lr)
    elif tr.baseline != "shared":
        raise ValueError(tr.baseline)
    M = tr.n_shared if tr.baseline == "shared" else 1
    if tr.batch_size % M:
        raise ValueError("batch_size must be divisible by n_shared")

    best = validate(policy, val, env_cfg)
    log(f"[{run_dir.name}] device={device}  params={sum(p.numel() for p in policy.parameters()):,}  "
        f"untrained val cost={best:.4f}")
    torch.save({"policy": policy.state_dict(), "epoch": 0, "val": best}, run_dir / "best.pt")

    steps = math.ceil(tr.epoch_size / tr.batch_size)
    t0 = time.time()
    with open(run_dir / "log.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "minutes", "train_cost", "val_cost", "loss", "critic_mse", "grad_norm"])
        for epoch in range(1, tr.epochs + 1):
            sums = torch.zeros(4, device=device)
            for _ in range(steps):
                batch = world.sample(tr.batch_size // M, gen)
                if M > 1:
                    batch = batch.repeat_interleave(M)
                batch = batch.to(device)
                r = rollout(policy, batch, env_cfg.delivery, env_cfg.weather_input, decode="sample")
                if critic is not None:
                    feats = SVRPEnv(batch, env_cfg.delivery, env_cfg.weather_input).node_features()
                    b = critic(feats)
                    critic_loss = (b - r.cost).pow(2).mean()
                    critic_opt.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), tr.grad_clip)
                    critic_opt.step()
                    adv = r.cost - b.detach()
                else:
                    c = r.cost.view(-1, M)
                    adv = (c - c.mean(1, keepdim=True)).view(-1)
                    critic_loss = torch.zeros((), device=device)
                loss = (adv * r.logp).mean()
                opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(policy.parameters(), tr.grad_clip)
                opt.step()
                sums += torch.stack([r.cost.mean().detach(), loss.detach(), critic_loss.detach(), gn.detach()])
            sums = (sums / steps).tolist()
            v = validate(policy, val, env_cfg)
            minutes = (time.time() - t0) / 60
            w.writerow([epoch, f"{minutes:.2f}", f"{sums[0]:.4f}", f"{v:.4f}", f"{sums[1]:.4f}",
                        f"{sums[2]:.4f}", f"{sums[3]:.3f}"])
            fh.flush()
            flag = ""
            if v < best:
                best, flag = v, " *"
                torch.save({"policy": policy.state_dict(), "epoch": epoch, "val": v}, run_dir / "best.pt")
            log(f"epoch {epoch:3d}  {minutes:6.1f} min  train {sums[0]:.4f}  val {v:.4f}{flag}")
            if minutes > tr.time_limit_min:
                log(f"time budget of {tr.time_limit_min} min reached")
                break
    log(f"best val {best:.4f}  ->  {run_dir}")
    (run_dir / "DONE").write_text(f"{best:.6f}\n")
    return run_dir


def load_policy(run_dir: Path, device="cpu"):
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    env_cfg, tr = EnvConfig(**cfg["env"]), TrainConfig(**cfg["train"])
    policy = make_policy(tr, env_cfg).to(device)
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, env_cfg, tr, ckpt
