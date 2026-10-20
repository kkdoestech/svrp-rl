"""Running a policy on days: greedy / sampling rollouts, beam search, and a-priori planning.

Deployability (SPEC S8):
    greedy online        deployable     uses only what has been revealed so far
    apriori              deployable     plan on the estimates, then drive it with recourse
    beam (online)        LOOKAHEAD      beams "visit" customers on the real day = peeks at hidden demand
    best-of-n sampling   HINDSIGHT      chooses the rollout with the best realized cost
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from svrp.data import Batch
from svrp.env.executor import execute_many
from svrp.env.svrp_env import SVRPEnv


@dataclass
class Rollout:
    cost: torch.Tensor      # (B,) realized total cost
    logp: torch.Tensor      # (B,) Σ_t log π(a_t)  (carries gradients when sampling in training)
    n_fail: torch.Tensor    # (B,) recourse events
    actions: torch.Tensor   # (B, T)


def index_state(state, idx):
    if isinstance(state, torch.Tensor):
        return state[idx]
    if isinstance(state, (tuple, list)):
        return type(state)(index_state(s, idx) for s in state)
    if isinstance(state, dict):
        return {k: index_state(v, idx) for k, v in state.items()}
    return state


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def rollout(policy, batch: Batch, delivery: str = "partial", weather_input: bool = True,
            decode: str = "greedy") -> Rollout:
    env = SVRPEnv(batch, delivery, weather_input)
    state = policy.init_state(env)
    logp_sum = torch.zeros(batch.size, device=batch.demand.device)
    actions = []
    for _ in range(env.max_steps()):
        if env.done.all():
            break
        logits, state = policy(env, state)
        logp = masked_log_probs(logits, env.mask())
        if decode == "greedy":
            a = logp.argmax(-1)
        elif decode == "sample":
            a = torch.multinomial(logp.exp(), 1).squeeze(1)
        else:
            raise ValueError(decode)
        logp_sum = logp_sum + logp.gather(1, a[:, None]).squeeze(1) * (~env.done)
        env.step(a)
        actions.append(a)
    assert env.done.all(), "episode did not finish within max_steps"
    return Rollout(env.total_cost, logp_sum, env.n_fail, torch.stack(actions, 1))


@torch.no_grad()
def best_of_n(policy, batch: Batch, n: int, delivery="partial", weather_input=True, chunk=4096):
    """HINDSIGHT: sample n rollouts per day on the real day and keep the cheapest."""
    costs = []
    for b0 in range(0, batch.size, max(1, chunk // n)):
        sub = batch.index(slice(b0, b0 + max(1, chunk // n))).repeat_interleave(n)
        r = rollout(policy, sub, delivery, weather_input, decode="sample")
        costs.append(r.cost.view(-1, n).min(1).values)
    return torch.cat(costs)


@torch.no_grad()
def beam_search(policy, batch: Batch, width: int, delivery="partial", weather_input=True):
    """Keeps the `width` most probable partial routes. Returns (cost of most-probable beam, cheapest beam)."""
    B, N1, W = batch.size, batch.n_nodes, width
    dev = batch.demand.device
    env = SVRPEnv(batch, delivery, weather_input).select(torch.arange(B, device=dev).repeat_interleave(W))
    state = policy.init_state(env)
    score = torch.full((B, W), float("-inf"), device=dev)
    score[:, 0] = 0.0                                       # start with a single live beam per day
    rows = torch.arange(B, device=dev)[:, None]
    for _ in range(env.max_steps()):
        if env.done.all():
            break
        logits, state = policy(env, state)
        logp = masked_log_probs(logits, env.mask())        # finished beams: depot has log p = 0
        cand = (score.view(-1, 1) + logp).view(B, W * N1)
        top, ti = cand.topk(W, dim=1)
        parent, action = ti // N1, ti % N1
        dead = ~torch.isfinite(top)                         # fewer than W legal continuations
        parent = torch.where(dead, parent[:, :1], parent)
        action = torch.where(dead, action[:, :1], action)
        flat = (rows * W + parent).view(-1)
        env, state = env.select(flat), index_state(state, flat)
        env.step(action.view(-1))
        score = top
    cost = env.total_cost.view(B, W)
    most_probable = cost[torch.arange(B, device=dev), score.argmax(1)]
    return most_probable, cost.min(1).values


@torch.no_grad()
def apriori(policy, batch: Batch, delivery="partial", weather_input=True, decode="greedy"):
    """DEPLOYABLE plan-then-drive: roll out in the *believed* world, then execute that plan for real."""
    r = rollout(policy, batch.believed(), delivery, weather_input, decode=decode)
    acts = r.actions.cpu().numpy()
    plans = [np.concatenate([[0], a]) for a in acts]
    cost, fails = execute_many(plans, batch.demand.cpu().double().numpy(), batch.cost.cpu().double().numpy(),
                               batch.capacity, delivery)
    return cost, fails
