"""The paper's policy network (part 5.md): Conv1d "summarizer" -> LSTM "memory" -> attention "focus" -> mask.

Per step t:
    e_i  = Conv1d_{k=1}( [w, d_i^t/Q, ĉ_i0..ĉ_iN] )            node embeddings (recomputed: demand changes)
    h_t  = LSTMCell( [e_{p_t}, q_t/Q], h_{t-1} )                truck memory: where am I, how full am I
    u_i  = vᵀ tanh(W_ref e_i + W_q h_t)                         how attractive is node i right now
    logits = 10 · tanh(u)                                       clipping keeps exploration alive
The mask (-inf on illegal nodes) is applied by the rollout code.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from svrp.env.svrp_env import SVRPEnv


class PaperNet(nn.Module):
    def __init__(self, n_nodes: int, n_weather: int = 3, dim: int = 128, clip: float = 10.0):
        super().__init__()
        self.clip = clip
        self.embed = nn.Conv1d(n_weather + 1 + n_nodes, dim, kernel_size=1)
        self.lstm = nn.LSTMCell(dim + 1, dim)
        self.W_ref = nn.Linear(dim, dim, bias=False)
        self.W_q = nn.Linear(dim, dim)
        self.v = nn.Parameter(torch.empty(dim).uniform_(-1 / math.sqrt(dim), 1 / math.sqrt(dim)))

    def init_state(self, env: SVRPEnv):
        B, D = env.pos.shape[0], self.lstm.hidden_size
        z = torch.zeros(B, D, device=env.pos.device)
        return (z, z.clone())

    def forward(self, env: SVRPEnv, state):
        x = env.node_features()                                     # (B, N+1, F)
        e = self.embed(x.transpose(1, 2)).transpose(1, 2)           # (B, N+1, D)
        e_pos = e[env.ar, env.pos]
        h, c = self.lstm(torch.cat([e_pos, (env.load / env.Q)[:, None]], -1), state)
        u = torch.tanh(self.W_ref(e) + self.W_q(h)[:, None, :]) @ self.v
        return self.clip * torch.tanh(u), (h, c)


class Critic(nn.Module):
    """Baseline b_φ: predicts the cost of the day from the initial observation (trained with MSE)."""

    def __init__(self, n_nodes: int, n_weather: int = 3, dim: int = 128):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(n_weather + 1 + n_nodes, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.head(self.node(node_features).mean(1)).squeeze(-1)
