"""GraphNet — improvement I2: a Transformer ("attention model", Kool et al. 2019) instead of Conv1d + LSTM.

Why it might be better (Part 7, limitation #3): a map is a graph, and attention lets every customer look
at every other customer. Why it can run on any N: inputs are per-node [x, y, d̂/Q, is_depot] plus edge
biases, never a length-N row, so a model trained on N=20 can be run on N=50.

Encoder (once per day):   tokens = N+1 nodes + 1 weather token
                          L × [ x = LN(x + MHA(x) with bias −β_h·ĉ_ij) ; x = LN(x + FF(x)) ]
Decoder (every step):     dynamic info [d_i^t/Q, revealed_i] shifts each node's key/value
                          context  = W[mean(nodes), weather, node at p_t, load/Q]
                          glimpse  = masked MHA(context → nodes)
                          logit_i  = 10·tanh( qᵀk_i/√D − β·ĉ_{p_t,i} )
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from svrp.env.svrp_env import SVRPEnv

SOFTPLUS_ONE = math.log(math.e - 1)          # softplus(SOFTPLUS_ONE) = 1


class MHA(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.o = nn.Linear(dim, dim)

    def split(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.h, D // self.h).transpose(1, 2)

    def forward(self, q, k, v, attn_bias=None):
        out = F.scaled_dot_product_attention(self.split(q), self.split(k), self.split(v), attn_mask=attn_bias)
        B, H, L, d = out.shape
        return self.o(out.transpose(1, 2).reshape(B, L, H * d))


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff: int = 512):
        super().__init__()
        self.mha = MHA(dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.ReLU(), nn.Linear(ff, dim))
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, x, bias):
        x = self.n1(x + self.mha(x, x, x, bias))
        return self.n2(x + self.ff(x))


class GraphNet(nn.Module):
    def __init__(self, n_weather: int = 3, dim: int = 128, heads: int = 8, layers: int = 3, clip: float = 10.0):
        super().__init__()
        self.dim, self.heads, self.clip = dim, heads, clip
        self.node_in = nn.Linear(4, dim)
        self.weather_in = nn.Linear(n_weather, dim)
        self.enc_beta = nn.Parameter(torch.full((heads,), SOFTPLUS_ONE))
        self.layers = nn.ModuleList(EncoderLayer(dim, heads) for _ in range(layers))
        self.ctx = nn.Linear(3 * dim + 1, dim)
        self.glimpse = MHA(dim, heads)
        self.dyn_k, self.dyn_v, self.dyn_l = (nn.Linear(2, dim) for _ in range(3))
        self.ptr_q, self.ptr_k = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.dec_beta = nn.Parameter(torch.tensor(SOFTPLUS_ONE))

    # -------------------------------------------------------------- encoder (static, once per day)
    def init_state(self, env: SVRPEnv):
        b = env.batch
        B, N1 = b.demand.shape
        is_depot = torch.zeros(B, N1, device=b.demand.device)
        is_depot[:, 0] = 1.0
        x = self.node_in(torch.stack([b.coords[..., 0], b.coords[..., 1], b.dem_hat / env.Q, is_depot], -1))
        w = b.w if env.weather_input else torch.zeros_like(b.w)
        x = torch.cat([x, self.weather_in(w)[:, None]], 1)                     # (B, N+2, D)
        bias = torch.zeros(B, self.heads, N1 + 1, N1 + 1, device=x.device)
        bias[:, :, :N1, :N1] = -F.softplus(self.enc_beta)[None, :, None, None] * b.cost_hat[:, None]
        for layer in self.layers:
            x = layer(x, bias)
        nodes, weather = x[:, :N1], x[:, N1]
        return {"nodes": nodes, "graph": torch.cat([nodes.mean(1), weather], -1),
                "K": self.glimpse.k(nodes), "V": self.glimpse.v(nodes), "L": self.ptr_k(nodes)}

    # -------------------------------------------------------------- decoder (every step)
    def forward(self, env: SVRPEnv, s):
        dyn = torch.stack([env.observed_demand() / env.Q, env.visited.float()], -1)   # (B, N+1, 2)
        K, V, Lk = s["K"] + self.dyn_k(dyn), s["V"] + self.dyn_v(dyn), s["L"] + self.dyn_l(dyn)
        here = s["nodes"][env.ar, env.pos]
        q = self.ctx(torch.cat([s["graph"], here, (env.load / env.Q)[:, None]], -1))[:, None]   # (B, 1, D)

        mask = env.mask()
        attn = torch.zeros(mask.shape, device=q.device).masked_fill(~mask, float("-inf"))[:, None, None, :]
        g = self._glimpse(q, K, V, attn)                                       # (B, 1, D)
        u = (self.ptr_q(g) @ Lk.transpose(1, 2)).squeeze(1) / math.sqrt(self.dim)
        u = u - F.softplus(self.dec_beta) * env.batch.cost_hat[env.ar, env.pos]
        return self.clip * torch.tanh(u), s

    def _glimpse(self, q, K, V, attn):
        m = self.glimpse
        out = F.scaled_dot_product_attention(m.split(m.q(q)), m.split(K), m.split(V), attn_mask=attn)
        B, H, L, d = out.shape
        return m.o(out.transpose(1, 2).reshape(B, L, H * d))
