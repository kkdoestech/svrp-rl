"""What the agent *believes* about today's demands and travel costs before it visits anybody.

constant     d̂ = θ̄,          ĉ = distance          (long-run averages; ignores the weather)
knn          inverse-distance-weighted mean of the k past days whose weather was most similar to today
oracle_mean  d̂ = E[ξ | w],   ĉ = E[c | w]          (perfect knowledge of the weather effect, no noise;
                                                     a ceiling for how much weather can possibly help)

The k-NN formula from the paper:  d̂_i = Σ_g (ξ_ig / ‖w_g − w‖) / Σ_g (1 / ‖w_g − w‖)
"""
from __future__ import annotations

import torch


def knn_estimate(
    w_query: torch.Tensor,        # (C', S, M)  today's weather, S days per city
    w_hist: torch.Tensor,         # (C, H, M)   weather of past days (C = C' or 1)
    values_hist: torch.Tensor,    # (C, H, D)   what was observed on those days (demands or flattened costs)
    k: int,
    chunk: int = 256,
) -> torch.Tensor:
    """Returns (C', S, D)."""
    Cq, S, _ = w_query.shape
    C, H, D = values_hist.shape
    k = min(k, H)
    out = torch.empty(Cq, S, D, dtype=values_hist.dtype, device=values_hist.device)
    cities = torch.arange(Cq, device=w_query.device)[:, None, None] if C > 1 else 0
    for s0 in range(0, S, chunk):                       # chunked so big N does not blow up memory
        wq = w_query[:, s0:s0 + chunk]
        d = torch.cdist(wq, w_hist.expand(Cq, H, -1))   # (C', s, H) weather distances
        dk, idx = d.topk(k, dim=-1, largest=False)
        wgt = 1.0 / (dk + 1e-6)
        wgt = wgt / wgt.sum(-1, keepdim=True)
        vals = values_hist[cities, idx]                 # (C', s, k, D)
        out[:, s0:s0 + chunk] = (wgt[..., None] * vals).sum(-2)
    return out
