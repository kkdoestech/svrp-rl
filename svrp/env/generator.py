"""Stochastic world generator: the equation from a.md, turned into code.

    θ = θ̄  +  wᵀ α w  −  E[wᵀ α w]  +  ε
        base   weather effect (centred)   noise

θ is either a customer's demand ξ_i (θ̄_i ~ U[5,15]) or the travel cost of an edge c_ij (θ̄_ij = distance).

Shapes (C = number of cities, P = (N+1)² edges, K = M² weather products, node 0 = depot):
    coords (C, N+1, 2)      theta (C, N+1)      dist (C, N+1, N+1)
    alpha_dem (C, N+1, K)   alpha_cost (C, P, K)
    shift_* / sigma_*       same shape as the quantity they belong to

Everything is sampled with an explicit torch.Generator on the CPU, so a seed always gives the same world.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from svrp.config import EnvConfig

DEPOT_XY = 0.5

# For w ~ U(-1, 1):  E[w²] = 1/3,  Var(w²) = E[w⁴] − E[w²]² = 1/5 − 1/9 = 4/45,  E[w_m² w_n²] = 1/9 (m≠n)
_E_W2 = 1.0 / 3.0
_VAR_W2 = 4.0 / 45.0
_E_W2W2 = 1.0 / 9.0


def weather_moments(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and variance of S = wᵀ a w for symmetric a (..., M, M) and independent w_m ~ U(-1, 1).

    S = Σ_m a_mm w_m² + 2 Σ_{m<n} a_mn w_m w_n, and all those terms are uncorrelated, so
        E[S]   = Σ_m a_mm / 3
        Var(S) = (4/45) Σ_m a_mm² + 4·(1/9) Σ_{m<n} a_mn²
    """
    diag = torch.diagonal(a, dim1=-2, dim2=-1)
    mean = diag.sum(-1) * _E_W2
    off = torch.triu(a, diagonal=1)
    var = _VAR_W2 * (diag ** 2).sum(-1) + 4.0 * _E_W2W2 * (off ** 2).sum((-2, -1))
    return mean, var


@dataclass
class City:
    coords: torch.Tensor
    theta: torch.Tensor
    dist: torch.Tensor
    alpha_dem: torch.Tensor
    shift_dem: torch.Tensor
    sigma_dem: torch.Tensor
    alpha_cost: torch.Tensor
    shift_cost: torch.Tensor
    sigma_cost: torch.Tensor
    near_venue: torch.Tensor    # (C, N+1) 1.0 for customers close to the event venue (all 0 if events are off)

    @property
    def n_cities(self) -> int:
        return self.coords.shape[0]

    def index(self, idx) -> "City":
        return City(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})


def _scaled_alpha(raw: torch.Tensor, base: torch.Tensor, b_over_a: float):
    """Symmetrise raw weights and scale them so Var(weather term) = (B/A)·base²."""
    a = 0.5 * (raw + raw.transpose(-1, -2))
    _, var = weather_moments(a)
    target = b_over_a * base ** 2
    scale = torch.sqrt(target / var.clamp_min(1e-12))
    a = a * scale[..., None, None]
    mean, _ = weather_moments(a)
    return a.flatten(-2), mean          # flattened (…, K) weights and the E[wᵀaw] to subtract


def make_cities(cfg: EnvConfig, n: int, gen: torch.Generator) -> City:
    N, M = cfg.n_customers, cfg.n_weather
    coords = torch.rand(n, N + 1, 2, generator=gen)
    coords[:, 0] = DEPOT_XY
    theta = cfg.demand_low + (cfg.demand_high - cfg.demand_low) * torch.rand(n, N + 1, generator=gen)
    theta[:, 0] = 0.0
    dist = torch.cdist(coords, coords)

    b_over_a = cfg.b_ratio / cfg.a_ratio
    noise_scale = (cfg.gamma_ratio / cfg.a_ratio) ** 0.5

    raw_dem = torch.randn(n, N + 1, M, M, generator=gen)
    alpha_dem, shift_dem = _scaled_alpha(raw_dem, theta, b_over_a)

    raw_cost = torch.randn(n, N + 1, N + 1, M, M, generator=gen)
    raw_cost = 0.5 * (raw_cost + raw_cost.transpose(1, 2))        # same weather effect for i→j and j→i
    alpha_cost, shift_cost = _scaled_alpha(raw_cost, dist, b_over_a)

    near_venue = torch.zeros(n, N + 1)
    if cfg.event_prob > 0:              # drawn last, so worlds without events keep the exact same RNG stream
        venue = torch.rand(n, 1, 2, generator=gen)
        near_venue = ((coords - venue).norm(dim=-1) < cfg.event_radius).float()
        near_venue[:, 0] = 0.0

    return City(
        coords=coords, theta=theta, dist=dist,
        alpha_dem=alpha_dem, shift_dem=shift_dem, sigma_dem=noise_scale * theta,
        alpha_cost=alpha_cost.flatten(1, 2), shift_cost=shift_cost, sigma_cost=noise_scale * dist,
        near_venue=near_venue,
    )


def sample_events(shape: tuple[int, ...], cfg: EnvConfig, gen: torch.Generator) -> torch.Tensor | None:
    """1.0 on event days. None (and no random draw at all) when events are switched off."""
    if cfg.event_prob <= 0:
        return None
    return (torch.rand(*shape, generator=gen) < cfg.event_prob).float()


def sample_weather(shape: tuple[int, ...], n_weather: int, gen: torch.Generator) -> torch.Tensor:
    return 2.0 * torch.rand(*shape, n_weather, generator=gen) - 1.0


def _weather_products(w: torch.Tensor) -> torch.Tensor:
    return (w[..., :, None] * w[..., None, :]).flatten(-2)      # (..., M²): all w_m · w_n


def realize(city: City, w: torch.Tensor, cfg: EnvConfig, gen: torch.Generator,
            event: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Draw the actual demands and travel costs for given weather (and event flags, shape (C', H)).

    w has shape (C', H, M): H days for each of C' cities. `city` has C = C' cities, or C = 1
    (one city shared by every day). Returns tensors of shape (C', H, N+1) and (C', H, N+1, N+1):
        demand, cost       the realized values (what actually happens)
        mean_dem, mean_cost   the noise-free values E[· | w] (used by the oracle_mean estimate)
    """
    Cq, H, _ = w.shape
    N1 = city.theta.shape[1]
    ww = _weather_products(w)                                              # (C', H, K)

    weather_dem = ww @ city.alpha_dem.transpose(-1, -2) - city.shift_dem[:, None, :]        # (C', H, N+1)
    weather_cost = (ww @ city.alpha_cost.transpose(-1, -2)).view(Cq, H, N1, N1) - city.shift_cost[:, None]

    eps_dem = torch.randn(Cq, H, N1, generator=gen) * city.sigma_dem[:, None, :]
    eps_cost = torch.triu(torch.randn(Cq, H, N1, N1, generator=gen), diagonal=1)
    eps_cost = (eps_cost + eps_cost.transpose(-1, -2)) * city.sigma_cost[:, None]

    theta = city.theta[:, None, :]
    dist = city.dist[:, None]
    if cfg.stochastic_demand:
        mean_dem, demand = theta + weather_dem, theta + weather_dem + eps_dem
    else:
        mean_dem = demand = theta.expand(Cq, H, N1)
    if event is not None:               # a known (observable) demand shock, so it is part of E[ξ | w, event]
        boost = event[..., None] * cfg.event_boost * theta * city.near_venue[:, None, :]
        mean_dem, demand = mean_dem + boost, demand + boost
    if cfg.stochastic_cost:
        mean_cost, cost = dist + weather_cost, dist + weather_cost + eps_cost
    else:
        mean_cost = cost = dist.expand(Cq, H, N1, N1)

    return {
        "demand": clip_demand(demand, cfg), "mean_dem": clip_demand(mean_dem, cfg),
        "cost": clip_cost(cost, dist), "mean_cost": clip_cost(mean_cost, dist),
    }


def clip_demand(d: torch.Tensor, cfg: EnvConfig) -> torch.Tensor:
    """Customers order at least 1 unit; under full delivery an order must fit in one truck."""
    hi = cfg.capacity if cfg.delivery == "full" else float("inf")
    d = d.clamp(1.0, hi).clone()
    d[..., 0] = 0.0                     # the depot has no demand
    return d


def clip_cost(c: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """A trip never costs less than 10% of its distance (diagonal stays exactly 0)."""
    return torch.maximum(c, 0.1 * dist)
