"""Configuration dataclasses + a tiny helper that turns them into command-line flags.

Every experiment is fully described by an EnvConfig (the "world") and a TrainConfig (the "learner").
Both are saved next to every checkpoint, so any result can be traced back to its exact settings.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields


@dataclass
class EnvConfig:
    n_customers: int = 20
    fill_rate: float = 0.5           # Φ = E[ξ_i] / Q  ->  Q = mean demand / Φ
    a_ratio: float = 0.6             # A: share of the constant (base) part
    b_ratio: float = 0.2             # B: share explained by weather
    gamma_ratio: float = 0.2         # Γ: share that is pure noise
    stochastic_demand: bool = True
    stochastic_cost: bool = True
    positions: str = "fixed"         # fixed (same city every day, B2B) | flexible (new city every day, B2C)
    delivery: str = "partial"        # partial (split allowed) | full (indivisible orders)
    estimate: str = "knn"            # constant | knn | oracle_mean
    weather_input: bool = True       # does the network see w?
    knn_k: int = 10
    history_days: int = 1000         # history bank size for a fixed city
    history_days_flexible: int = 100 # per-instance history bank when positions are flexible
    n_weather: int = 3
    demand_low: float = 5.0
    demand_high: float = 15.0
    city_seed: int = 0
    # I4 (Part 7, limitation #2): a second external variable besides weather. On an "event" day
    # (concert, football match, ...) customers within event_radius of a fixed venue order
    # event_boost × θ̄_i extra. event_input says whether the event flag is visible to the agent
    # and to the k-NN lookup. event_prob = 0 switches the feature off (all earlier worlds unchanged).
    event_prob: float = 0.0
    event_boost: float = 1.0
    event_radius: float = 0.3
    event_input: bool = True

    @property
    def n_external(self) -> int:
        """Size of the observable external-variable vector: weather (+ the event flag)."""
        return self.n_weather + (1 if self.event_prob > 0 else 0)

    @property
    def mean_demand(self) -> float:
        return 0.5 * (self.demand_low + self.demand_high)

    @property
    def capacity(self) -> float:
        return self.mean_demand / self.fill_rate

    def tag(self) -> str:
        """Short unique name for this world, used for result/caching folders."""
        r = f"{self.a_ratio:g}-{self.b_ratio:g}-{self.gamma_ratio:g}"
        sto = f"d{int(self.stochastic_demand)}c{int(self.stochastic_cost)}"
        ev = ""
        if self.event_prob > 0:
            ev = f"_ev{self.event_prob:g}x{self.event_boost:g}r{self.event_radius:g}" + ("" if self.event_input else "h")
        return (f"n{self.n_customers}_phi{self.fill_rate:g}_r{r}_{sto}_{self.positions}_"
                f"{self.delivery}_{self.estimate}_k{self.knn_k}_s{self.city_seed}{ev}")


@dataclass
class TrainConfig:
    model: str = "paper"             # paper | graph
    baseline: str = "critic"         # critic | shared
    n_shared: int = 8                # M rollouts per instance for the shared baseline
    embed_dim: int = 128
    batch_size: int = 256
    epochs: int = 50
    epoch_size: int = 10240
    lr: float = 1e-4
    critic_lr: float = 1e-4
    grad_clip: float = 1.0
    val_size: int = 1000
    seed: int = 1
    device: str = "auto"             # auto | cuda | cpu
    time_limit_min: float = 60.0     # stop training after this many minutes (the 1-hour budget)
    run_name: str = ""


def _str2bool(v: str) -> bool:
    if v.lower() in ("1", "true", "yes", "y"):
        return True
    if v.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def add_args(parser: argparse.ArgumentParser, cls) -> None:
    """Add one --flag per dataclass field (e.g. --n_customers 10 --delivery full)."""
    for f in fields(cls):
        default = f.default
        kind = bool if isinstance(default, bool) else type(default)
        parser.add_argument(f"--{f.name}", type=_str2bool if kind is bool else kind, default=default)


def from_args(args: argparse.Namespace, cls):
    return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


def to_dict(cfg) -> dict:
    return asdict(cfg)
