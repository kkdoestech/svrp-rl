"""Train a policy.  Examples:

    uv run python scripts/train.py --n_customers 10 --epochs 1                      # smoke test
    uv run python scripts/train.py --n_customers 20 --batch_size 256 --time_limit_min 60
    uv run python scripts/train.py --n_customers 20 --baseline shared --n_shared 8

Every EnvConfig / TrainConfig field is a flag (see svrp/config.py).
"""
import argparse

from svrp.config import EnvConfig, TrainConfig, add_args, from_args
from svrp.train.reinforce import train

p = argparse.ArgumentParser()
add_args(p, EnvConfig)
add_args(p, TrainConfig)
a = p.parse_args()
train(from_args(a, EnvConfig), from_args(a, TrainConfig))
