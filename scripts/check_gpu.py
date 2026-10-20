"""Phase 0: is the GTX 1060 usable, and how fast is one training step? (sets batch sizes for the 1-hour budget)

    uv run python scripts/check_gpu.py
"""
import time

import torch

from svrp.config import EnvConfig, TrainConfig
from svrp.data import World, train_generator
from svrp.inference import rollout
from svrp.train.reinforce import make_policy

print(f"torch {torch.__version__}  CUDA build {torch.version.cuda}")
assert torch.cuda.is_available(), "CUDA not available -> train on CPU (--device cpu)"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {name}  sm_{cap[0]}{cap[1]}  {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
assert f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list(), "this PyTorch build has no kernels for your GPU"
x = torch.randn(2048, 2048, device="cuda")
torch.testing.assert_close((x @ x).cpu(), x.cpu() @ x.cpu(), rtol=1e-3, atol=1e-2)
print("GPU matmul OK")


def bench(n, batch_size, device, steps=5):
    env = EnvConfig(n_customers=n)
    tr = TrainConfig(batch_size=batch_size)
    world, gen = World(env), train_generator(0)
    policy = make_policy(tr, env).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    times, gen_times = [], []
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 1):
        t0 = time.perf_counter()
        b = world.sample(batch_size, gen).to(device)
        t1 = time.perf_counter()
        r = rollout(policy, b, decode="sample")
        loss = (r.cost.detach() * r.logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        if i:                                   # skip warm-up
            times.append(time.perf_counter() - t0)
            gen_times.append(t1 - t0)
    per_step = sum(times) / steps
    mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
    epoch_s = per_step * 10240 / batch_size
    print(f"  N={n:3d} batch={batch_size:4d} {device:4s}: {per_step*1000:7.0f} ms/step "
          f"(data {sum(gen_times)/steps*1000:4.0f} ms)  peak {mem:6.0f} MB  ->  {epoch_s:5.1f} s/epoch, "
          f"{3600/epoch_s:5.0f} epochs/hour")


print("\nPaperNet REINFORCE step benchmark (1 epoch = 10,240 days):")
for n, bs in [(10, 512), (20, 256), (20, 512), (50, 128)]:
    bench(n, bs, "cuda")
bench(10, 512, "cpu")
