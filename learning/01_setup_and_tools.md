# 01 — Setup and tools: `pyproject.toml`, `uv`, the GPU, git

| | |
|---|---|
| **Layer** | 0 |
| **Files covered** | `pyproject.toml`, `.gitignore`, `scripts/check_gpu.py` (lines 1–22) |
| **You build** | your workspace folder + your own `check_gpu.py` |
| **Time** | 1–2 hours |
| **Before this** | [00 — Prerequisites](00_prerequisites.md) |

## Why this layer exists

Before any routing or AI, three things must be true:

1. Python can find every library the project needs, **at versions that work together**.
2. PyTorch can actually run on *your* graphics card (this was the trickiest part of the whole setup).
3. Your experiments are reproducible: anyone (including future you) can rebuild the same environment.

One file, `pyproject.toml`, handles all three, and one tool, `uv`, reads it.

## Concepts from zero

**Package manager.** Libraries like PyTorch and OR-Tools are downloaded from the internet. A package
manager downloads them and puts them somewhere Python can find. `pip` is the classic one. `uv` is a
newer, much faster one that also reads `pyproject.toml` and makes a **lock file** (`uv.lock`) listing
the *exact* version of every library, so the setup can be repeated exactly.

**Virtual environment (`.venv`).** A private folder of libraries just for this project. Without it, all
projects on your PC share one set of libraries, and upgrading one project can break another. `uv sync`
creates `.venv/` and fills it. `uv run <command>` runs the command *inside* that environment.

**GPU, CUDA, and "compute capability".** A GPU runs thousands of small calculations in parallel. PyTorch
talks to NVIDIA GPUs through **CUDA**. Each GPU generation has an architecture number called *compute
capability*:

| your card | architecture | compute capability | PyTorch calls it |
|---|---|---|---|
| GTX 1060 3GB | Pascal (2016) | 6.1 | `sm_61` |

PyTorch ships pre-compiled GPU code (**kernels**) for a list of architectures. Newer PyTorch builds
(CUDA 12.8 and later) **dropped Pascal**. With those builds, your GPU fails with "no kernel image is
available". The CUDA **12.6** builds still include `sm_61`. And PyTorch announced that 2.14 is the last
version with CUDA 12.6 builds. So the project pins: *torch from the cu126 download index, version below
2.15.* That is the whole story behind lines 7–9 and 26–32 of `pyproject.toml`.

**git.** Version control: it records snapshots ("commits") of your files, so you can always go back.
`.gitignore` lists files git should *not* track (big, generated or private ones).

## Code walkthrough

### `pyproject.toml`: the project's recipe

Open `pyproject.toml` (the whole file is 36 lines).

<!-- quote: pyproject.toml -->
```toml
[project]
name = "svrp-rl"
version = "0.1.0"
description = "Reproduce and improve RL for the Stochastic Vehicle Routing Problem (Iklassov et al., ACML 2023)"
requires-python = ">=3.13,<3.14"
dependencies = [
    # GTX 1060 = Pascal (sm_61). Only the CUDA 12.6 wheels still support it,
    # and PyTorch stops publishing those from 2.15 -> pin below 2.15.
    "torch>=2.6,<2.15",
    "numpy>=2.0",
    "ortools>=9.12",
    "matplotlib>=3.9",
    "tqdm>=4.66",
]
```

- Lines 1–4: the project's name, version and description (metadata).
- Line 5, `requires-python = ">=3.13,<3.14"`: only Python 3.13.x. Your PC has 3.13, and pinning it avoids
  surprises from a different Python.
- Lines 6–14, `dependencies`: the libraries the code imports.
  - `torch>=2.6,<2.15`: **at least 2.6** (the first with Python 3.13 support) and **below 2.15** (the last
    version that still has Pascal builds).
  - `numpy`: arrays on the CPU (used by the classical solvers and the executor).
  - `ortools`: Google's solver, which you already know.
  - `matplotlib`: plots. `tqdm`: progress bars (installed but barely used).
- The `#` comment lines are the most important lines in the file: they explain *why* the pin exists. A
  pin without a reason gets "upgraded" by someone later, and the GPU breaks.

<!-- quote: pyproject.toml -->
```toml
[dependency-groups]
dev = ["pytest>=8"]
```

Libraries needed only by *developers* (here: `pytest`, the test runner), not by people who just run the
code. `uv sync` installs them by default.

<!-- quote: pyproject.toml -->
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["svrp"]
```

This makes the folder `svrp/` an **installable package**. `uv sync` installs it into `.venv` in
"editable" mode: Python imports straight from your `svrp/` folder, so code edits take effect without
reinstalling. That's why `scripts/train.py` can write `from svrp.config import ...` from inside the
`scripts/` folder. (Your `mysvrp` will *not* be installed; that's why you use `python -m` from the
workspace, as the [README](README.md) explains.)

<!-- quote: pyproject.toml -->
```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu126" }]
```

- `[[tool.uv.index]]` adds a second place to download from: PyTorch's own server, the **cu126** folder
  (builds compiled with CUDA 12.6, which still contain `sm_61` kernels).
- `explicit = true`: use this index *only* for packages that ask for it, not for everything.
- `[tool.uv.sources] torch = ...`: "torch, specifically, comes from that index". Without this, `uv` would
  take torch from PyPI, which on Windows may give a CPU-only or a cu128 build.

<!-- quote: pyproject.toml -->
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = ["ignore:builtin type .* has no __module__ attribute:DeprecationWarning"]
```

Settings for `pytest`: look for tests in `tests/`, and hide one harmless warning that OR-Tools'
internals (written in C++ and wrapped by "SWIG") print on Python 3.13.

### `.gitignore`: what git should not track

<!-- quote: .gitignore -->
```
.venv/
__pycache__/
.pytest_cache/
*.pyc
data/
runs/
*.pt
*.log
```

- `.venv/`: gigabytes of libraries; anyone can recreate it with `uv sync`.
- `__pycache__/`, `*.pyc`, `.pytest_cache/`: Python's and pytest's automatic caches.
- `runs/`, `*.pt`: training runs and model checkpoints (`.pt` = PyTorch weights); big and regenerable.
- `*.log`: console logs. `data/`: reserved for generated data.

`results/` is **not** ignored: the small result tables and plots are worth keeping in git.

### `scripts/check_gpu.py`, lines 1–22: is the GPU usable?

<!-- quote: scripts/check_gpu.py -->
```python
"""Phase 0: is the GTX 1060 usable, and how fast is one training step? (sets batch sizes for the 1-hour budget)

    uv run python scripts/check_gpu.py
"""
import time

import torch

from svrp.config import EnvConfig, TrainConfig
from svrp.data import World, train_generator
from svrp.inference import rollout
from svrp.train.reinforce import make_policy
```

- The docstring (lines 1–4) says what the script is for and how to run it. Every script in the project
  starts this way.
- Lines 9–12 import parts of the project you haven't built yet. Your version will skip them for now (see
  "Rebuild" below).

<!-- quote: scripts/check_gpu.py -->
```python
print(f"torch {torch.__version__}  CUDA build {torch.version.cuda}")
assert torch.cuda.is_available(), "CUDA not available -> train on CPU (--device cpu)"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {name}  sm_{cap[0]}{cap[1]}  {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
assert f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list(), "this PyTorch build has no kernels for your GPU"
x = torch.randn(2048, 2048, device="cuda")
torch.testing.assert_close((x @ x).cpu(), x.cpu() @ x.cpu(), rtol=1e-3, atol=1e-2)
print("GPU matmul OK")
```

Line by line:

- **14**: print the torch version (expect `2.14.1+cu126`) and the CUDA version it was *built* with (`12.6`).
- **15**: `torch.cuda.is_available()` is True if a driver and a CUDA-capable GPU are found. `assert` stops
  the script with the message if not.
- **16–17**: ask the GPU for its name and compute capability. For you, `cap = (6, 1)`.
- **18**: print the name, `sm_61`, and the memory in GB (bytes ÷ 2³⁰). Expect 3.0 GB.
- **19**: **the key check.** `torch.cuda.get_arch_list()` lists the architectures this PyTorch build has
  kernels for. If `sm_61` is missing, every GPU operation would crash later with a confusing error. This
  line turns that into a clear message *now*.
- **20**: create a 2048×2048 matrix of random normal numbers **directly on the GPU** (`device="cuda"`).
- **21**: multiply it by itself on the GPU (`x @ x`), copy the result to the CPU (`.cpu()`), and compare it
  with the same product computed on the CPU. `assert_close` allows tiny rounding differences (`rtol`,
  `atol`), because GPUs add numbers in a different order. If this passes, the GPU computes correctly.
- **22**: celebrate.

Lines 25–58 (`bench`) time one full training step at several sizes. They use layers you'll build in
lessons 05–11, so they're explained line by line in [lesson 16](16_helper_scripts.md). The results are in
`experiments/LOG.md`, entry E0: batch 512 at N = 20 takes ~0.2 s per step and under 1 GB of GPU memory.

## Rebuild it yourself

1. **Make sure the environment exists.** In the project root: `uv sync` (fast, since it's already done).
2. **Check git.** `git status` shows what changed since the last commit. Commit what's there before you
   start, so your learning work sits on a clean history:
   ```bash
   git add -A
   git commit -m "Project before starting the learning course"
   ```
3. **Create your workspace:**
   ```bash
   mkdir -p learning/workspace/mysvrp/env learning/workspace/mysvrp/baselines \
            learning/workspace/mysvrp/models learning/workspace/mysvrp/train \
            learning/workspace/tests learning/workspace/scripts
   touch learning/workspace/mysvrp/__init__.py learning/workspace/mysvrp/env/__init__.py \
         learning/workspace/mysvrp/baselines/__init__.py learning/workspace/mysvrp/models/__init__.py \
         learning/workspace/mysvrp/train/__init__.py
   ```
4. **Write `learning/workspace/scripts/check_gpu.py` from memory**, with only the part from lines 5–22:
   imports `torch` only, then the version print, the two asserts, and the matmul check. (Leave out the
   `svrp` imports for now.)
5. Run it:
   ```bash
   cd learning/workspace
   uv run python -m scripts.check_gpu
   ```

## Verify

Expected output:

```
torch 2.14.1+cu126  CUDA build 12.6
GPU: NVIDIA GeForce GTX 1060 3GB  sm_61  3.0 GB
GPU matmul OK
```

## Common mistakes

- **`ModuleNotFoundError: No module named 'torch'`**: you ran `python` instead of `uv run python`, so
  Python looked outside the `.venv`.
- **`ModuleNotFoundError: No module named 'scripts'`**: you are not inside `learning/workspace`, or you
  ran `python scripts/check_gpu.py` instead of `python -m scripts.check_gpu`.
- **The asserts fail on another PC**: that PC may have no NVIDIA GPU. Every training command accepts
  `--device cpu`, which is slower (~3× at N = 10) but works.
- **Someone upgrades torch to 2.15+**: the `sm_61` assert fails. Read the comment in `pyproject.toml`.

## Check your understanding

1. What is the difference between `pyproject.toml` and `uv.lock`?
2. Why is torch downloaded from a separate index instead of PyPI?
3. Why does line 21 compare the GPU result with a CPU result instead of checking exact equality?
4. Why is `runs/` in `.gitignore` but `results/` is not?

**Answers** (try the questions first, then check here)

1. `pyproject.toml` states *ranges* ("torch below 2.15"). `uv.lock` records the *exact* versions that
   were actually installed, so the same setup can be rebuilt exactly.
2. Only PyTorch's own index offers the CUDA 12.6 builds, which still contain kernels for the GTX 1060's
   `sm_61` architecture.
3. GPU and CPU add floating-point numbers in a different order, which changes the last digits. A tolerance
   checks "correct up to rounding".
4. `runs/` holds big, regenerable checkpoints. `results/` holds small tables and plots that document
   what was found, which you want in history.


## Read more

- `uv` documentation: [docs.astral.sh/uv](https://docs.astral.sh/uv/), see "Projects" and "Using uv with PyTorch".
- PyTorch's notice about CUDA 12.6 wheels ending (on dev-discuss.pytorch.org): search "CUDA 12.6 wheels
  will no longer be published from PyTorch 2.15".
- *Pro Git*, chapters 1–2 (free at git-scm.com/book): the basics of commits.

**Next:** [02 — Tensors crash course](02_tensors.md)
