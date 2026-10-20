# 09 — The policy network: Conv1d → LSTM → attention (`svrp/models/paper_net.py`)

| | |
|---|---|
| **Layer** | 7 |
| **Files covered** | `svrp/models/paper_net.py` (53 lines) |
| **You build** | `mysvrp/models/paper_net.py` |
| **Time** | 3–4 hours |
| **Before this** | [08 — Environment](08_environment.md); re-read `part 5.md` part 2 ("The AI's brain") |

## Why this layer exists

`nn_online` (lesson 08) is a policy written by hand: "nearest customer who fits". It can't get any
smarter. The paper replaces it with a **neural network**: a function with ~170,000 adjustable numbers
(*parameters*, the "knobs" in `part 5.md`). It takes the observation and outputs a score for every node.
Training (lesson 11) turns the knobs until good routes become likely.

This lesson is only the network's *shape*: how numbers flow from the observation to the scores. It is
the four steps from your `part 5.md` notes, in 22 lines of code.

## Concepts from zero

### A neuron, a layer, an activation

A **linear layer** `nn.Linear(n_in, n_out)` computes y = W·x + b: each of the n_out outputs is a weighted
sum of the n_in inputs, plus a bias. W (n_out × n_in) and b (n_out) are the parameters. An **activation**
bends the result, so that stacked layers can represent curved, complicated functions:

- `tanh(x)` squashes any number into (−1, 1).
- `ReLU(x) = max(0, x)` keeps positives and zeroes negatives.

Without activations, ten linear layers would collapse into one linear layer.

### Embeddings: a summary vector per node

The raw input for node i is 4 + N1 numbers: [3 weather, demand/Q, the cost row]. An **embedding** maps it
to D = 128 numbers that the network learns to make useful. That is the "neat summary" from your notes.
The network decides by itself what the 128 numbers mean.

### Conv1d with kernel size 1 = the same Linear for every node

A 1-D convolution slides a small window along a sequence. With `kernel_size=1`, the window is one
element wide, so it applies **the same linear layer to each node independently**. It's identical to
`nn.Linear` applied on the last dimension. The paper calls it a "1D convolutional layer". It just means
*one shared summariser for all customers*. Why shared? A customer at node 3 and a customer at node 7
should be summarised by the same rules. And it works for any number of nodes.

### LSTM: memory of the trip so far

A **recurrent** network keeps a hidden state h that is updated at every step: h_t = f(input_t, h_{t−1}).
An **LSTM** (Long Short-Term Memory) is a recurrent cell with *gates*, little learned switches that
decide what to forget, what to write, and what to output. It's good at keeping information over many
steps. `nn.LSTMCell` does **one step**. It carries two vectors, h (the output memory) and c (the internal
"cell" memory).

Here the input at each step is "the summary of the node I'm at + how full my truck is". So h_t can
remember the journey: "I've been in the north, my truck is half empty."

### Attention (Bahdanau style): how relevant is each node right now?

For each node i, compute a **compatibility score** between "this node's summary" e_i and "my current
memory" h_t:

```
u_i = vᵀ · tanh(W_ref·e_i + W_q·h_t)
```

Read it as: transform both into the same 128-dim space, add them, squash, then reduce to one number with
a learned direction v. A high u_i means "given where I am and what I carry, node i looks like a good next
stop". The scores are called **logits**. Softmax turns logits into probabilities (lesson 02 section 6),
after the mask sets illegal nodes to −∞.

### Logit clipping: 10 · tanh(u)

If a logit grows to 50 while the others are 0, softmax gives probability ≈ 1 to that node, and training
stops exploring alternatives. `10·tanh(u)` keeps logits in (−10, 10), so every legal node keeps a small
chance. (This trick comes from Bello et al. 2016, "Neural Combinatorial Optimization with RL", and is
standard in routing networks.)

## Code walkthrough

### Lines 1–17: the docstring is the whole model

<!-- quote: svrp/models/paper_net.py -->
```python
"""The paper's policy network (part 5.md): Conv1d "summarizer" -> LSTM "memory" -> attention "focus" -> mask.

Per step t:
    e_i  = Conv1d_{k=1}( [w, d_i^t/Q, ĉ_i0..ĉ_iN] )            node embeddings (recomputed: demand changes)
    h_t  = LSTMCell( [e_{p_t}, q_t/Q], h_{t-1} )                truck memory: where am I, how full am I
    u_i  = vᵀ tanh(W_ref e_i + W_q h_t)                         how attractive is node i right now
    logits = 10 · tanh(u)                                       clipping keeps exploration alive
The mask (-inf on illegal nodes) is applied by the rollout code.
"""
```

These four formulas *are* `part 5.md`'s four steps. The code below is just these lines in PyTorch.

<!-- quote: svrp/models/paper_net.py -->
```python
from __future__ import annotations

import math

import torch
from torch import nn

from svrp.env.svrp_env import SVRPEnv
```

`math` is for `math.sqrt` in the initialisation. `torch.nn` (imported as `nn`) holds every layer type.
`SVRPEnv` is imported only for the type hints: the network reads its observation straight from the env.

### Lines 20–28: the layers (the knobs)

<!-- quote: svrp/models/paper_net.py -->
```python
class PaperNet(nn.Module):
    def __init__(self, n_nodes: int, n_weather: int = 3, dim: int = 128, clip: float = 10.0):
        super().__init__()
        self.clip = clip
        self.embed = nn.Conv1d(n_weather + 1 + n_nodes, dim, kernel_size=1)
        self.lstm = nn.LSTMCell(dim + 1, dim)
        self.W_ref = nn.Linear(dim, dim, bias=False)
        self.W_q = nn.Linear(dim, dim)
        self.v = nn.Parameter(torch.empty(dim).uniform_(-1 / math.sqrt(dim), 1 / math.sqrt(dim)))
```

- `nn.Module` is PyTorch's base class for networks. `super().__init__()` must come first: it sets up the
  machinery that finds all parameters inside (so that `.to(device)`, `.parameters()` and `state_dict()`
  work).
- `embed`: input channels = 3 + 1 + N1 (the per-node features from `node_features()`), output = 128.
  **The input size depends on N**: a PaperNet built for N = 10 can't run on N = 20. That's a limitation
  of the paper's design. Lesson 13's GraphNet removes it.
- `lstm`: input = 128 (the node summary) + 1 (load / Q). Hidden size 128.
- `W_ref`, `W_q`, `v`: the attention's three parameters. `nn.Parameter(...)` registers a raw tensor as a
  knob. It's initialised uniformly in ±1/√128, the same scale PyTorch uses for Linear layers.

Parameter count for N = 5 (n_nodes = 6), as printed by PyTorch:

| name | shape | why |
|---|---|---|
| `embed.weight` | (128, 10, 1) | 128 outputs × 10 input channels × kernel 1 |
| `lstm.weight_ih` | (512, 129) | 4 gates × 128 = 512 rows; input 129 |
| `lstm.weight_hh` | (512, 128) | the gates also look at the previous h |
| `W_ref.weight`, `W_q.weight` | (128, 128) each | |
| `v` | (128,) | |
| total | **167,040** | (about 167,680 for N = 10, because `embed` has more inputs) |

### Lines 30–33: the memory starts empty

<!-- quote: svrp/models/paper_net.py -->
```python
    def init_state(self, env: SVRPEnv):
        B, D = env.pos.shape[0], self.lstm.hidden_size
        z = torch.zeros(B, D, device=env.pos.device)
        return (z, z.clone())
```

At the start of a day, h and c are zeros, one pair per day in the batch. Policies return their state as
a tuple; the rollout passes it back in at each step. (The second `z.clone()` gives c its own memory. The
LSTM doesn't modify inputs in place, but separate tensors are the safe habit.)

### Lines 35–41: one decision

<!-- quote: svrp/models/paper_net.py -->
```python
    def forward(self, env: SVRPEnv, state):
        x = env.node_features()                                     # (B, N+1, F)
        e = self.embed(x.transpose(1, 2)).transpose(1, 2)           # (B, N+1, D)
        e_pos = e[env.ar, env.pos]
        h, c = self.lstm(torch.cat([e_pos, (env.load / env.Q)[:, None]], -1), state)
        u = torch.tanh(self.W_ref(e) + self.W_q(h)[:, None, :]) @ self.v
        return self.clip * torch.tanh(u), (h, c)
```

Calling `net(env, state)` runs `forward`. Line by line, with shapes for B days, N1 nodes, D = 128:

1. **Observe.** `x = env.node_features()`, shape (B, N1, 4 + N1). Called every step, because demands
   change as the truck delivers and reveals. (That's why the comment says "recomputed".)
2. **Summarise.** Conv1d expects (batch, channels, length), so transpose to (B, F, N1), convolve to
   (B, D, N1), and transpose back to (B, N1, D). Now `e[b, i]` is node i's 128-number summary on day b.
3. **Where am I?** `e_pos = e[env.ar, env.pos]`, shape (B, D): each truck's current node summary.
4. **Remember.** The LSTM input is [e_pos, load/Q], shape (B, 129). It returns the new (h, c), each (B, D).
5. **Score every node.** `W_ref(e)` is (B, N1, D). `W_q(h)[:, None, :]` is (B, 1, D), broadcast over the N1
   nodes. Add them, apply tanh, then `@ self.v` turns each node's 128 numbers into one score, so `u` is
   (B, N1).
6. **Clip and return** the logits (B, N1) and the new state.

Notice what is *not* here: no mask and no softmax. The network only scores. The rollout applies the
rules (lesson 10). Keeping "brain" and "rules" separate means the same network works under partial or
full delivery.

### Lines 44–53: the Critic (the "class average" from `part 5.md`)

<!-- quote: svrp/models/paper_net.py -->
```python
class Critic(nn.Module):
    """Baseline b_φ: predicts the cost of the day from the initial observation (trained with MSE)."""

    def __init__(self, n_nodes: int, n_weather: int = 3, dim: int = 128):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(n_weather + 1 + n_nodes, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.head(self.node(node_features).mean(1)).squeeze(-1)
```

The baseline b_φ from `part 5.md` is a *separate* small network that looks at the morning observation and
**guesses how expensive the day will be**. Lesson 11 explains why REINFORCE needs that guess.

- `nn.Sequential(...)` chains layers: Linear → ReLU → Linear → ReLU.
- `self.node(features)`: each node gets a 128-dim summary, shape (B, N1, D). (`nn.Linear` acts on the last
  dimension, so it's the same per-node idea as Conv1d with kernel 1.)
- `.mean(1)`: average over all nodes, giving one summary of the *whole day*, (B, D). Averaging (*mean
  pooling*) doesn't care about the order of the nodes.
- `head`: (B, D) → (B, 1), and `.squeeze(-1)` gives (B,), one predicted cost per day.

## Rebuild it yourself

1. Write `PaperNet.__init__` and `init_state`.
2. Write `forward`, printing each shape the first time. Compare with the shapes listed above.
3. Write `Critic`.

## Verify

From `learning/workspace` (the env and data from earlier lessons must be done):

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
import torch
from mysvrp.config import EnvConfig
from mysvrp.data import World
from mysvrp.env.svrp_env import SVRPEnv
from mysvrp.models.paper_net import PaperNet, Critic
torch.manual_seed(0)
b = World(EnvConfig(n_customers=5)).eval_set('test', 2)
env = SVRPEnv(b)
net = PaperNet(n_nodes=6, n_weather=3, dim=128)
print('params', sum(p.numel() for p in net.parameters()))
x = env.node_features(); print('features', tuple(x.shape))
logits, state = net(env, net.init_state(env))
torch.set_printoptions(precision=3, sci_mode=False)
print('logits', logits)
p = torch.log_softmax(logits.masked_fill(~env.mask(), float('-inf')), -1).exp()
print('probs ', p)
print('critic', Critic(6, 3, 128)(x))
"
```

Reference output (identical if you create the layers in the same order, since `manual_seed(0)` then gives
the same initial knobs):

```
params 167040
features (2, 6, 10)
logits tensor([[ 0.750, -1.546,  2.062,  1.271,  0.332, -0.060],
        [ 1.163, -2.480,  2.469,  0.808,  0.033, -0.393]],
       grad_fn=<MulBackward0>)
probs  tensor([[0.000, 0.015, 0.562, 0.255, 0.100, 0.067],
        [0.000, 0.005, 0.745, 0.142, 0.065, 0.043]], grad_fn=<ExpBackward0>)
critic tensor([0.078, 0.082], grad_fn=<SqueezeBackward1>)
```

What to notice:

- The depot's probability is exactly 0: at the start the truck is at the depot, and the mask forbids
  depot → depot.
- The untrained network already has preferences (customer 2 is at 56%), but they're random: the knobs
  are random.
- `grad_fn=...` means PyTorch remembers how these numbers were computed, so it can later compute
  gradients (lesson 11).
- The untrained critic guesses a cost of ~0.08 for a day that really costs ~5. It will learn.

If your parameter count differs, compare the table above row by row.

## Common mistakes

- **Forgetting the transposes around Conv1d**: you convolve over the wrong axis, and a shape error
  usually appears in the LSTM.
- **`W_q(h)` without `[:, None, :]`**: (B, D) + (B, N1, D) broadcasts only by luck when B = N1, which
  gives silently wrong results.
- **Applying softmax inside the network**: then the mask must be applied before it, inside the network.
  Keep logits raw, and do mask + softmax in the rollout.
- **Creating layers in `forward`**: their parameters are new every call and are never trained. All
  layers belong in `__init__`.

## Check your understanding

1. Why does the network recompute the node embeddings at every step instead of once per day?
2. Why can't a PaperNet trained on N = 10 be run on N = 20? Which line is responsible?
3. What does mean-pooling in the Critic make it independent of?
4. What would happen to exploration without `10 · tanh`?

**Answers** (try the questions first, then check here)

1. The input includes each node's *current observed demand*, which changes after visits (revealed or
   reduced). Old summaries would describe a world that no longer exists.
2. `nn.Conv1d(n_weather + 1 + n_nodes, ...)`: the cost row has N1 entries, so the input size is tied to N.
3. The order (numbering) of the nodes, and, for the pooled summary itself, their count.
4. Logits could grow huge, softmax would put ~100% on one node, and the sampled routes would stop varying.
   REINFORCE then gets no signal about alternatives, and learning stalls.

## Read more

- 3Blue1Brown, *Neural networks* (YouTube), chapters 1–3: what a layer and a gradient are, visually.
- Christopher Olah, "Understanding LSTM Networks": [colah.github.io/posts/2015-08-Understanding-LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Bahdanau, Cho & Bengio (2014), "Neural Machine Translation by Jointly Learning to Align and Translate"
  (arXiv 1409.0473): where this attention formula comes from.
- Vinyals, Fortunato & Jaitly (2015), "Pointer Networks" (arXiv 1506.03134): attention used to *point at an
  input* (a node), exactly what happens here.
- Nazari et al. (2018), arXiv 1802.04240, figure 2: the VRP network this one is based on.
- PyTorch tutorial, *Build the Neural Network*: [pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)

**Next:** [10 — Rollouts and inference](10_rollout_and_inference.md)
