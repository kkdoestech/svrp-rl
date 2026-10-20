# 13 — Improvements: shared baseline, a Transformer policy, and a new external variable

| | |
|---|---|
| **Layer** | 11 |
| **Files covered** | `svrp/models/graph_net.py` (105 lines); the shared-baseline branch of `svrp/train/reinforce.py`; the event parts of `svrp/config.py`, `svrp/env/generator.py`, `svrp/data.py`; `experiments/queue_2.sh` |
| **You build** | `mysvrp/models/graph_net.py`, the event feature, and your own improvement experiments |
| **Time** | 5–8 hours |
| **Before this** | [11](11_training_reinforce.md), [12](12_evaluation_and_experiments.md); re-read the paper's Part 7 limitations (in `part 6.md`, under "Section B") |

## Why this layer exists

The paper's own conclusion lists what it couldn't do:

1. it only goes up to 100 customers;
2. weather is the only external variable, though traffic or events could be added;
3. the architecture is older (LSTM + basic attention), while Transformers and graph networks are now standard.

This lesson builds one improvement for each, plus a training trick:

| id | idea | answers limitation | where |
|---|---|---|---|
| **I1** | shared (POMO-style) baseline instead of the critic | — (training quality) | `reinforce.py` shared branch |
| **I2** | GraphNet: a Transformer encoder with travel-cost-aware attention, size-agnostic | #3, and #1 (any N) | `models/graph_net.py` |
| **I4** | a second external variable: local *events* that boost demand near a venue | #2 | config, generator, data |

(I3, "more inference compute", is just evaluation modes: `--rl_modes beam10,best128`.)

Each is tested **one change at a time** with the keep-or-reject rule from lesson 12. The results appear
in `experiments/LOG.md` as E5 (I1), E6 (I2) and E7 (I4) once their runs finish.

---

## Part A — I1: the shared baseline (POMO-style)

You already wrote it in lesson 11 (lines 109–112 of `reinforce.py`):

<!-- quote: svrp/train/reinforce.py -->
```python
                else:
                    c = r.cost.view(-1, M)
                    adv = (c - c.mean(1, keepdim=True)).view(-1)
                    critic_loss = torch.zeros((), device=device)
```

**The idea** (Kwon et al. 2020, "POMO"): the critic is a *guess* of a day's cost. If you drive the same
day 8 times with 8 sampled routes, their average **is** a very good "expected cost of this day". No guess
and no second network. Each route is then judged against its 7 siblings on the *same* day, so day-to-day
difficulty cancels out perfectly.

**The trade-off:** a batch of 512 now holds only 64 *different* days (× 8 tries each). The model sees less
variety per step, but every comparison is cleaner. Whether that's better is an empirical question.

**The experiment** (`experiments/queue_1.sh`): `E5_n10_paper_shared_s{1,2,3}` vs `E2_n10_critic_s{1,2,3}`.
Same world, budget and learning rate; only `--baseline` differs. Three seeds each.

---

## Part B — I2: GraphNet, a Transformer for routing

### Concepts from zero

**Self-attention.** Every node (a "token") asks every other node: *how relevant are you to me?*

Each token's vector x is turned into three vectors with learned linear maps:

- a **query** q ("what I'm looking for"),
- a **key** k ("what I contain"),
- a **value** v ("what I'll tell you").

Then:

```
score(i, j) = q_i · k_j / √d         (scaled dot product: how well i's question matches j's answer)
weights     = softmax over j          (who to listen to)
new x_i     = Σ_j weights(i, j) · v_j (a weighted mix of the others' values)
```

After this, each node's vector contains information about the *other* nodes: "I'm a customer in a
cluster of three high-demand customers near the depot." PaperNet's Conv1d summarised each node *alone*.

**Multi-head.** Do it 8 times in parallel with smaller vectors (128 / 8 = 16 numbers each). Each head can
learn a different notion of relevance (nearby nodes, big orders, …). Then concatenate the heads.

**Attention bias = making it graph-aware.** Add a number to every score before the softmax:
score(i, j) − β·ĉ_ij. Nodes that are expensive to reach get lower attention. β is learned, one per head.
This is how the *estimated travel costs*, the stochastic part, enter the Transformer.

**Residual + LayerNorm + feed-forward.** Each encoder layer is

```
x = LayerNorm(x + Attention(x))      # "add what you learned from the others" (residual), then normalise
x = LayerNorm(x + FeedForward(x))    # a small 2-layer network applied to each node separately
```

The residual "+ x" makes deep stacks trainable (the signal can always pass straight through).
LayerNorm keeps the numbers in a stable range.

**Encoder once, decoder every step.** The expensive part (3 layers of all-pairs attention) only uses
information known in the morning, so it runs **once per day**. Each step then runs only a cheap decoder:
one query ("I'm at node p with this load") attending over the encoded nodes.

**Why size-agnostic?** No layer's size depends on N. Inputs are 4 numbers per node, and attention works
for any number of tokens. A GraphNet trained on N = 20 can be *run* on N = 50, which is impossible for
PaperNet (lesson 09).

### Code walkthrough

#### Lines 1–22: the header

<!-- quote: svrp/models/graph_net.py -->
```python
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
```

The docstring is the whole design on one screen:

- why it might help (limitation #3 from the paper);
- why it can run at any N (the inputs never contain a length-N row);
- the encoder and decoder as formulas, which the code follows line by line.

The imports match PaperNet's, plus `F` for `softplus` and `scaled_dot_product_attention`.

#### Lines 24–41: a constant and a multi-head attention block

<!-- quote: svrp/models/graph_net.py -->
```python
SOFTPLUS_ONE = math.log(math.e - 1)          # softplus(SOFTPLUS_ONE) = 1
```

β must stay **positive** (a far node should get *less* attention, never more). So the code stores a raw
number and uses softplus(raw) = log(1 + e^raw), which is always > 0. Starting the raw value at
log(e − 1) makes softplus = exactly 1.0: "start with a bias of −1 × cost, then learn".

<!-- quote: svrp/models/graph_net.py -->
```python
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
```

- Three projections q, k, v (bias-free), and an output projection o.
- `split`: (B, L, 128) → (B, L, 8, 16) → (B, 8, L, 16), i.e. one 16-number slice per head, heads moved
  forward so each head is processed in parallel.
- `F.scaled_dot_product_attention`: PyTorch's built-in, fast version of the three formulas above. Its
  `attn_mask` argument, when it holds floats, is **added to the scores**. That's how both the cost bias and
  the mask (−∞ for illegal nodes) get in.
- Merge the heads back, (B, 8, L, 16) → (B, L, 128) (`reshape`, because the data isn't contiguous after
  `transpose`, lesson 02 section 11), then the output projection.

#### Lines 44–53: one encoder layer

<!-- quote: svrp/models/graph_net.py -->
```python
class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff: int = 512):
        super().__init__()
        self.mha = MHA(dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.ReLU(), nn.Linear(ff, dim))
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, x, bias):
        x = self.n1(x + self.mha(x, x, x, bias))
        return self.n2(x + self.ff(x))
```

The two formulas above, literally. `self.mha(x, x, x, …)` means queries, keys and values all come from
the same tokens: **self**-attention. The feed-forward widens to 512 and back to 128.

#### Lines 56–68: the parts of GraphNet

<!-- quote: svrp/models/graph_net.py -->
```python
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
```

| part | role |
|---|---|
| `node_in` | 4 numbers per node → 128 |
| `weather_in` | the weather vector → one extra 128-dim **weather token** |
| `enc_beta` | 8 cost-bias strengths, one per head |
| `layers` | 3 encoder layers. `nn.ModuleList` (not a plain list!) so PyTorch finds their parameters |
| `ctx` | builds the decoder's query from [graph summary (256), current node (128), load (1)] |
| `glimpse` | the decoder's attention over the nodes |
| `dyn_k/v/l` | turn each step's *dynamic* info (2 numbers per node) into adjustments of keys/values/pointer keys |
| `ptr_q/ptr_k` | the final scoring ("pointer") |
| `dec_beta` | the cost-bias strength in the final scores |

743,817 parameters in total, ~4.4× PaperNet.

#### Lines 70–85: the encoder, once per day

<!-- quote: svrp/models/graph_net.py -->
```python
    # -------------------------------------------------------------- encoder (static, once per day)
    def init_state(self, env: SVRPEnv):
        b = env.batch
        B, N1 = b.demand.shape
        is_depot = torch.zeros(B, N1, device=b.demand.device)
        is_depot[:, 0] = 1.0
        x = self.node_in(torch.stack([b.coords[..., 0], b.coords[..., 1], b.dem_hat / env.Q, is_depot], -1))
        w = b.w if env.weather_input else torch.zeros_like(b.w)
        x = torch.cat([x, self.weather_in(w)[:, None]], 1)                     # (B, N+2, D)
```

- Node input = [x, y, estimated demand / Q, is-this-the-depot]. The depot flag matters: without it the
  depot would look like a customer with demand 0.
- The weather becomes one extra token appended at position N1, so the tokens are (B, N1 + 1, 128).
  Every customer can attend to "the weather" like to any other node.

<!-- quote: svrp/models/graph_net.py -->
```python
        bias = torch.zeros(B, self.heads, N1 + 1, N1 + 1, device=x.device)
        bias[:, :, :N1, :N1] = -F.softplus(self.enc_beta)[None, :, None, None] * b.cost_hat[:, None]
        for layer in self.layers:
            x = layer(x, bias)
        nodes, weather = x[:, :N1], x[:, N1]
        return {"nodes": nodes, "graph": torch.cat([nodes.mean(1), weather], -1),
                "K": self.glimpse.k(nodes), "V": self.glimpse.v(nodes), "L": self.ptr_k(nodes)}
```

- The bias tensor has one (N1+1) × (N1+1) matrix per head. The node-to-node block is −β_h · ĉ_ij. The
  weather token's row and column stay 0 (it has no travel cost).
- Run the 3 layers. Split the output into node vectors and the weather vector.
- The returned state is a **dict**:
  - the node vectors;
  - a "graph" summary (the mean of all nodes + the weather token, 256 numbers);
  - and the keys/values/pointer keys, **precomputed once** (they're static), so each step only adds the
    dynamic adjustments.

#### Lines 87–99: the decoder, every step

<!-- quote: svrp/models/graph_net.py -->
```python
    # -------------------------------------------------------------- decoder (every step)
    def forward(self, env: SVRPEnv, s):
        dyn = torch.stack([env.observed_demand() / env.Q, env.visited.float()], -1)   # (B, N+1, 2)
        K, V, Lk = s["K"] + self.dyn_k(dyn), s["V"] + self.dyn_v(dyn), s["L"] + self.dyn_l(dyn)
        here = s["nodes"][env.ar, env.pos]
        q = self.ctx(torch.cat([s["graph"], here, (env.load / env.Q)[:, None]], -1))[:, None]   # (B, 1, D)
```

- **Dynamic info** per node: the current observed demand (revealed or estimated, lesson 08) and whether
  it's been revealed.
- Why *add* `dyn_k(dyn)` to the keys instead of re-running the encoder? Re-encoding every step would cost
  3 attention layers × ~30 steps. Adding a learned linear function of the 2 dynamic numbers is cheap, and
  enough to tell the decoder "this node's demand changed". (It's a cheap alternative: some published
  "dynamic attention" models instead re-run the whole encoder after each trip, which is more expressive
  but slower.)
- `here`: the encoded vector of each truck's current node.
- The query `q` = [graph summary, current node, load] → 128 numbers, shape (B, 1, 128): one "question" per day.

<!-- quote: svrp/models/graph_net.py -->
```python
        mask = env.mask()
        attn = torch.zeros(mask.shape, device=q.device).masked_fill(~mask, float("-inf"))[:, None, None, :]
        g = self._glimpse(q, K, V, attn)                                       # (B, 1, D)
        u = (self.ptr_q(g) @ Lk.transpose(1, 2)).squeeze(1) / math.sqrt(self.dim)
        u = u - F.softplus(self.dec_beta) * env.batch.cost_hat[env.ar, env.pos]
        return self.clip * torch.tanh(u), s
```

- **Glimpse:** the query attends over the nodes, but only legal ones (the mask becomes −∞ in the
  attention, shape (B, 1, 1, N1) to broadcast over heads and the one query). The result `g` is a
  context-aware summary: "given where I am, here's what matters among the nodes I can still go to".
- **Pointer:** compare `g` with every node's pointer key: scaled dot product → one score per node.
- **Cost bias again:** subtract β·(estimated cost from here to each node). Nearby nodes start out
  favoured, and the network learns how much to trust that.
- Clip with 10·tanh, as in PaperNet. The state `s` is returned unchanged, because the decoder doesn't
  carry memory between steps. Its memory is the env itself: positions, loads and revealed demands.

#### Lines 101–105

<!-- quote: svrp/models/graph_net.py -->
```python
    def _glimpse(self, q, K, V, attn):
        m = self.glimpse
        out = F.scaled_dot_product_attention(m.split(m.q(q)), m.split(K), m.split(V), attn_mask=attn)
        B, H, L, d = out.shape
        return m.o(out.transpose(1, 2).reshape(B, L, H * d))
```

`_glimpse` is MHA's `forward`, but with precomputed K and V (they already include the dynamic part), so only
the query is projected (`m.q(q)`). Then it splits heads, runs the attention with the mask, merges heads and
applies the output projection, exactly like `MHA.forward`.

Two signatures you've now seen in the widened quotes: `GraphNet(n_weather=3, dim=128, heads=8, layers=3,
clip=10.0)` (the defaults are the configuration that was trained), and `init_state` / `forward`, the same two
methods as PaperNet. That shared interface is why `rollout` and `beam_search` work with either network
unchanged.

---

## Part C — I4: a new external variable (local events)

**The idea:** some days have a concert or a football match. Customers near the venue order much more. A
dispatcher knows from the calendar whether today is an event day, so it's an *observable* variable, like
the weather. Does telling the model about it help?

The design rule for adding it: **with `event_prob = 0` (the default), every earlier world must stay
bit-identical**, so no old result changes. Each piece:

**Config** (`svrp/config.py`, lines 36–44): `event_prob`, `event_boost`, `event_radius`, `event_input`,
plus `n_external` = 3 weather numbers + 1 flag when events are on (lesson 03).

**The venue** (`svrp/env/generator.py`, lines 95–99), drawn **last** in `make_cities`, so that worlds without
events consume exactly the same random numbers as before:

<!-- quote: svrp/env/generator.py -->
```python
    near_venue = torch.zeros(n, N + 1)
    if cfg.event_prob > 0:              # drawn last, so worlds without events keep the exact same RNG stream
        venue = torch.rand(n, 1, 2, generator=gen)
        near_venue = ((coords - venue).norm(dim=-1) < cfg.event_radius).float()
        near_venue[:, 0] = 0.0
```

**Event days** (lines 109–113), with *no random draw at all* when events are off:

<!-- quote: svrp/env/generator.py -->
```python
def sample_events(shape: tuple[int, ...], cfg: EnvConfig, gen: torch.Generator) -> torch.Tensor | None:
    """1.0 on event days. None (and no random draw at all) when events are switched off."""
    if cfg.event_prob <= 0:
        return None
    return (torch.rand(*shape, generator=gen) < cfg.event_prob).float()
```

**The effect on demand** (lines 150–152), an extra +boost·θ̄ for customers near the venue on event days.
It is added to *both* the true demand and the noise-free mean, because it is known information, not noise:

<!-- quote: svrp/env/generator.py -->
```python
    if event is not None:               # a known (observable) demand shock, so it is part of E[ξ | w, event]
        boost = event[..., None] * cfg.event_boost * theta * city.near_venue[:, None, :]
        mean_dem, demand = mean_dem + boost, demand + boost
```

**What the agent observes** (`svrp/data.py`, lines 81–90):

<!-- quote: svrp/data.py -->
```python
    def _external(self, w: torch.Tensor, event: torch.Tensor | None):
        """(what the agent observes, what the k-NN lookup matches on). Without events both are just w.
        A hidden event flag is observed as 0 and ignored by the k-NN lookup."""
        if event is None:
            return w, w
        flag = EVENT_SCALE * event[..., None]
        if self.cfg.event_input:
            ext = torch.cat([w, flag], -1)
            return ext, ext
        return torch.cat([w, torch.zeros_like(flag)], -1), w
```

- **Visible** (`event_input=True`): the agent sees [weather, 2.0 × flag], and k-NN matches on it too.
  `EVENT_SCALE = 2.0` makes event days "far" from normal days in k-NN distance (weather differences are
  ~0.4), so an event day's neighbours are other event days.
- **Hidden** (`event_input=False`): the flag is observed as 0 (same input *size*, so the network shape
  doesn't change), and k-NN ignores it. **The true days are identical in both cases**. Only the
  information differs, so the comparison is clean.

**The experiment** (`experiments/queue_2.sh`): `E7_n10_events_seen_s{1,2}` vs `E7_n10_events_hidden_s{1,2}`,
with event_prob 0.3 and boost 1.5. Expectation: seeing the flag should help clearly, because event days are
big, predictable shocks.

---

## Rebuild it yourself

1. **I1 first.** It's tiny (you already have it). Train 3 seeds of each baseline at N = 10, 15 min each, and
   compare their validation costs with rule S10. Log it as your E5.
2. **I2:** write `MHA`, test it on random tensors (output shape = input shape), then `EncoderLayer`, then
   `GraphNet`. Add the `"graph"` branch to `make_policy`. Smoke-train with `--model graph --epochs 3`.
3. **I4:** add the config fields → `make_cities` (venue last!) → `sample_events` → the boost in `realize` →
   `_external` + `EVENT_SCALE` in `data.py` (and use `key` in `_history`/`sample`). Then check your lesson 05
   fingerprint still matches exactly. That's the "nothing old changed" test.

## Verify

```bash
uv run python -m pytest tests/test_models.py -q                # now including the GraphNet test
uv run python -m pytest tests/test_generator.py -q             # now including the 2 event tests
```

- `test_graphnet_runs_on_any_size_and_beam1_is_greedy`: the same weights run on N = 6 and N = 12.
- `test_events_boost_demand_near_the_venue_only`: deterministic demand + events, so near-venue demand is
  exactly 2 × θ̄ on event days, and nothing changes elsewhere.
- `test_hidden_event_flag_changes_only_the_belief`: same true days, and k-NN is better when the flag is seen.

The size-agnostic check, from `learning/workspace`:

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
import torch
from mysvrp.config import EnvConfig
from mysvrp.data import World
from mysvrp.env.svrp_env import SVRPEnv
from mysvrp.models.graph_net import GraphNet
torch.manual_seed(0)
net = GraphNet(3, 128)
print('params', sum(p.numel() for p in net.parameters()))
for n in (6, 12, 50):
    env = SVRPEnv(World(EnvConfig(n_customers=n, positions='flexible')).eval_set('test', 4))
    s = net.init_state(env); logits, s = net(env, s)
    print(n, tuple(s['nodes'].shape), tuple(logits.shape))
"
```

Expected: `params 743817`, then `6 (4, 7, 128) (4, 7)`, `12 (4, 13, 128) (4, 13)`, `50 (4, 51, 128) (4, 51)`.

## Common mistakes

- **A plain Python list of layers** instead of `nn.ModuleList`: the layers' parameters are invisible to the
  optimiser and to `.to(device)`, and the error appears as "tensors on different devices".
- **`view` after `transpose`** when merging heads: a runtime error. Use `reshape`.
- **A boolean mask where a float bias is expected**: for `scaled_dot_product_attention`, a *bool* mask means
  "True = allowed", while a *float* mask is added. Mixing them up silently inverts the mask.
- **Drawing the venue before the other city randoms**: every old world changes, and the fingerprint no
  longer matches.
- **Hidden events implemented as a *smaller* input**: then the two arms have different network sizes, and
  the experiment changes two things at once.

## Check your understanding

1. Why does the GraphNet encoder run once per day, while PaperNet's Conv1d runs every step?
2. Why is β passed through softplus?
3. What makes GraphNet size-agnostic, in one sentence?
4. In the I4 experiment, why must the hidden flag be shown as 0 rather than removed?
5. Before running E5 (shared vs critic), write down your *hypothesis*. What do you expect, and why?

**Answers** (try the questions first, then check here)

1. GraphNet's encoder only uses morning information (positions, estimates, weather), and the changing
   parts enter through the cheap dynamic adjustments. PaperNet's input *includes* the changing demand, so
   it must be recomputed.
2. To keep it positive: expensive-to-reach nodes should get less attention, never more.
3. Every layer acts per node or per pair of nodes, with weights that don't depend on how many nodes there are.
4. So both arms have identical networks and identical days, and the *only* difference is the information.
5. (Your answer.) One reasonable hypothesis: the shared baseline gives cleaner advantages (same-day
   comparison) and learns faster per minute; the critic sees 8× more distinct days per step. The
   experiment decides.

## Read more

- Vaswani et al. (2017), "Attention Is All You Need" (arXiv 1706.03762), sections 3.1–3.3.
- Jay Alammar, "The Illustrated Transformer": [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
- Kool, van Hoof & Welling (2019), "Attention, Learn to Solve Routing Problems!" (arXiv 1803.08475). GraphNet
  follows its encoder/decoder design.
- Kwon et al. (2020), "POMO: Policy Optimization with Multiple Optima for RL" (arXiv 2010.16011): the
  shared baseline.

**Next (optional):** [14 — ACO](14_aco_optional.md), the paper's "champion" classical method.
