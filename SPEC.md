# SVRP-RL — Project Spec

Goal: reproduce the core result of *Iklassov, Sobirov, Solozabal, Takáč — "Reinforcement Learning for
Solving Stochastic Vehicle Routing Problem"* (ACML 2023, [arXiv 2311.07708](https://arxiv.org/abs/2311.07708))
on a single GTX 1060 3GB, then try improvements and **keep only what measurably works**.

What "reproduce" means here: the paper leaves out hyperparameters, the noise covariance Σ, k for k-NN and
the truck capacity Q, and the authors' public code ([Zangir/SVRP](https://github.com/Zangir/SVRP)) differs
from the paper. So we do **not** chase the absolute numbers of Table 2. We reproduce the **relative**
claims: every method runs on the *same* scenarios from *our* generator, and we compare gaps.

---

## S1 — Generator (`svrp/env/generator.py`)

The paper's stochastic model (see `a.md`):

    θ_i = θ̄_i  +  Σ_m Σ_n α_imn · w_m · w_n  +  ε_i
          base     weather effect (learnable)    noise (unpredictable)

**City** (one per experiment when positions are `fixed`):
- Depot at (0.5, 0.5); N customers ~ U[0,1]².
- Mean demand θ̄_i ~ U[5, 15]. Mean travel cost θ̄_ij = Euclidean distance d_ij.
- Weather weights α_i ∈ R^{3×3} per customer, α_ij per edge (symmetric in i,j).
  **Drawn once per city** and kept fixed. This is the fix for the authors' code, which re-draws α
  every scenario and so turns the weather term into pure noise.
- α is scaled so the weather term has exactly the target variance share (below).

**Scenario** (= one day):
- Weather w ~ U(−1, 1)³ (temperature, pressure, humidity).
- Demand  ξ_i  = max(1, θ̄_i + S_i(w) + ε_i); under full delivery also ≤ Q.
- Cost    c_ij = max(0.1·d_ij, d_ij + S_ij(w) + ε_ij); symmetric, c_ii = 0.
- S(w) = wᵀαw − E[wᵀαw] is mean-centred, so E[ξ_i] ≈ θ̄_i and E[c_ij] ≈ d_ij.
  (Closed form, w ~ U(−1,1): E[S] = tr(a)/3, Var(S) = (4/45)Σ a_mm² + (4/9)Σ_{m<n} a_mn², a = sym(α).)

**Signal ratio** A / B / Γ (see `b.md`), default 0.6 / 0.2 / 0.2:
- Var(weather term) = (B/A)·θ̄², Var(ε) = (Γ/A)·θ̄²  ⇒  shares θ̄² : E[S²] : E[ε²] = A : B : Γ.

**Toggles**: `stochastic_demand`, `stochastic_cost`, `positions = fixed|flexible`, `delivery = partial|full`.

**Fill rate** Φ = E[ξ_i] / Q, so Q = 10 / Φ; default Φ = 0.5 → Q = 20 (≈ 2 customers per trip).
The paper prints Φ = Q/E[ξ_i]. Taken literally, Φ = 0.5 would mean a truck cannot carry even one
customer's order. Table 2's cost also grows linearly with N, which fits our reading.

## S2 — Estimates (`svrp/env/estimates.py`): what the agent believes before visiting

| name          | demand estimate d̂_i                     | cost estimate ĉ_ij          |
|---------------|------------------------------------------|-----------------------------|
| `constant`    | θ̄_i (long-run average, ignores weather)  | d_ij                        |
| `knn`         | inverse-distance-weighted mean of the k=10 historical days with the closest weather | same, per edge |
| `oracle_mean` | true E[ξ_i \| w] (no noise) — diagnostic ceiling for "how much can weather help" | true E[c_ij \| w] |

History bank: 1000 past days per fixed city; 100 per instance when positions are flexible.
`weather_input=False` additionally hides w from the network.

## S3 — Environment / MDP (`svrp/env/svrp_env.py`), batched torch, 1 vehicle

- **Observation per node i**: [w (3), d_i^t / Q, ĉ_i0 … ĉ_iN].
  d_i^t = estimate until the first visit; after that, the true remaining demand (revealed on arrival).
- **Vehicle**: current node p_t, load q_t / Q.
- **Action**: index of the next node (0 = depot).
- **Cost** of a step: the *realized* c_{p_t, a}.
- **Partial delivery**: deliver min(load, remaining). If the truck empties first, the customer stays open
  and the agent must come back later. Depot refills to Q.
- **Full delivery**: an order is indivisible. Arriving with load < demand triggers automatic recourse:
  round trip c_j0 + c_0j, refill, deliver.
- **Mask**: customers with nothing left are masked; with load = 0 only the depot is allowed;
  depot→depot is not allowed until everything is served.
- **End**: all demand served and the truck is back at the depot.
- `n_fail` counts arrivals where load < true remaining demand (= recourse events).

## S4 — Executor (`svrp/env/executor.py`)

Runs an **a-priori plan** (a node sequence such as `[0, 3, 5, 0, 2, 0]`) on a realized scenario:
- Arrival with too little load → detour-to-depot recourse (round trip), then finish the customer.
- Load hits 0 mid-route → return to depot before the next planned customer.
- Customers that are already fully served are skipped.

## S5 — Baselines (`svrp/baselines/`) — all scored by S3/S4 on the same scenarios

| method        | kind                      | notes |
|---------------|---------------------------|-------|
| NN-online     | reoptimizing heuristic    | nearest feasible customer by ĉ; the floor RL must beat |
| Clarke–Wright | a priori                  | savings on estimates |
| OR-Tools GLS  | a priori                  | CVRP on estimates, Guided Local Search |
| OR-Tools TS   | a priori                  | same, `TABU_SEARCH` (the paper's TS) |
| ACO           | a priori                  | own numpy Ant System + elitist ant; params tuned on validation |
| Hindsight     | reference (not deployable)| OR-Tools on the *realized* demand and cost |

Planning demands are clipped to Q (the executor handles the rest through recourse). With `constant`
estimates and fixed positions, a planner solves once and reuses the plan for every scenario.

## S6 — Models (`svrp/models/`)

- **PaperNet** (faithful): Conv1d(k=1) node embedding (D=128, recomputed each step) →
  LSTMCell over [e_{p_t}, q_t/Q] → Bahdanau pointer attention, 10·tanh logit clipping, S3 mask.
  Critic: Conv1d embedding → mean-pool → MLP → predicted cost.
- **GraphNet** (improvement, attention-model style): 3-layer Transformer encoder (8 heads, D=128) over
  [x, y, d̂/Q, is_depot] + a weather token; per-step dynamic demand embedding; decoder context
  [graph, current node, load] → masked glimpse → pointer + learned edge bias −β·ĉ_{p_t,i}.
  It does not depend on N, so we can test generalization to larger N.

## S7 — Training (`svrp/train/reinforce.py`)

- REINFORCE (see `part 5.md`): ∇J ≈ mean[(C − b) · ∇ Σ_t log π(a_t)].
- Adam lr 1e-4, grad-norm clip 1.0, fp32 (Pascal fp16 is very slow, so no AMP).
- Scenarios generated on the fly; 1 epoch = 10,240 scenarios (≈ the paper's 10k).
- Baseline b: `critic` (paper; MSE-trained) or `shared` (M samples per instance, b = their mean cost).
- Greedy validation on 1000 fixed scenarios every epoch; keep the best checkpoint.

## S8 — Inference (`svrp/inference.py`)

Inference modes are labeled by whether a real dispatcher could actually run them:

| mode                       | deployable? | why |
|----------------------------|-------------|-----|
| greedy (online)            | yes         | only uses information revealed so far |
| a-priori greedy / beam     | yes         | plans on the estimates, then executes with recourse |
| beam n_b (online)          | **lookahead** | the beams "visit" customers on the realized day, so they peek at demand you cannot know yet |
| best-of-n sampling         | **hindsight** | picks the rollout with the best *realized* cost |

The paper's headline RL number uses beam search. We report it for comparison, labeled.

## S9 — Evaluation (`scripts/evaluate.py`)

- Separate deterministic seeds for city, history bank, validation (1000) and test (1000).
- Paired per-scenario costs are saved for every method, so all comparisons are paired.
- Metrics: mean cost ± 95% bootstrap CI, gap % vs OR-Tools GLS and vs hindsight, recourse events,
  ms/instance, training wall-clock.
- 3 training seeds for headline results, 1–2 for exploratory runs.

## S10 — Keep-or-reject rule

A change is **KEPT** if the mean validation cost across seeds improves by more than 2 standard errors,
at no larger training budget. Every experiment is logged in `experiments/LOG.md` and
`results/results.csv`, kept or not. The test set is only used at checkpoints.

---

## Deviations from the paper (and why)

| paper                                  | us                                   | why |
|----------------------------------------|--------------------------------------|-----|
| Φ = Q/E[ξ]                             | Φ = E[ξ]/Q                           | literal reading is infeasible |
| α in authors' code re-drawn per day    | α fixed per city                     | otherwise weather carries no learnable signal |
| noise ε ~ N(0, Σ), Σ unspecified       | independent ε                         | Σ not given |
| K vehicles                             | K = 1                                | turn order across vehicles not specified |
| A100 40GB                              | GTX 1060 3GB, ≤1h per run            | hardware; N ≤ 20 headline, N = 50 stretch |
| beam-search RL as headline             | greedy headline, beam labeled lookahead | beam on the realized day peeks at hidden demand |
| 10k fixed training scenarios           | on-the-fly scenarios                 | free, avoids overfitting |
