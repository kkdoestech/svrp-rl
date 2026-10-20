# Experiment log (lab notebook)

One entry per experiment: **hypothesis → setup → result → decision**. Kept *and* rejected ideas both stay here.
Numbers are mean realized travel cost over the 1000 test (or validation) days; lower is better.
All methods in a table were scored on the *same* days, so differences are paired.

Default world unless stated (SPEC.md): fixed city, partial delivery, k-NN estimates (k=10),
A/B/Γ = 0.6/0.2/0.2, Φ = 0.5 (Q = 20, ≈ 2 customers per trip), 1 truck, stochastic demand **and** cost.

---

## E0 — Setup (Phase 0)

- torch 2.14.1+cu126 (the last CUDA-12.6 line that still ships Pascal kernels), OR-Tools 9.15, Python 3.13.
- `scripts/check_gpu.py`: GTX 1060 3GB, sm_61 is in the build's arch list, GPU matmul OK.
- Benchmark, PaperNet REINFORCE step (sample a batch, roll out, backprop):

| N  | batch | device | ms/step | peak GPU MB | s/epoch (10,240 days) | epochs/hour |
|----|-------|--------|---------|-------------|------------------------|-------------|
| 10 | 512   | cuda   | 92      | 336         | 1.8                    | ~1960       |
| 20 | 256   | cuda   | 170     | 459         | 6.8                    | ~530        |
| 20 | 512   | cuda   | 195     | 899         | 3.9                    | ~920        |
| 50 | 128   | cuda   | 396     | 1309        | 31.7                   | ~110        |
| 10 | 512   | cpu    | 274     | –           | 5.5                    | ~660        |

**What this tells us:** a step costs almost the same at batch 256 and 512. The time goes into the
Python loop over route steps, not into the arithmetic. So we use **batch 512 for N ≤ 20**: it doubles
the data per second for free. N = 50 fits in 3 GB at batch 128. The GPU is ~3× faster than the CPU.

- ACO tuned on 100 validation days (N = 20). The paper's four (unlabelled) settings lost to the textbook
  α=1, β=2, 20 ants (15.75 vs 16.18–16.63), so that setting is used everywhere (`results/aco_params.json`).

---

## E1 — Classical baselines (Phase 2, **Checkpoint 1**) — test, 1000 days

Gap = paired per-day difference vs OR-Tools GLS, 95% bootstrap CI. "fails" = recourse events per day.

**N = 10**

| method        | mean cost | gap vs GLS [95% CI]     | vs hindsight | fails | ms/day |
|---------------|-----------|-------------------------|--------------|-------|--------|
| NN-online     | 8.028     | +0.10% [−1.19, +1.43]   | +33.0%       | 2.19  | 0.2    |
| Clarke–Wright | 8.001     | −0.23% [−0.97, +0.53]   | +32.6%       | 2.04  | 0.1    |
| OR-Tools GLS  | 8.020     | —                       | +32.9%       | 2.09  | 502    |
| OR-Tools Tabu | 8.146     | +1.57% [+0.93, +2.22]   | +34.9%       | 2.11  | 32     |
| ACO (tuned)   | 8.102     | +1.03% [+0.26, +1.81]   | +34.2%       | 2.13  | 57     |
| *Hindsight*   | *6.036*   | *−24.7%*                | —            | 1.21  | 501    |

**N = 20**

| method        | mean cost | gap vs GLS [95% CI]     | vs hindsight | fails | ms/day |
|---------------|-----------|-------------------------|--------------|-------|--------|
| NN-online     | 14.597    | +0.47% [−0.46, +1.43]   | +36.8%       | 4.76  | 0.3    |
| Clarke–Wright | 14.615    | +0.59% [−0.01, +1.17]   | +37.0%       | 4.45  | 0.3    |
| OR-Tools GLS  | 14.529    | —                       | +36.2%       | 4.67  | 1001   |
| OR-Tools Tabu | 14.790    | +1.79% [+1.28, +2.30]   | +38.6%       | 4.69  | 106    |
| ACO (tuned)   | 15.006    | +3.28% [+2.65, +3.89]   | +40.6%       | 4.72  | 100    |
| *Hindsight*   | *10.672*  | *−26.6%*                | —            | 2.80  | 1001   |

**What this tells us**
- **The classical methods are nearly tied.** With Φ = 0.5 a trip serves only about 2 customers, so the
  routing puzzle is easy; better optimization barely matters. CW (0.1 ms) ties OR-Tools (1 s).
- **Uncertainty is the real enemy.** Every a-priori plan pays about 2 (N=10) or 4.7 (N=20) recourse
  trips per day and ends up **33–37% above hindsight**. That gap is the cost of not knowing the day in
  advance, and it is the room an adaptive policy (RL) can win back.
- Ranking differs from the paper: there, ACO beat Tabu, which beat CW. Here OR-Tools GLS is best and
  ACO is last, even after tuning. Our CW / Tabu / ACO are not the paper's implementations, so
  "beats ACO by X%" is not a portable number. We use **OR-Tools GLS** (the strongest one we have) as
  the reference.
- Even the simple online rule (NN-online, re-decided at each stop) already matches the best a-priori
  plan. Adapting matters about as much as optimizing.

---

## E2 — Reproduce at N = 10: PaperNet + critic (Phase 3)

**Learning rate** (the paper does not report one). Same everything, 15 min each, seed 1 — `results/plots/E2_n10_lr.png`:

| lr   | best val cost | note |
|------|---------------|------|
| 1e-4 | 7.625         | stuck on a plateau ≈ 8.25 for ~8 min, still improving slowly at the end |
| 1e-3 | **6.069**     | breaks through at ~2 min, still creeping down at 15 min |

→ **lr 1e-3 for everything from here on.** The curve has a sharp "aha" drop at ~2 min. Before it, the
policy behaves like the nearest-neighbour rule (≈ 8.2). After it, the policy has discovered a
different way to drive (see the behaviour table below).

**Test, 1000 days** (paired vs OR-Tools GLS):

| method                     | cost  | gap vs GLS [95% CI]     | vs hindsight | deployable |
|----------------------------|-------|-------------------------|--------------|------------|
| OR-Tools GLS (a priori)    | 8.020 | —                       | +32.9%       | yes        |
| nn_split (control)         | 7.867 | −1.91% [−3.28, −0.53]   | +30.3%       | yes        |
| **RL greedy (online)**     | **6.060** | **−24.43% [−25.29, −23.56]** | +0.4% | yes        |
| RL a-priori (plan on estimates, then drive) | 7.062 | −11.95% [−12.90, −10.98] | +17.0% | yes |
| RL beam 3                  | 6.044 | −24.63%                 | +0.1%        | lookahead  |
| RL best-of-16              | 5.612 | −30.02%                 | −7.0%        | hindsight  |

**Is this too good to be true? Checks done:**
1. *Leaks?* The policy sees only [w, estimate-or-revealed demand, ĉ]. The mask uses true remaining
   demand, but for an unvisited customer that is always ≥ 1, so it carries no information. Tests confirm
   the env charges exactly the realized arcs.
2. *Is it just split delivery?* Partial delivery lets a truck hand over what it has and come back later.
   The CVRP planners never plan that, but RL can. Control **nn_split** (nearest open customer, always
   hand over everything, go home only when empty) gets only −1.9%. So splitting alone is worth
   little at N = 10.
3. *Why does RL ≈ hindsight?* "Hindsight" = OR-Tools with perfect information but **no splitting**, so
   it is not a lower bound under partial delivery. RL adapts to revealed demand *and* splits on purpose.

**What it learned** (`scripts/inspect_policy.py`, test days):

|                                   | nn_online | nn_split | RL greedy |
|-----------------------------------|-----------|----------|-----------|
| cost                              | 8.03      | 7.87     | **6.06**  |
| trips per day                     | 6.78      | 5.81     | 6.50      |
| cost per trip                     | 1.18      | 1.35     | **0.93**  |
| splits per day                    | 2.19      | 4.81     | 3.68      |
| trips ending with load left       | 68%       | 17%      | 44%       |

RL does *not* win by packing more into each trip; it makes more trips than nn_split. It wins because
each trip is ~30% cheaper. It learned *which customers belong together and in what order*,
and when it is worth going home early. The greedy "nearest customer next" rule cannot express that.

**Compared to the paper:** the paper reports RL 3.4% better than ACO. We see 24% better than the best
classical method. The likely reason is that our world is very noisy (demand std ≈ 0.8 × mean at
A = 0.6, with ~2 customers per trip). A-priori plans pay many recourse trips there, so adapting matters a lot.
The paper's Σ and demand range are unknown, so its world may be much calmer. Phase 4 varies exactly these knobs.
