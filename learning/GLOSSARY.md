# Glossary: every term in plain words

The lesson in brackets is where the term is explained in depth.

**A-priori plan.** A whole day's route decided in the morning from estimates, then driven as planned
(with recourse when reality differs). Opposite of *online / reoptimisation*. [07, 10]

**A / B / Γ (signal ratio).** The shares of a random quantity's total "size" that come from its constant
base (A), from the weather (B) and from pure noise (Γ). Default 0.6 / 0.2 / 0.2. [04]

**Action.** The node the truck drives to next (0 = depot). [08]

**Adam.** The optimiser used to update the network's parameters; it adapts the step size for each
parameter. [11]

**Advantage.** C − b: how much better or worse a sampled route was than the baseline expected. [11]

**α (alpha).** In the generator: the secret weights linking weather to demand/cost, fixed per city. In
ACO: the importance of pheromone. [04, 14]

**Attention.** A layer that scores how relevant each item (node) is to a query, and mixes information
accordingly. [09, 13]

**Autograd.** PyTorch's automatic computation of gradients via `loss.backward()`. [11]

**Batch.** (1) A group of B days simulated together. (2) The `Batch` class holding truth + beliefs for those
days. [02, 05]

**Baseline (REINFORCE).** b, the expected cost a route is compared against. Either a *critic* network or
the *shared* mean of M routes on the same day. [11]

**Baseline (method).** A classical method RL is compared against (CW, OR-Tools, ACO, NN). [06, 07]

**Beam search.** Keep the W most probable partial routes at every step. Here labeled *lookahead*, because
the beams peek at true demand on the real day. [10]

**Belief.** What may be used for decisions: `dem_hat`, `cost_hat`, weather, positions. [05]

**Best-of-n.** Sample n routes on the real day and keep the cheapest. *Hindsight*: not deployable. [10]

**Bootstrap.** Estimating uncertainty by resampling the test days with replacement many times. [12]

**Broadcasting.** PyTorch's automatic stretching of size-1 dimensions so that shapes fit. [02]

**Checkpoint.** Saved network weights (`best.pt`). [11]

**City.** Everything that stays fixed over days: positions, mean demands θ̄, distances, the weights α. [04]

**Clarke–Wright (CW).** The savings heuristic: start with one trip per customer, then merge the pairs with
the largest savings. [06]

**Clipping (logits).** 10·tanh(u): keeps scores in (−10, 10) so that no move ever gets probability ~1.
[09]

**Clipping (gradients).** Rescale the gradient if its length exceeds 1.0, to avoid one huge,
destructive update. [11]

**Compute capability / `sm_61`.** The architecture number of an NVIDIA GPU (GTX 1060 = 6.1). A PyTorch
build must contain kernels for it. [01]

**Control (experiment).** A test of a boring explanation for a surprising result (e.g. `nn_split`). [12]

**Critic.** A small network that predicts a day's cost from the morning observation; one of the REINFORCE
baselines. [09, 11]

**CUDA.** NVIDIA's platform for running computations on the GPU. [01]

**CVRP.** Capacitated vehicle routing problem: deterministic demand, a truck capacity, minimise distance.
What OR-Tools solves. [06]

**Decode (greedy / sample).** How a node is chosen from the probabilities: the most likely one, or a
random draw. [10]

**Deployable.** Uses only information a real dispatcher has at that moment. [10, 12]

**Depot.** Node 0, at (0.5, 0.5). The truck starts, refills and ends there. [04]

**`detach()`.** Treat a tensor as a constant: no gradient flows through it. [02, 11]

**Device.** Where tensors live and compute: `cpu` or `cuda` (GPU). [02]

**Done.** Per-day flag: everything delivered and the truck is home. [08]

**Embedding.** A learned vector (here 128 numbers) summarising an item such as a node. [09]

**Environment (env).** The game the agent plays: state, actions, transitions, costs (`SVRPEnv`). [08]

**Epoch.** Here: 10,240 freshly generated training days. [11]

**EPS.** 1e-6, a tolerance for floating-point crumbs ("is it zero?"). [07, 08]

**ε (epsilon).** The unpredictable noise part of demand or cost. [04]

**Estimate (constant / k-NN / oracle_mean).** Ways to guess demand and cost before visiting. [05]

**Event (I4).** An observable day type (concert, match) that boosts demand near a venue. [13]

**Executor.** Code that drives a fixed plan on the true day, with recourse. [07]

**Fail / recourse event.** Arriving at a customer with less load than needed. [07, 08]

**Fill rate Φ.** E[demand of one customer] / Q. Default 0.5 gives Q = 20, about 2 customers per trip. [03]

**Fingerprint.** Sums over the test days used to check that two implementations generate identical
worlds. [05]

**Fixed / flexible positions.** The same city every day (B2B), or a new random city each day (B2C). [05]

**Full / partial delivery.** An order must be delivered in one piece (full), or may be split across visits
(partial). [08]

**Gather.** Pick one value per row by index (`logp.gather(1, a[:, None])`). [02]

**Generator (torch.Generator).** A seeded random-number stream: same seed, same numbers. [02, 05]

**GLS.** Guided Local Search, OR-Tools' strongest metaheuristic here. [06]

**Gradient.** For each parameter, how much the loss changes if that parameter is nudged. [11]

**GraphNet.** This project's Transformer policy (improvement I2), usable at any N. [13]

**Greedy.** Always choose the most probable node; deterministic. [10]

**Hindsight.** OR-Tools given the *true* demands and costs. An impossible reference showing what
uncertainty costs. [07]

**History bank.** The past days (weather, demand, cost) that k-NN looks up. [05]

**k-NN.** k-nearest neighbours: average what happened on the k past days with the most similar weather,
weighted by 1/distance. [05]

**Label (deployable / lookahead / hindsight / reference).** The honesty tag printed next to every result.
[12]

**Lab notebook (`LOG.md`).** A record of every experiment: hypothesis, setup, result, decision. [12]

**LayerNorm.** Rescales a vector to mean 0 and spread 1 (then learned scale/shift), which keeps deep
networks stable. [13]

**Learning rate (lr).** The size of each optimisation step. 1e-3 worked here; 1e-4 was too slow. [11]

**Logit.** A raw score before softmax. Higher means more likely. [09]

**Log-probability (logp).** log π(a). A route's logp is the sum of its steps' logps. [10]

**Lookahead.** Uses information from the future of the real day (e.g. beam search here). [10]

**LSTM / LSTMCell.** A recurrent network cell with gates, carrying a memory (h, c) from step to step. [09]

**Mask.** The legal-move table: True = allowed. Illegal moves get logit −∞, so probability 0. [08]

**MDP.** Markov Decision Process: the mathematical form of a sequential decision game. [08]

**Metaheuristic.** A strategy to escape local optima in search (GLS, Tabu, ACO). [06, 14]

**MHA (multi-head attention).** Several attention computations in parallel, each on a slice of the
vector. [13]

**`nn.Module`.** PyTorch's base class for networks; it finds and manages all parameters inside. [09]

**NN-online / nn_split.** The nearest-feasible-neighbour rule re-decided at every stop; `nn_split` always
hands over everything (a control). [08, 12]

**Observation.** The part of the state the policy may see. [08]

**OR-Tools.** Google's optimisation library; here a CVRP solver on the estimates. [06]

**Paired comparison.** Comparing methods day by day on the *same* days, so day difficulty cancels. [05, 12]

**PaperNet.** The paper's network: Conv1d embedding → LSTM → Bahdanau attention. [09]

**Parameter.** A learned number inside the network (a "knob"). PaperNet has ~167k. [09]

**Plan.** A node list like `[0, 3, 5, 0, 2, 0]`; each segment between zeros is a trip. [06]

**Policy π.** The decision rule: observation → probabilities over next nodes. [08, 09]

**POMO / shared baseline.** Use the mean cost of M routes on the same day as the baseline. [11, 13]

**Q (capacity).** How much the truck carries: mean demand / Φ. [03]

**Recourse.** The "oops" penalty: extra depot trips when the truck runs out (R(x) in `a.md`). [07]

**REINFORCE.** The policy-gradient algorithm: push the log-probability of each sampled route against its
advantage. [11]

**Reoptimisation / online.** Deciding the next stop after each arrival, using revealed information. [08]

**Residual connection.** x + f(x): lets a layer add to its input instead of replacing it. [13]

**Rollout.** Playing a whole day with a policy (`rollout()`). [10]

**Run.** One training job: folder `runs/<name>/` with config, log, checkpoint. [11]

**Savings.** s_ij = c_i0 + c_0j − c_ij, what CW gains by joining i → j. [06]

**Seed.** The number that starts a random stream. Different training seeds give different, repeatable
runs. [05, 11]

**Softmax / log_softmax.** Turns logits into probabilities (or their logs), stably. [02, 10]

**Softplus.** log(1 + e^x), always positive; used to keep β > 0. [13]

**State.** Everything that describes the situation mid-day (positions, loads, remaining demand, revealed
flags). [08]

**Stochastic.** Random. SVRP = vehicle routing with random demand and/or travel cost. [04]

**Tabu Search.** A metaheuristic that forbids recently undone moves; the paper's "TS". [06]

**Tag.** The short name of a world (`n10_phi0.5_r0.6-0.2-0.2_…`), used for result folders. [03]

**Tensor.** PyTorch's multi-dimensional array. [02]

**Test / validation split.** Validation days are for choices; test days only for final reports. [05]

**θ̄ (theta-bar).** A quantity's long-run mean: θ̄_i is customer i's mean demand, and θ̄_ij = d_ij is an
edge's distance. [04]

**Transformer.** A network built from self-attention + feed-forward layers with residuals and LayerNorm.
[13]

**Trip / route.** One depot-to-depot segment of a day. [06]

**Truth.** The real demands and costs (`demand`, `cost`); only the env and executor may read them. [05]

**2-opt.** Reverse a route segment if that shortens the route ("uncross" it). [14]

**w (weather).** 3 observable numbers in [−1, 1]: temperature, pressure, humidity. [04]
