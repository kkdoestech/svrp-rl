# Reading list, by lesson

**must** = needed to understand the lesson · **should** = deepens it a lot · **optional** = for the curious.
Everything marked must or should is free online.

## The paper and its relatives

| | what | why |
|---|---|---|
| **must** | Iklassov, Sobirov, Solozabal & Takáč (2023), "Reinforcement Learning for Solving Stochastic Vehicle Routing Problem", ACML 2023, [arXiv 2311.07708](https://arxiv.org/abs/2311.07708) | the paper this project reproduces; your notes `a.md` … `part 6.md` summarise it |
| should | Nazari, Oroojlooy, Snyder & Takáč (2018), "Reinforcement Learning for Solving the Vehicle Routing Problem", [arXiv 1802.04240](https://arxiv.org/abs/1802.04240) | the direct parent of PaperNet (same group); read sections 3–4 |
| should | Kool, van Hoof & Welling (2019), "Attention, Learn to Solve Routing Problems!", [arXiv 1803.08475](https://arxiv.org/abs/1803.08475) | the Transformer design behind GraphNet, and a very clear paper |
| optional | Kwon et al. (2020), "POMO", [arXiv 2010.16011](https://arxiv.org/abs/2010.16011) | the shared baseline (I1) |
| optional | Vinyals, Fortunato & Jaitly (2015), "Pointer Networks", [arXiv 1506.03134](https://arxiv.org/abs/1506.03134) | where "attention that points at an input" started |
| optional | Bello et al. (2016), "Neural Combinatorial Optimization with Reinforcement Learning", [arXiv 1611.09940](https://arxiv.org/abs/1611.09940) | REINFORCE + critic for routing, and the logit-clipping trick |
| optional | The authors' code: [github.com/Zangir/SVRP](https://github.com/Zangir/SVRP) | compare its `env/weather.py` with lesson 04 (the α-per-day issue) |

## Per lesson

| lesson | must | should / optional |
|---|---|---|
| [00](00_prerequisites.md) Prerequisites | [Python tutorial](https://docs.python.org/3/tutorial/) ch. 4–9 | should: 3Blue1Brown *Essence of Linear Algebra* ep. 1–4, 9 (YouTube); *Seeing Theory* (seeing-theory.brown.edu) ch. 1–3 |
| [01](01_setup_and_tools.md) Setup | [uv docs](https://docs.astral.sh/uv/): "Projects", "Using uv with PyTorch" | optional: *Pro Git* ch. 1–2 (git-scm.com/book) |
| [02](02_tensors.md) Tensors | [PyTorch *Tensors* tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) | should: [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) |
| [03](03_config.md) Config | [dataclasses docs](https://docs.python.org/3/library/dataclasses.html) | optional: [argparse tutorial](https://docs.python.org/3/howto/argparse.html) |
| [04](04_generator.md) Generator | your `a.md`, `b.md`; the paper's section 3 | optional: any probability textbook on expectation and variance of sums |
| [05](05_estimates_and_data.md) Data | — | should: *An Introduction to Statistical Learning* (statlearning.com) §3.5 (k-NN) and ch. 5 (validation) |
| [06](06_plans_and_classical_solvers.md) Solvers | [OR-Tools CVRP guide](https://developers.google.com/optimization/routing/cvrp) | should: [OR-Tools routing options](https://developers.google.com/optimization/routing/routing_options); optional: Clarke & Wright (1964), *Operations Research* 12(4) |
| [07](07_executor_and_runner.md) Executor | — | should: Gendreau, Laporte & Séguin (1996), "Stochastic vehicle routing", *EJOR* 88(1), §2; Python `concurrent.futures` docs |
| [08](08_environment.md) Environment | Sutton & Barto, [*RL: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html), ch. 3 | should: Nazari et al. (2018) §3 |
| [09](09_policy_network.md) Network | [PyTorch *Build the Neural Network*](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) | should: 3Blue1Brown *Neural networks* ch. 1–3; [colah, Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/); optional: Bahdanau et al. (2014), [arXiv 1409.0473](https://arxiv.org/abs/1409.0473) |
| [10](10_rollout_and_inference.md) Inference | the paper's §4.3; `part 6.md` | optional: Kool et al. (2019) §4 |
| [11](11_training_reinforce.md) REINFORCE | [Karpathy, "Pong from Pixels"](https://karpathy.github.io/2016/05/31/rl/) | should: Sutton & Barto ch. 13.1–13.4; [Lilian Weng, "Policy Gradient Algorithms"](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/); [PyTorch *Optimization* tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html); optional: Williams (1992), *Machine Learning* 8 |
| [12](12_evaluation_and_experiments.md) Evaluation | — | should: Henderson et al. (2018), "Deep RL that Matters", [arXiv 1709.06560](https://arxiv.org/abs/1709.06560); optional: Efron & Tibshirani, *An Introduction to the Bootstrap*, ch. 1–2 |
| [13](13_improvements.md) Improvements | [Jay Alammar, "The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/) | should: Vaswani et al. (2017), [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) §3; Kool et al. (2019); optional: POMO |
| [14](14_aco_optional.md) ACO | — | optional: Dorigo & Stützle, *Ant Colony Optimization* (2004), ch. 1–3; Bullnheimer, Hartl & Strauss (1999), *Annals of OR* 89 |

## If you only have time for five things

1. Karpathy, "Pong from Pixels": policy gradients finally make sense.
2. Kool et al. (2019): how modern RL-for-routing is done.
3. The PyTorch *Learn the Basics* tutorials (tensors, building a network, optimisation).
4. Sutton & Barto, chapter 3 (MDPs) and chapter 13 (policy gradients).
5. The OR-Tools CVRP guide, re-read with the plan format of lesson 06 in mind.
