<!-- group: Reference | reviewed: 2026-11-26 -->
# Glossary

Terms, symbols and review links used across these pages.

## Terms

| Term | Meaning |
| --- | --- |
| actor | the policy network: observation → motor commands; the only network that flies |
| aggregator | combines the per-objective actor gradients into one update (TorchJD) |
| Aligned-MTL | aggregator that makes the gradients' principal components equally strong |
| arm | one training setting in a study (for example `td3`, `moo4svdpga_split`) |
| asymmetric actor-critic | the critic sees privileged, noise-free information; the actor only what the drone measures |
| bootstrap (TD) | using the critic's own estimate of the next state's value in the target |
| bootstrap (statistics) | resampling seeds with replacement to get an interval for a statistic |
| checkpoint | the actor saved every 100k steps (`actor_<step>.h5`) |
| clipped double-Q | TD3's target uses the smaller of two critics' estimates |
| crash-better state | a state where crashing at once would pay more than flying on (`reward_check.py`) |
| crash penalty $P$ | reward of a crash step: −200 (build), −400 (current runs) |
| critic | the network that estimates the value of an action; used only in training |
| curriculum | episode types, randomization and reward weights that grow during training |
| domain randomization | new random physics every episode |
| DualProj | aggregator: the smallest change to the sum that hurts no objective |
| dual cone | the updates that improve every objective at once (to first order) |
| E (variant E) | the current inputs: 25 numbers including measured rotor speeds, no action history; also "the TD3 baseline" |
| episode | one simulated flight, at most 5 s (500 steps) |
| evaluate_robustness | the C++ test program (mission, waypoint, landing, probe modes) |
| FAST_TANH | RLtools' rational approximation of tanh, clipped at ±3 |
| free flight | episodes starting from a random, often tilted state away from the target |
| group (reward group) | an objective made of reward parts: time, precision, efficiency, safety |
| head scaling | MOO critic outputs ×1/k, initial output layer ×√k (`--head-scale`); collapses the groups |
| Huber cap | position cost quadratic near the target, linear beyond 0.2 m |
| I1 | the first flown mission policy (149 inputs with a 32-step action history) |
| jitter | spread of the motor commands in hover: mean over motors of the standard deviation of the command; lower is smoother |
| Lighthouse | Bitcraze's positioning with laser-sweeping base stations |
| mission | the flight profile: take-off, hover, circle, back, land |
| MOO | multi-objective optimization: one value and one gradient per objective |
| old path | the actor sees the true state plus noise (not the Kalman estimate) |
| old3 | `policies/2026-11-22_old3_moosum3_s40_1800k`, the reference policy of the reward studies |
| OS-NMF-PGA | aggregator: orthonormal directions with non-negative coefficients, averaged |
| PCGrad | aggregator: remove each gradient's component along the ones it conflicts with |
| `pdlp`, `vdlp` | firmware parameters: clip of the position error (0.5 m; 2.0 for waypoints) and velocity error (2 m/s) inputs |
| Polyak averaging | target ← 0.995 target + 0.005 trained network |
| progress reward | shaping term $K(d_t - \gamma d_{t+1})$ |
| reward check | `l2f/py/reward_check.py`: is flying worth more than crashing everywhere, at the final weights |
| reward part | one term of the reward, stored separately in the replay buffer |
| seed | one independent training run |
| sensor path | the actor sees the flown Kalman filter's estimate from simulated sensors (`L2F_SENSOR_SIM`) |
| split critic | one separate critic network per reward group (`--split-critic`) |
| survival bonus $b$ | constant reward per flying step (paper: 2; here 0) |
| SVD-PGA | aggregator: equal-weight average of the gradients' oriented principal directions |
| TD3 | Twin Delayed Deep Deterministic policy gradient: the training algorithm |
| time 0.5 recipe | crash 400, Huber 0.2, 20% waypoint episodes of 0.3-0.5 m, time cost 0.5, simulator options: the base of the MOO studies |
| TorchJD | the PyTorch library for Jacobian descent that provides the aggregators |
| UPGrad | aggregator: project every gradient onto the dual cone, then add |
| uneven motors | the test condition with per-motor mismatch ("randomized") |
| waypoint | an episode or mission where the target jumps |

## Symbols

| Symbol | Meaning | Units / value |
| --- | --- | --- |
| $s, o, a$ | state, observation, action | – |
| $r_t$, $G_t$ | reward at step $t$; discounted return from $t$ | – |
| $\gamma$ | discount factor | 0.99 |
| $\Delta t$ | time step | 10 ms |
| $Q_\theta$, $\mu_\phi$ | critic with weights $\theta$, actor with weights $\phi$ | – |
| $\theta', \phi'$ | target networks | – |
| $d$ (TD) | termination flag (1 = crash or landing) | – |
| $d$ (reward) | distance to the target | m |
| $w_k$, $c_k$ | weight and cost of reward term $k$ | – |
| $s$ (reward) | reward scale | 0.5 |
| $P$ | crash penalty | 200 or 400 |
| $K$ | progress-reward gain | per metre |
| $c$ | Huber cap | 0.2 m |
| $\Omega_i$, $\Omega_h$ | rotor speed; hover speed | rpm; 15,469 rpm |
| $c_f$ | thrust constant | 4.233e-10 N/rpm² |
| $\tau_m$ | motor time constant | 0.06 s (0.045-0.09 randomized) |
| $m$, $J$ | mass; inertia | 41.3 g; kg m² |
| $g_k$ | actor gradient of objective $k$ | – |
| $J$ (MOO) | Jacobian with rows $g_k$ | – |
| $M = J J^\top$ | Gram matrix of the objective gradients | – |
| $\mathbf{w}$ (MOO) | aggregator weights, update $d = J^\top \mathbf{w}$ | – |
| $C$ | dual cone $\{x : g_k \cdot x \ge 0\ \forall k\}$ | – |
| $\lambda_j, v_j$ | eigenvalues and eigenvectors of $M$ | – |
| $k$ (curriculum) | mission progress, 0 to 1 over 800k steps | – |

## Review links

**Reinforcement learning:**
[Sutton and Barto (book)](http://incompleteideas.net/book/the-book-2nd.html) ·
[Spinning Up: key concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) ·
[Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process) ·
[Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation) ·
[temporal-difference learning](https://en.wikipedia.org/wiki/Temporal_difference_learning) ·
[Q-learning](https://en.wikipedia.org/wiki/Q-learning) ·
[deterministic policy gradient](https://proceedings.mlr.press/v32/silver14.html) ·
[DDPG](https://arxiv.org/abs/1509.02971) ·
[TD3](https://arxiv.org/abs/1802.09477) ([Spinning Up](https://spinningup.openai.com/en/latest/algorithms/td3.html)) ·
[Double DQN](https://arxiv.org/abs/1509.06461) ·
[asymmetric actor-critic](https://arxiv.org/abs/1710.06542) ·
[domain randomization](https://arxiv.org/abs/1703.06907) ·
[potential-based shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf) ·
[Learning to Fly in Seconds](https://arxiv.org/abs/2311.13081)

**Optimization and networks:**
[chain rule](https://en.wikipedia.org/wiki/Chain_rule) ·
[Adam](https://arxiv.org/abs/1412.6980) ·
[AdamW](https://arxiv.org/abs/1711.05101) ·
[Kaiming initialization](https://arxiv.org/abs/1502.01852) ·
[Padé approximant](https://en.wikipedia.org/wiki/Pad%C3%A9_approximant) ·
[Huber loss](https://en.wikipedia.org/wiki/Huber_loss) ·
[exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing) ·
[geometric series](https://en.wikipedia.org/wiki/Geometric_series) ·
[PopArt](https://arxiv.org/abs/1602.07714)

**Multi-objective optimization:**
[multi-objective optimization](https://en.wikipedia.org/wiki/Multi-objective_optimization) ·
[Pareto efficiency](https://en.wikipedia.org/wiki/Pareto_efficiency) ·
[TorchJD / UPGrad](https://arxiv.org/abs/2406.16232) ·
[PCGrad](https://arxiv.org/abs/2001.06782) ·
[MGDA for multi-task learning](https://arxiv.org/abs/1810.04650) ·
[Aligned-MTL](https://arxiv.org/abs/2305.19000)

**Linear algebra:**
[3Blue1Brown, essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra) ·
[Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) ·
[Gram matrix](https://en.wikipedia.org/wiki/Gram_matrix) ·
[cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) ·
[vector projection](https://en.wikipedia.org/wiki/Vector_projection) ·
[dual cone](https://en.wikipedia.org/wiki/Dual_cone_and_polar_cone) ·
[quadratic programming](https://en.wikipedia.org/wiki/Quadratic_programming) ·
[eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) ·
[singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) ·
[whitening](https://en.wikipedia.org/wiki/Whitening_transformation) ·
[non-negative matrix factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)

**Dynamics and estimation:**
[Newton-Euler equations](https://en.wikipedia.org/wiki/Newton%E2%80%93Euler_equations) ·
[Euler's rotation equations](https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)) ·
[quaternions and rotation](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation) ·
[rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix) ·
[Runge-Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) ·
[time constant](https://en.wikipedia.org/wiki/Time_constant) ·
[autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model) ·
[Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) ·
[extended Kalman filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter) ·
[Crazyflie state estimation](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/sensor-to-control/state_estimators/) ·
[ground effect](https://en.wikipedia.org/wiki/Ground_effect_(aerodynamics))

**Statistics:**
[Mann-Whitney U test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test) ·
[bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) ·
[Holm-Bonferroni method](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method) ·
[effect size (Cliff's delta)](https://en.wikipedia.org/wiki/Effect_size) ·
[Fisher's exact test](https://en.wikipedia.org/wiki/Fisher%27s_exact_test)
