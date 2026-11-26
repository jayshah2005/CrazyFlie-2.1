<!-- group: Foundations | reviewed: 2026-11-26 -->
# RL foundations used in this project

This page reviews the reinforcement-learning ideas the rest of the documentation builds on, each
tied to where it shows up in the project. If a formula is new to you, the **Review** links lead
to a short explanation. A compact table of all symbols is in the [Glossary](15-glossary.md).

Good general references: Sutton and Barto, *Reinforcement Learning: An Introduction*
([free online](http://incompleteideas.net/book/the-book-2nd.html)), and OpenAI's
[Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html), whose TD3 page is
close to what this project runs.

## The control problem as an MDP

A [Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process) has states
$s$, actions $a$, a transition $s_{t+1} \sim p(\cdot \mid s_t, a_t)$, a reward $r_t$ and a
discount $\gamma$. A policy $\pi$ picks actions; the goal is the policy with the largest expected
discounted return.

| MDP element | In this project |
| --- | --- |
| Time step | 10 ms (the simulator and the onboard policy run at 100 Hz) |
| State $s$ | The drone's true position, attitude, velocities and rotor speeds relative to a target, plus hidden things: mass, motor gains, wind force, estimate delay |
| Observation $o$ | What the drone can measure: 25 numbers for variant E ([Policy I/O](05-policy-io.md)). The state is only partly observed |
| Action $a$ | Four motor commands in $[-1, 1]$, mapped to 0-24,000 rpm |
| Reward $r$ | Minus a weighted sum of costs (position error, tilt, effort, ...); a crash ends the episode with a penalty ([Reward](06-reward.md)) |
| Episode | 5 s (500 steps), or until a crash or a landing |
| Discount $\gamma$ | 0.99 |

The policy sees observations, not states, so strictly this is a partially observed problem. Two
inputs help it infer what it cannot see: the measured rotor speeds (motor mismatch) and a running
sum of the position error (steady pushes such as wind or a mis-calibrated motor).

## Return and discount

The return from step $t$ is the discounted sum of future rewards:

$$
G_t = \sum_{k=0}^{\infty} \gamma^{k}\, r_{t+k}
$$

- $r_{t+k}$: reward $k$ steps ahead (dimensionless; here a negative cost).
- $\gamma \in [0, 1)$: discount factor. A reward $k$ steps ahead counts $\gamma^k$ as much.

Because $\sum_k \gamma^k = 1/(1-\gamma)$ (a [geometric series](https://en.wikipedia.org/wiki/Geometric_series)),
$1/(1-\gamma)$ is the **effective horizon**: with $\gamma = 0.99$ about 100 steps, which is
**1 second of flight**. A constant reward $b$ on every step is worth $b/(1-\gamma)$ in total: the
paper's survival bonus $b = 2$ is worth 200, the same as this project's crash penalty of 200
([Reward](06-reward.md#crash-penalty-against-survival-bonus)).

<div data-widget="discount-horizon">

> **Interactive:** open this page in the viewer (`documentation/index.html`) to change $\gamma$ and see the horizon.

</div>

Why the horizon matters here: every decision the policy makes is judged by roughly the next
second of costs. A waypoint 1.5 m away costs a lot during that second; a crash costs a fixed
amount once. Comparing the two is the heart of the [crash incentive](06-reward.md#the-crash-incentive).

## Value functions and the Bellman equation

The action-value of policy $\pi$ is the expected return after taking action $a$ in state $s$:

$$
Q^{\pi}(s, a) = \mathbb{E}\big[\, G_t \mid s_t = s,\ a_t = a,\ \text{then } \pi \,\big]
$$

It satisfies the [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation), which says
the value now is the reward now plus the discounted value next:

$$
Q^{\pi}(s, a) = \mathbb{E}_{s'}\big[\, r(s, a) + \gamma\, Q^{\pi}(s', \pi(s')) \,\big]
$$

If the episode **terminates** at $s'$ (a crash, a landing), there is no future: the value is just
the final reward. If it is only **cut off** by the 5 s time limit, the future still exists and
the value at $s'$ must be estimated ("bootstrapped"). Getting this distinction right matters:
treating a time-out as a crash would teach the policy that the end of an episode is a cliff.

Review: [Q-learning](https://en.wikipedia.org/wiki/Q-learning), Sutton and Barto chapters 3-4.

## Learning Q from samples: temporal-difference targets

A critic network $Q_\theta(s, a)$ with weights $\theta$ is trained toward a **TD target**
([temporal-difference learning](https://en.wikipedia.org/wiki/Temporal_difference_learning)):

$$
y = r + \gamma\,(1 - d)\, Q_{\theta'}(s', a'), \qquad
\mathcal{L}(\theta) = \big(Q_\theta(s, a) - y\big)^2
$$

- $d = 1$ if $s'$ ended the episode by termination (crash or landing), else 0.
- $Q_{\theta'}$: a **target network**, a slowly moving copy of the critic. Without it the target
  moves with every update and training chases itself.
- $a'$: the action the (target) policy would take in $s'$.

The target copy follows the trained network by Polyak averaging ([Spinning Up's DDPG page](https://spinningup.openai.com/en/latest/algorithms/ddpg.html), "target networks")
(an [exponential moving average](https://en.wikipedia.org/wiki/Exponential_smoothing)):

$$
\theta' \leftarrow 0.995\,\theta' + 0.005\,\theta
$$

so the target averages roughly the last 200 updates. Transitions come from a **replay buffer**
(here up to 700,000 of them), sampled at random in batches of 256, which breaks the correlation
between consecutive steps and lets every flight be reused many times. Learning from data the
current policy did not produce is what makes the method **off-policy**.

## Actor-critic with deterministic actions

For continuous actions (motor speeds) the policy is a network $\mu_\phi(s)$ that outputs the action
directly. The [deterministic policy gradient](https://proceedings.mlr.press/v32/silver14.html)
moves it uphill on the critic, by the [chain rule](https://en.wikipedia.org/wiki/Chain_rule):

$$
\nabla_\phi J(\phi) \approx \mathbb{E}_{s \sim \text{buffer}}\Big[\, \nabla_a Q_\theta(s, a)\big|_{a = \mu_\phi(s)}\ \nabla_\phi \mu_\phi(s) \,\Big]
$$

In words: the critic says which way to change the action to raise the value; the actor changes its
weights to move its action that way. Exploration comes from noise added to the action while
collecting data: $a = \operatorname{clip}(\mu_\phi(s) + \mathcal{N}(0, \sigma^2), -1, 1)$, with
$\sigma = 0.1$ shrinking later in training. This is **DDPG**
([Lillicrap et al. 2015](https://arxiv.org/abs/1509.02971)).

The gradient $\nabla_a Q$ is the only thing the actor learns from. If the critic is wrong about
which way is up, the actor follows it anyway. That is why so much of this project's MOO work is
about the critic ([Multi-objective TD3](09-moo.md#critic-designs)).

## TD3: three fixes to DDPG

TD3 ([Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477); [Spinning Up page](https://spinningup.openai.com/en/latest/algorithms/td3.html))
keeps DDPG and fixes its main failure, **overestimation**:

1. **Clipped double-Q.** Two critics are trained; the target uses the smaller of the two target
   critics: $y = r + \gamma (1-d)\, \min\big(Q_{\theta'_1}(s', a'), Q_{\theta'_2}(s', a')\big)$.
   The actor climbs critic 1, so it settles where critic 1 errs high; critic 2's error there is
   independent, so the minimum cancels most of that bias.
2. **Delayed policy updates.** The actor (and its target) update every second critic update,
   so the actor follows a critic that has had time to settle.
3. **Target policy smoothing.** The target action gets clipped noise,
   $a' = \operatorname{clip}\big(\mu_{\phi'}(s') + \operatorname{clip}(\mathcal{N}(0, 0.2^2), \pm 0.5), -1, 1\big)$,
   so the target is an average over nearby actions and narrow spikes of the critic are smoothed.

<div data-widget="overestimation">

> **Interactive:** in the viewer, a simulation of why the maximum of noisy estimates is biased up and how the minimum of two fixes it.

</div>

The exact schedule this project uses (update intervals, warm-up, noise decay) is on
[TD3 as implemented](07-td3.md). Review: [Double DQN](https://arxiv.org/abs/1509.06461) for the
idea of decoupling selection and evaluation.

## Asymmetric actor-critic

Only the actor runs on the drone, so only the actor is limited to what the drone can measure. The
critic is used in training only, so it can see the **true, noise-free state and hidden
quantities**: the random external force and the true rotor speeds
([Pinto et al. 2017](https://arxiv.org/abs/1710.06542)). A better-informed critic gives better
gradients. In this project the actor sees 25 noisy (or estimated) inputs and the critic 28 clean
ones ([Policy I/O](05-policy-io.md#what-the-critic-sees)).

## Getting from simulation to the real drone

- **Domain randomization** ([Tobin et al. 2017](https://arxiv.org/abs/1703.06907)): every episode
  draws new physics (mass, inertia, thrust, motor lag, motor mismatch, ground effect, delay), so
  the policy must work for a range of drones, the real one hopefully included.
- **Observation noise and delay** matching the measured sensors, so the policy does not rely on
  precision it will not have.
- **Curriculum**: hard parts (mission episodes, wide randomization, large reward weights) ramp in
  over training, because a policy that cannot hover yet learns nothing from a landing episode.

All three are on the [Simulator](04-simulator.md) page.

## Termination and the crash incentive

In this project every per-step reward is a cost (negative), and a crash ends the episode with a
fixed penalty $-P$. From any state, the policy can compare two futures:

$$
\text{fly on: } G_t = \sum_k \gamma^k r_{t+k} \qquad \text{crash now: } \approx -P
$$

If the costs ahead add up to more than $P$, crashing is the better choice, and TD3, which
maximizes return, will learn to crash. This happens with large position errors (free-flight starts
far from the target, waypoint jumps) and large weights. It explains collapsed training seeds and a
whole failed study; the [Reward](06-reward.md#the-crash-incentive) page has the numbers and a
calculator.

## Reward shaping

Adding a term $F(s, s') = \gamma\, \Phi(s') - \Phi(s)$ for any function $\Phi$ of the state
("potential-based shaping") does not change which policy is optimal
([Ng, Harada and Russell 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)):
over a whole episode the terms telescope, $\sum_t \gamma^t F_t = \gamma^T \Phi(s_T) - \Phi(s_0)$.
With $\Phi = -K d$ ($d$ = distance to the target) this becomes the **progress reward**
$K(d_t - \gamma d_{t+1})$: each metre closed is worth $K$, however long the flight takes. Two
caveats: the theorem needs $\Phi = 0$ at termination handled consistently (a crash would otherwise
"refund" the remaining distance), and "same optimum" says nothing about how easy the reward is to
learn. In this project the progress reward passed every check and still failed in training
([Reward](06-reward.md#progress-reward-potential-based-shaping)).

## Multi-objective RL

A multi-objective problem has a **vector** of rewards $\mathbf{r} = (r_1, \dots, r_K)$, one per
objective (for example time, precision, efficiency, safety). Ordinary RL needs a scalar, usually
the weighted sum $r = \mathbf{w}^\top \mathbf{r}$ (linear **scalarization**); the weights $\mathbf{w}$
decide the trade-off. A policy is [Pareto optimal](https://en.wikipedia.org/wiki/Pareto_efficiency)
if no other policy is better on one objective without being worse on another; different weights
pick different Pareto-optimal policies (review:
[multi-objective optimization](https://en.wikipedia.org/wiki/Multi-objective_optimization)).

Gradient-based multi-objective methods keep one gradient per objective and combine them with an
**aggregator** instead of a fixed weighted sum. In this project the critic learns one value per
objective group, the actor gets one gradient per group, and TorchJD's aggregators combine them:
[Multi-objective TD3](09-moo.md).

## Where this is in the code

| Idea | File |
| --- | --- |
| TD3 loop, replay buffer, targets, Polyak | [`l2f/py/td3.py`](../l2f/py/td3.py) |
| Actor and critic networks, optimizer | [`l2f/py/networks.py`](../l2f/py/networks.py) |
| Multi-objective TD3 | [`l2f/py/moo.py`](../l2f/py/moo.py) |
| Episodes, randomization, observations, termination | [`operations_generic.h`](../l2f/learning_to_fly/include/learning_to_fly/simulator/operations_generic.h) |
| Reward | [`squared.h`](../l2f/learning_to_fly/include/learning_to_fly/simulator/parameters/reward_functions/squared.h), [`brushless.h`](../l2f/learning_to_fly/include/learning_to_fly/simulator/parameters/reward_functions/brushless.h) |
