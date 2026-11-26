<!-- group: The system | reviewed: 2026-11-26 -->
# Reward

The reward is the only description of the task the policy ever gets, and in this project it has
been the source of most failures. This page lists every term, shows how the weights change during
training, and explains the problem that shaped most of the recent work: with every reward a cost,
**crashing can be the best choice**.

## The formula

On every step that does not end the episode:

$$
r_t = -\,s \sum_{k} w_k\, c_k(s_t, a_t, s_{t+1}) \;+\; b
$$

On the step that ends it: $r_t = -P$ for a crash, or the landing payout for a motor cut near the
floor.

- $c_k$: the cost terms below, all $\ge 0$.
- $w_k$: their weights (some change during training).
- $s = 0.5$: overall scale.
- $b$: survival bonus per flying step. The paper uses 2; **this project uses 0** since
  2026-11-22.
- $P$: crash penalty. 200 in the build (`-DL2F_CRASH_PENALTY=200`); **400 in current runs**
  (`L2F_CRASH_PENALTY_OVERRIDE=400`).

Sources: [`squared.h`](../l2f/learning_to_fly/include/learning_to_fly/simulator/parameters/reward_functions/squared.h)
(the terms) and [`brushless.h`](../l2f/learning_to_fly/include/learning_to_fly/simulator/parameters/reward_functions/brushless.h)
(the weights).

## Every cost term

| Term | Cost $c_k$ | Units | Weight: start → final | Purpose |
| --- | --- | --- | --- | --- |
| position | $\lVert \mathbf{p} - \mathbf{p}_{\text{target}} \rVert^2$ | m² | 5 → 40 | be at the target |
| orientation | $1 - q_w^2 = \sin^2(\theta/2)$, $\theta$ = rotation away from level at yaw 0 | – | 5 | stay level and face yaw 0 |
| linear velocity | $\lVert \mathbf{v} - \mathbf{v}_{\text{target}} \rVert^2$ | m²/s² | 0.01 → 1 | match the target's motion, damp |
| angular acceleration | $\lVert (\boldsymbol\omega_{t+1} - \boldsymbol\omega_t)/\Delta t \rVert^2$ | rad²/s⁴ | 2e-4 | no jerky rotation (the chatter fix) |
| action | $\lVert \mathbf{a} - a_{\text{hover}} \rVert^2$, $a_{\text{hover}} = 0.288$ | – | 0.01 → 1 | do not waste thrust |
| action rate | $\lVert \mathbf{a}_t - \mathbf{a}_{t-1} \rVert^2$ | – | 1 | smooth commands |
| landing speed | $v_z^2$ when within 10 cm of the floor and sinking | m²/s² | 5 | gentle touchdown |
| zig-zag (option) | $\lVert \mathbf{a}_t - 2\mathbf{a}_{t-1} + \mathbf{a}_{t-2} \rVert^2$ | – | 0 (0.25 tested) | penalize only reversals, not ramps |
| time (option) | 1 while a static target is not reached (5 cm, 0.2 m/s) | – | 0 (0.5 in current runs) | arrive sooner |
| power, speed limit, energy (options) | rotor power above hover; speed above a braking curve; power until arrival | – | 0 | waypoint studies only |
| progress (option) | a reward $K(d_t - \gamma d_{t+1})$, see below | – | 0 | shaping for far targets |

**Worked example.** Hovering 5 cm from the target at the final weight costs
$0.5 \cdot 40 \cdot 0.05^2 = 0.05$ per step; 30 cm off costs $0.5 \cdot 40 \cdot 0.3^2 = 1.8$ per
step, 36 times more. The squared cost is gentle near the target and steep far away.

**The weights are unit conversions as much as priorities.** Angular acceleration is measured in
rad²/s⁴, so its numbers are huge: for a well-flying policy this term costs 0.36 per step at
2e-4, and would cost 1,819 per step at weight 1. An "untuned" reward with every weight 1 therefore
crashed in every episode for every method: one step of angular acceleration cost more than a
crash ([results.md, untuned weights](../l2f/docs/results.md#pytorch-moo-on-the-no-bonus-reward-2026-11-22)).

## The weight curriculum

Position, velocity and action weights start small and grow every 100k steps: position ×1.2 up
to 40 (reached at 1.2M), velocity and action ×1.4 up to 1 (reached at 1.4M). Early on, the policy
only has to learn not to fall; precision becomes expensive once it can fly. The stored reward
parts in the replay buffer are re-weighted with the current weights every time they are sampled
([TD3](07-td3.md#reward-parts-in-the-replay-buffer)).

<div data-widget="curriculum">

> **Interactive:** in the viewer, slide through training and read every weight and noise level.

</div>

> [!IMPORTANT]
> Any analysis of the reward must use the **final** weights. A check with the starting weights
> (position 5) understates position costs by 8 times; that mistake hid the crash incentive of
> the first waypoint study for a whole day ([Waypoints](10-waypoints.md#what-went-wrong-with-the-diagnosis)).

## Crash penalty against survival bonus

The paper rewards every flying step with $b = 2$; crashing loses all future bonuses, worth
$b/(1-\gamma) = 200$. A crash penalty of $P = 200$ with no bonus gives the same incentive in
theory. In training they differ (Step 1 of the plan, 2026-11-21, variant E, 3 seeds, medians over
checkpoints 1.2M-2.0M):

| Option set | Circle | Uneven motors | Crashed | Checkpoints without a crash |
| --- | --- | --- | --- | --- |
| bonus 2, no penalty (the paper) | 6.3 cm | 12.6 cm | 3.1% | 3/27 |
| bonus 2 + penalty 200 | 7.6 cm | 13.1 cm | 2.1% | 10/27 |
| **no bonus, penalty 200** | **5.3 cm** | **10.0 cm** | **0.8%** | 17/27 |
| no bonus, penalty 400 | 6.3 cm | 9.6 cm | 0.7% | 19/27 |

The penalty without the bonus is clearly better (per seed 0.2-1.7% crashed against 3.0-3.5%). A
likely reason: with a bonus, every value is a large positive number dominated by "how long will I
survive", and the critic must resolve small cost differences on top of it; with costs only, the
values directly measure the costs. On target steps the no-bonus policies also had the fewest
failed responses, no crash and the least overshoot
([results.md, Step 1](../l2f/docs/results.md#simulator-and-reward-options-plan-step-1)).

The price: with no bonus, **staying alive is worth nothing in itself**. That is the crash
incentive.

## The crash incentive

TD3 maximizes the discounted return about one second ahead. Every reward is a cost, and a crash
ends the episode at $-P$ with no costs after it. So from any state the policy compares

$$
\underbrace{G_t = \sum_{k \ge 0} \gamma^k r_{t+k}}_{\text{fly on}} \quad\text{with}\quad \underbrace{-P}_{\text{crash now}}
$$

and **wherever the costs still ahead add up to more than $P$, crashing is the better choice.**
Training then drifts toward crashing in those states. This is not a bug in TD3; it is what the
reward asks for.

Where it bites:

- **Free flight** starts up to 0.2 m away per axis and tilted up to 90°. Recovering costs about
  as much as the crash penalty: old3's surviving free-flight episodes have a median return of −182
  against −200.
- **Waypoint jumps.** A target 1.5 m away costs $0.5 \cdot 40 \cdot 1.5^2 = 45$ per step with the
  squared cost. A perfectly flown 1 m leg costs about −612 over its first 3 s.

<div data-widget="crash-incentive" data-jump="1.0" data-shape="huber">

> **Interactive:** in the viewer, a calculator for the return of a perfect waypoint leg under each position-cost shape, against the crash penalty.

</div>

What was measured:

| Policy and settings | Free flight | Take-off | Landing | Circle | Waypoint |
| --- | --- | --- | --- | --- | --- |
| old3 (best flown policy, crash 200) | 8.6% of states | 2.5% | 0.7% | 3.6% | – |
| a healthy TD3 seed | 22.4% | 2.4% | 0.6% | 4.1% | – |
| old3 with the first waypoint study's settings | 18.2% | 2.9% | 1.2% | 5.2% | **83.6%** |

(Share of surviving states, first 2 s, where crashing would have paid more: the output of
`reward_check.py`.) In the first waypoint study, surviving policies paid a discounted return of
about −430 after each jump, more than twice the crash penalty, and every method learned to crash
([Waypoints](10-waypoints.md)). In the reward studies, one TD3 seed in eight (base seed 71) gave in
to the free-flight incentive: its training return **improved** from −757 to −347 while its episodes
shrank to 0.2-0.3 s.

## Fixes that were tried

Scored first with the reward check on the same flights (free-flight crash-better share, old3 /
a TD3 seed), then trained ([reward_studies.md](../l2f/docs/reward_studies.md)):

| Change | Reward check | Training result | Verdict |
| --- | --- | --- | --- |
| none (crash 200) | 8.6% / 22.4% | reference; 1 seed in 8 collapses | – |
| survival bonus 1 | 2.9% / 11.0% | not trained | – |
| **crash penalty 400** | 0.8% / 5.8% | crash-free checkpoints 11/36 → 22/36, circle +1.2 cm | **new base** |
| Huber position cap 0.2 m | 4.8% / 16.6% | slightly better circle, less reliable | needed for waypoints only |
| progress reward $K = 20$ | 1.2% / 8.0% | 10-20 cm steady offset, 46% crashes | **failed** |
| progress 20 + crash 400 | 0.1% / 1.2% | 14 cm, 26% crashes | **failed** |
| power cost without a floor (first waypoint study) | – | paid +1.07 per step for idling: every run fell | **bug, fixed** |

### Huber position cap

$$
c_{\text{pos}}(d) = \begin{cases} d^2 & d \le c \\ c^2 + 2c\,(d - c) & d > c \end{cases}
$$

with $c = 0.2$ m ([Huber loss](https://en.wikipedia.org/wiki/Huber_loss)), and no velocity cost
beyond the cap. Quadratic near the target (precision unchanged), linear far away (a 1.5 m jump
costs $0.5 \cdot 40 \cdot 0.56 = 11.2$ per step instead of 45). A **flat** cap ($\min(d^2, c^2)$)
was tried first and left the drones about 30 cm off: beyond the cap nothing pulled them in. The
Huber cap pulls, but a long leg is still expensive, because the same slope sets both the pull and
the total cost of a leg.

### Progress reward (potential-based shaping)

$$
r_{\text{prog}} = K\,(d_t - \gamma\, d_{t+1}), \qquad \text{and } -K d_t \text{ on a crash step}
$$

with the position cost flat beyond 0.2 m (`L2F_PROGRESS_K`, `L2F_PROGRESS_CAP`). It is
[potential-based shaping](02-rl-foundations.md#reward-shaping) with $\Phi = -Kd$: over a leg it
telescopes to about $K d_0$, **however long the flight takes**, so the pull is $K$ per metre while
a whole leg costs only about −40 (table in the widget above). The crash term stops a crash from
refunding the distance still to go. It passed the reward check easily, yet every trained policy
held a steady 10-20 cm offset in every phase (hover 9-21 cm against 3 cm) and crashed often.
Not diagnosed; the likely reasons are that the critic must learn a term that changes sign with
the motion (plus noise from moving targets), and that the flat cost beyond 0.2 m leaves little
slope there. The option stays in the code, off. **Lesson: the reward check finds incentive
problems; it cannot tell whether a reward is learnable.**

### Time cost

1 per step while a static target is not reached, weight 0.5 (0.25 per step after the scale). It
is small far away and about the size of the position cost 10 cm out, so it mostly pays for closing
the last 10 cm. Over 8 seeds a weight of 0.5 was the best waypoint-trained arm, and 0.2 was
consistently **worse than no time cost** (not explained). Larger weights failed the reward check
([Waypoints](10-waypoints.md#the-time-cost)).

## The reward check

[`l2f/py/reward_check.py`](../l2f/py/reward_check.py) flies a trained policy through the training
episode mix at full curriculum and the **final** weights and, for every surviving state in the
first 2 s, compares the discounted return of flying on with crashing at once ($G_t < -P - K d_t$).
It reports per episode type the share of crash-better states, the median return and the main cost
parts, and a sign check (every part but survival and progress must be a cost). Verdict: **FAIL**
above 5% of states, **WARN** above 0%, and a warning when more than 20% of a type's episodes crash
(then only the easy episodes survive and the numbers look better than the reward is).
`--candidate` scores other rewards on the same flights; `--group-scale` scores an untuned reward.
It takes seconds and gates every reward change before training.

## Reward parts and groups for MOO

The simulator module returns the reward split into **parts** that add up exactly to it:
survival (the terminal reward), position, orientation, linear velocity, angular velocity, linear
acceleration, angular acceleration, action, action rate, zig-zag, landing speed, time, power,
speed limit, energy, progress. Multi-objective training groups them:

| Group | Parts |
| --- | --- |
| time | time |
| precision | position, linear velocity, progress |
| efficiency | action, action rate, zig-zag, angular acceleration, power, energy |
| safety | survival (the crash penalty), orientation, landing speed, speed limit |

The critic learns one value per group, and the actor gets one gradient per group
([Multi-objective TD3](09-moo.md)). **Equal shares** (`train.py --group-scale`) rescale each
group so all four have the same mean cost per step, with the same total: the "untuned" reward of
Study 8 (factors time 2.3, precision 0.53, efficiency 0.68, safety 7.3; it needed a crash penalty
of 600 to pass the reward check).

## Lessons

- Every reward change gets a reward check with the final weights, then a training study.
- A new term can change the crash incentive of **every** episode type, not only the one it was
  designed for.
- A cost that can go negative pays the policy (the power cost paid for idling).
- Equal-in-theory rewards are not equal in training (bonus against penalty).
- Weights encode units as well as priorities; an "untuned" reward needs at least a unit scaling.
