"""Multi-objective TD3 with TorchJD (plan A6).

Framing A (objectives = reward parts): the critics have K = 8 outputs, one per part group
  (parts=3: the groups tracking = position + velocity, smoothness = action + smoothing +
  angular acceleration, safety = survival + orientation + landing speed)
  0 survival (the constant; on a terminated step the whole reward: crash penalty or landing
    payout), 1 position, 2 orientation, 3 linear velocity, 4 action, 5 smoothing (action rate
    or zig-zag), 6 angular acceleration, 7 landing speed.
  Output k learns its own TD target r_k + gamma * Q'_k, with the twin taken whose summed value
  sum_k f_k Q'_k is smaller, so the summed target equals the scalar TD3 target. The actor gets
  one gradient per output, g_k = grad of -mean_b f_k Q1_k(s_b, actor(s_b)), and TorchJD's
  aggregator combines the rows of the Jacobian J = [g_1; ...; g_K] into the step. With "sum"
  the step is sum_k g_k, the gradient of the summed value: plain TD3. parts=1 merges all parts
  into one output (the MOO code path learning the scalar reward; must match td3.py).
Framing B (objectives = mission episode types, tasks=5 or 9): scalar critic; objective t is
  -mean critic value over a batch of task t's transitions only; "task_share" weights each task
  by its share of the replay buffer (plain TD3 on the mixed buffer, split per task).

Options: total (critic loss (sum_k f_k e_k)^2 + total * sum_k e_k^2 per sample instead of
sum_k e_k^2) and norm (output k learns u_k / s_k with s_k = RMS(u_k) / (1 - gamma) from 20k
buffer samples when critic training starts, f_k = w_k s_k). Preference vectors are all ones
(TorchJD's default is 1/K; the weights are linear in it).

critic_agg (framing A, shared critic only; added 2026-11-25): MOO inside the critic too. The
critic loss is split into objectives: the total error mean (sum_k f_k e_k)^2 when `total` is set,
and one per group, total * mean e_k^2 (mean e_k^2 without `total`; left out when total = 0). Their
gradients on the shared hidden layers are combined by `critic_agg`; the output layer gets their
plain sum (each head's gradient comes from its own group's error, plus the total term). With
"sum" the step equals the plain critic loss's. critic_norm: each group's TD error is divided by
its running RMS (exponential average of mean e_k^2, rate 0.01 per critic update, one per twin),
so the group losses stay about 1 through training (--norm scales once at critic start and left
the groups' gradients 1e5 apart, 2026-11-25).
"""
import numpy as np
import torch
from torchjd import aggregation
from torchjd.autojac import backward, jac_to_grad

from networks import Critic, SplitCritic, TowerCritic, make_optimizer
from td3 import TD3

OBJECTIVES = ['survival', 'position', 'orientation', 'linear_velocity', 'action', 'smoothing',
              'angular_acceleration', 'landing_speed']
# the progress reward (L2F_PROGRESS_K) goes with position everywhere: it is the far-field position term
GROUP_OF = {'survival': 0, 'position': 1, 'orientation': 2, 'linear_velocity': 3, 'action': 4,
            'action_rate': 5, 'zigzag': 5, 'angular_acceleration': 6, 'landing_speed': 7,
            'time': 1, 'power': 4, 'speed_limit': 7, 'energy': 4, 'progress': 1}
# parts=3: the groups tracking, smoothness, safety; untested before 2026-11-22
GROUPS3 = ['tracking', 'smoothness', 'safety']
GROUP3_OF = {'position': 0, 'linear_velocity': 0, 'action': 1, 'action_rate': 1, 'zigzag': 1,
             'angular_acceleration': 1, 'survival': 2, 'orientation': 2, 'landing_speed': 2,
             'time': 0, 'power': 1, 'speed_limit': 2, 'energy': 1, 'progress': 0}
# parts=4 (time-to-target study, 2026-11-22): time, precision, efficiency, safety
GROUPS4 = ['time', 'precision', 'efficiency', 'safety']
GROUP4_OF = {'time': 0, 'position': 1, 'linear_velocity': 1, 'action': 2, 'action_rate': 2, 'zigzag': 2,
             'angular_acceleration': 2, 'power': 2, 'survival': 3, 'orientation': 3, 'landing_speed': 3,
             'speed_limit': 3, 'energy': 2, 'progress': 1}
# regrouped by measured conflict (2026-11-23, ungrouped run: survival fights the dense costs,
# time agrees with position): --grouping regroup with parts=4 (crash, progress, effort, limits)
# or parts=5 (+ energy per leg as its own objective)
REGROUPS = {4: ['crash', 'progress', 'effort', 'limits'], 5: ['crash', 'progress', 'effort', 'limits', 'energy']}
REGROUP_OF = {4: {'survival': 0, 'position': 1, 'linear_velocity': 1, 'time': 1, 'energy': 1, 'progress': 1,
                  'action': 2, 'action_rate': 2, 'zigzag': 2, 'angular_acceleration': 2, 'power': 2,
                  'orientation': 3, 'speed_limit': 3, 'landing_speed': 3}}
REGROUP_OF[5] = {**REGROUP_OF[4], 'energy': 4}
TASK_NAMES = ['free_flight', 'takeoff', 'landing', 'circle', 'line', 'eight', 'fast_circle', 'p2p', 'gust', 'waypoint']
AGGREGATORS = ['sum', 'mean', 'upgrad', 'aligned_mtl', 'dualproj', 'pcgrad', 'mgda', 'svd_pga', 'osnmf_pga', 'cagrad',
               'task_share']


def part_factors(group_scale, part_names):
    """Per-part weight factors from a group scale "time=a,precision=b,efficiency=c,safety=d" (4 groups,
    GROUP4_OF); survival (the crash penalty) and parts outside the 4 groups keep factor 1. Empty: all 1.
    Used for the untuned comparison: each group scaled to the same average cost per step."""
    f = np.ones(len(part_names))
    if not group_scale:
        return f
    scale = {}
    for item in group_scale.split(','):
        key, _, val = item.partition('=')
        if key not in GROUPS4 or not val:
            raise ValueError(f'group scale {group_scale!r}: give {"=x,".join(GROUPS4)}=x')
        scale[key] = float(val)
    for i, n in enumerate(part_names):
        if n != 'survival' and n in GROUP4_OF:
            f[i] = scale.get(GROUPS4[GROUP4_OF[n]], 1.0)
    return f


def make_aggregator(name, m):
    """TorchJD aggregator `name` for m objectives (preference vectors all ones)."""
    ones = torch.ones(m)
    makers = {'sum': aggregation.Sum, 'mean': aggregation.Mean,
              'upgrad': lambda: aggregation.UPGrad(pref_vector=ones),
              'aligned_mtl': lambda: aggregation.AlignedMTL(pref_vector=ones),
              'dualproj': lambda: aggregation.DualProj(pref_vector=ones),
              'pcgrad': aggregation.PCGrad,  # added 2026-11-22 (MOO round 2): stays near the sum
              'mgda': aggregation.MGDA, 'svd_pga': aggregation.SVDPGA, 'osnmf_pga': aggregation.OSNMFPGA,
              # added 2026-11-27 (critic aggregation): c = 0.5, the CAGrad paper's default; needs cvxpy, clarabel
              'cagrad': lambda: aggregation.CAGrad(c=0.5)}
    if name not in makers:
        raise ValueError(f'unknown aggregator {name} ({", ".join(AGGREGATORS)})')
    return makers[name]()


def group_parts(part_names, weights, k, objectives=None, grouping='default'):
    """(U, W): unweighted grouping U (n_parts x k) and group weights W (k) with W_j U_ij = w_i
    for every part i in group j: a group with
    one weighted part takes that part's weight; one with several (rate and zig-zag both on)
    holds the weighted sum with group weight 1. k = 1: one group of all parts, weight 1;
    k = 3: GROUPS3."""
    w = np.asarray(weights, np.float64)
    if objectives is not None:
        # ungrouped (parts=0): one objective per listed reward part, with that part's weight
        U = np.zeros((len(part_names), len(objectives)))
        W = np.zeros(len(objectives))
        for j, name in enumerate(objectives):
            i = part_names.index(name)
            U[i, j], W[j] = 1.0, w[i]
        unlisted = [n for i, n in enumerate(part_names) if n not in objectives and w[i] != 0]
        if unlisted:
            raise ValueError(f'reward parts {unlisted} have a weight but no objective')
        return U.astype(np.float32), W.astype(np.float32)
    if k == 1:
        return w[:, None].astype(np.float32), np.ones(1, np.float32)
    group_of = REGROUP_OF[k] if grouping == 'regroup' else {3: GROUP3_OF, 4: GROUP4_OF}.get(k, GROUP_OF)
    U = np.zeros((len(part_names), k))
    W = np.zeros(k)
    for j in range(k):
        members = [i for i, n in enumerate(part_names) if group_of.get(n) == j]
        on = [i for i in members if w[i] != 0]
        if len(on) <= 1:
            i = on[0] if on else members[0]
            U[i, j], W[j] = 1.0, w[i]
        else:
            U[on, j], W[j] = w[on], 1.0
    ungrouped = [n for i, n in enumerate(part_names) if n not in group_of and w[i] != 0]
    if ungrouped:
        raise ValueError(f'reward parts {ungrouped} have a weight but no objective')
    return U.astype(np.float32), W.astype(np.float32)


class MooTD3(TD3):
    """TD3 whose actor step is a TorchJD aggregation of per-objective gradients."""

    def __init__(self, cfg, obs_dim, obs_dim_p, act_dim, agg='sum', parts=8, tasks=0, total=None, norm=False,
                 log_every=50, objectives=None, grouping='default', agg_start=0, head_scale=False, split_critic=False,
                 critic_agg=None, critic_norm=False, critic_hidden=None, critic_towers=False, critic_norm_by='error',
                 critic_layer_norm=False, crash_split=None):
        self.tasks = tasks
        if grouping == 'regroup' and parts not in REGROUP_OF:
            raise ValueError('--grouping regroup needs parts 4 or 5')
        self.grouping = grouping
        # parts=0: ungrouped, one objective per reward part in `objectives` (added 2026-11-23)
        self.objectives = list(objectives) if parts == 0 and not tasks else None
        if parts == 0 and not tasks and not self.objectives:
            raise ValueError('parts=0 needs the list of reward parts (objectives)')
        k = 1 if tasks else (len(self.objectives) if self.objectives else parts)
        if norm and k == 1:
            raise ValueError('norm needs several parts')
        super().__init__(cfg, obs_dim, obs_dim_p, act_dim, n_out=k)
        self.k, self.n_obj = k, (tasks or k)
        # head_scale (k > 1): the total value is the sum of k output heads. When it is wrong every
        # head gets the same correction and Adam moves each about as far as TD3 moves its one head,
        # so the sum moves about k times as far per update, and k random heads start with sqrt(k)
        # times the spread. Outputs x 1/k and initial output-layer weights and biases x sqrt(k) give
        # the sum TD3's step and starting spread (ablation_moo2: MOO sum trails TD3 even with
        # --total 0; 2026-11-23).
        self.head_scale = head_scale and k > 1
        # split_critic (k > 1): one separate critic network per group instead of k heads on shared
        # hidden layers. Replaces the critics TD3 built (the actor's initialization is unchanged).
        self.split_critic = split_critic and k > 1
        if self.split_critic:
            if self.head_scale:
                raise ValueError('--split-critic and --head-scale do not combine')
            # critic_hidden with split_critic: the width of each group's network (narrow split critic,
            # 2026-11-27: 4 x 32-32 = 8,964 parameters, 1.33x the shared 64-64 critic)
            width = critic_hidden or cfg.hidden
            self.critics = [SplitCritic(obs_dim_p, act_dim, width, k) for _ in range(2)]
            self.critic_targets = [SplitCritic(obs_dim_p, act_dim, width, k) for _ in range(2)]
            for c, ct in zip(self.critics, self.critic_targets):
                ct.load_state_dict(c.state_dict())
            self.critic_opts = [make_optimizer(c) for c in self.critics]
        # critic_hidden: the shared critic's width (capacity control for the split critic, which has
        # 3.9x the shared critic's parameters at 64); critic_towers: a partly shared critic
        self.critic_towers = critic_towers and k > 1
        if critic_layer_norm and (self.critic_towers or self.head_scale or k == 1):
            raise ValueError('--critic-layer-norm needs a shared or split critic (not towers or head scaling)')
        if critic_layer_norm and self.split_critic:
            from networks import SplitCritic as _SC
            make_split = lambda: _SC(obs_dim_p, act_dim, cfg.hidden, k)
            self.critics = [make_split() for _ in range(2)]
            self.critic_targets = [make_split() for _ in range(2)]
            for c in self.critics + self.critic_targets:
                c.nets = torch.nn.ModuleList([Critic(obs_dim_p, act_dim, cfg.hidden, 1, layer_norm=True) for _ in range(k)])
            for c, ct in zip(self.critics, self.critic_targets):
                ct.load_state_dict(c.state_dict())
            self.critic_opts = [make_optimizer(c) for c in self.critics]
        elif critic_layer_norm:
            self.critics = [Critic(obs_dim_p, act_dim, critic_hidden or cfg.hidden, k, layer_norm=True) for _ in range(2)]
            self.critic_targets = [Critic(obs_dim_p, act_dim, critic_hidden or cfg.hidden, k, layer_norm=True) for _ in range(2)]
            for c, ct in zip(self.critics, self.critic_targets):
                ct.load_state_dict(c.state_dict())
            self.critic_opts = [make_optimizer(c) for c in self.critics]
            critic_hidden = None  # built here
        # crash_split (2026-11-27): the terminal reward (crash penalty / landing payout, the
        # 'survival' part) is split across the groups in these shares (summing to 1) instead of all
        # going to safety, so no cost group gains from ending the episode; the scalar reward is
        # unchanged (the shares sum to 1)
        if crash_split is not None:
            crash_split = np.asarray(crash_split, np.float32)
            if parts != 4 or len(crash_split) != k or crash_split.min() < 0 or abs(crash_split.sum() - 1) > 1e-6:
                raise ValueError('--crash-split needs --parts 4 and 4 non-negative shares summing to 1')
        self.crash_split = crash_split
        if (critic_hidden and self.head_scale) or (self.critic_towers and (self.split_critic or self.head_scale)):
            raise ValueError('--critic-hidden / --critic-towers do not combine with --split-critic or --head-scale')
        if critic_hidden and self.critic_towers:
            raise ValueError('--critic-hidden and --critic-towers do not combine')
        if (critic_hidden and not self.split_critic) or self.critic_towers:
            make = (lambda: TowerCritic(obs_dim_p, act_dim, cfg.hidden, k)) if self.critic_towers else \
                (lambda: Critic(obs_dim_p, act_dim, critic_hidden, k))
            self.critics = [make() for _ in range(2)]
            self.critic_targets = [make() for _ in range(2)]
            for c, ct in zip(self.critics, self.critic_targets):
                ct.load_state_dict(c.state_dict())
            self.critic_opts = [make_optimizer(c) for c in self.critics]
        if self.head_scale:
            with torch.no_grad():
                for c in self.critics:
                    c.layers[-1].weight.mul_(k ** 0.5)
                    c.layers[-1].bias.mul_(k ** 0.5)
            for c, ct in zip(self.critics, self.critic_targets):
                ct.load_state_dict(c.state_dict())
            for c in self.critics + self.critic_targets:
                c.out_scale = 1.0 / k
        self.agg_name = agg
        if agg == 'task_share' and not tasks:
            raise ValueError('task_share needs tasks')
        self.aggregator = None if agg == 'task_share' else make_aggregator(agg, self.n_obj)
        # agg_start > 0: the plain sum until that step, then `agg` (the aggregators act mostly during
        # the curriculum ramp, when the crash term dominates; ablation_aggregators, 2026-11-23)
        self.agg_start = agg_start
        self.warmup_aggregator = aggregation.Sum() if agg_start > 0 else None
        self.total, self.norm = total, norm
        # critic_agg: the critic's per-objective gradients on its shared hidden layers are aggregated
        if critic_agg is not None and (tasks or k == 1 or self.split_critic or self.critic_towers):
            raise ValueError('--critic-agg needs --moo parts with several groups and a shared critic (no --split-critic)')
        self.critic_agg = critic_agg
        self.critic_names = None
        if critic_agg is not None:
            groups = [] if total == 0 else self.names_for(k, grouping)
            self.critic_names = (['total'] if total is not None else []) + groups
            # one aggregator per twin (some aggregators keep state between steps)
            self.critic_aggregators = [make_aggregator(critic_agg, len(self.critic_names)) for _ in range(2)]
        self.critic_updates, self.critic_rows = 0, []
        if critic_norm and critic_agg is None:
            raise ValueError('--critic-norm needs --critic-agg')
        self.critic_norm = critic_norm
        # critic_norm_by: 'error' divides each group's TD error by the error's running RMS (every
        # group's loss stays ~1 however well it fits: size-blind, like PGA); 'target' divides it by
        # the running std of the group's TD targets (common units, error size still counts; PopArt-like)
        if critic_norm_by not in ('error', 'target'):
            raise ValueError('critic_norm_by is error or target')
        self.critic_norm_by = critic_norm_by
        self.td_ms = [None, None]  # critic_norm: running mean e_k^2 per twin
        self.t = 0
        self.scale = np.ones(k, np.float32)
        self.group_weights = np.ones(k, np.float32)
        self.log_every, self.updates, self.rows = log_every, 0, []
        self.share_rng = np.random.default_rng(54321)
        self.names = [TASK_NAMES[t] for t in range(tasks)] if tasks else self.names_for(k, grouping)

    def names_for(self, k, grouping):
        """Names of the k critic outputs (framing A)."""
        return list(self.objectives or (REGROUPS[k] if grouping == 'regroup' else {8: OBJECTIVES, 4: GROUPS4, 3: GROUPS3, 1: ['total']}[k]))

    def reward_matrix(self, part_names, weights):
        """R for the critic outputs (see td3.py); also sets the value factors f."""
        U, W = group_parts(part_names, weights, self.k, self.objectives, self.grouping)
        self.group_weights = W
        self.part_names = list(part_names)
        if self.crash_split is not None:
            if self.norm:
                raise ValueError('--crash-split does not combine with --norm')
            R = U * W
            i = self.part_names.index('survival')
            R[i, :] = R[i, :].sum() * self.crash_split
            self.factors = torch.ones(self.k)
            return R
        if self.norm:
            self.factors = torch.from_numpy(W * self.scale)
            return U / self.scale
        self.factors = torch.ones(self.k)
        return U * W

    def before_critic_training(self, buf, rng, reward_matrix):
        """norm: s_k = RMS(u_k) / (1 - gamma) from 20k buffer samples (1 for a part not seen yet)."""
        if not self.norm:
            return
        idx = np.random.default_rng(12345).integers(0, buf.n, 20_000)
        u = buf.parts[idx] @ (reward_matrix * self.scale)  # unweighted group values
        rms = np.sqrt(np.mean(u.astype(np.float64) ** 2, axis=0))
        self.scale = np.where(rms > 0, rms / (1 - self.cfg.gamma), 1.0).astype(np.float32)
        self.factors = torch.from_numpy(self.group_weights * self.scale)

    def critic_loss(self, q, y):
        """sum_k e_k^2, or with `total`: (sum_k f_k e_k)^2 + total * sum_k e_k^2 (batch means)."""
        e = q - y
        if self.total is None:
            return torch.mean(e ** 2) * self.k
        return torch.mean((e * self.factors).sum(1) ** 2) + self.total * torch.mean(e ** 2) * self.k

    def critic_objectives(self, q, y, i=0):
        """critic_agg: the critic loss split into objectives (their sum is critic_loss(q, y) without
        critic_norm; with it each group's error is divided by a running scale of twin i: the error's
        RMS, or with critic_norm_by 'target' the std of the group's TD targets)."""
        e = q - y
        # the total objective is the error of the real total value (raw errors, real units)
        losses = [torch.mean((e * self.factors).sum(1) ** 2)] if self.total is not None else []
        if self.critic_norm:
            ms = y.detach().var(0) if self.critic_norm_by == 'target' else (e.detach() ** 2).mean(0)
            self.td_ms[i] = ms if self.td_ms[i] is None else 0.99 * self.td_ms[i] + 0.01 * ms
            # groups without weighted reward keep their raw (tiny) error: scaling it to 1 would fit noise
            on = torch.as_tensor(self.group_weights, dtype=e.dtype) > 0
            e = torch.where(on, e / self.td_ms[i].clamp_min(1e-12).sqrt(), e)
        w = 1.0 if self.total is None else self.total
        if w != 0:
            losses += [w * torch.mean(e[:, j] ** 2) for j in range(self.k)]
        return losses

    def train_critic(self, i, batch, noise=None):
        """Without critic_agg as TD3. With it: the shared hidden layers step along the aggregated
        per-objective gradients, the output layer along their sum."""
        if self.critic_agg is None:
            return super().train_critic(i, batch, noise)
        aggregator = self.critic_aggregators[i]
        if self.t < self.agg_start:  # --agg-start applies to the critic as well
            aggregator = self.warmup_aggregator
        obs_p, act = batch[1], batch[2]
        y = self.td_target(batch, noise)
        critic = self.critics[i]
        losses = self.critic_objectives(critic(obs_p, act), y, i)
        shared = [p for layer in critic.layers[:-1] for p in layer.parameters()]
        head = list(critic.layers[-1].parameters())
        self.critic_opts[i].zero_grad(set_to_none=True)
        total_loss = sum(losses)
        head_grads = torch.autograd.grad(total_loss, head, retain_graph=True)
        backward(losses, inputs=shared)
        if i == 0:
            self.critic_updates += 1
        log_now = i == 0 and self.critic_updates % self.log_every == 0
        weights = jac_to_grad(shared, aggregator, retain_jac=log_now)
        for p, g in zip(head, head_grads):
            p.grad = g
        if log_now:
            self.critic_rows.append([self.critic_updates, self.critic_agg]
                                    + self.gradient_stats(shared, weights, len(losses)) + [float(x.detach()) for x in losses])
        self.critic_opts[i].step()
        return float(total_loss.detach())

    def update(self, t, buf, rng, part_names, weights):
        """TD3's update; remembers the step (the critic aggregator's --agg-start warm-up)."""
        self.t = t
        return super().update(t, buf, rng, part_names, weights)

    def critic_log_fields(self):
        """CSV header for the critic rows (critic 0, shared hidden layers)."""
        n = self.critic_names
        return (['critic_update', 'aggregator'] + [f'norm_{x}' for x in n] + [f'weight_{x}' for x in n]
                + [f'cos_{n[a]}_{n[b]}' for a in range(len(n)) for b in range(a + 1, len(n))]
                + ['step_norm', 'check_rel_error'] + [f'loss_{x}' for x in n])

    def objective_losses(self, buf, rng, reward_matrix):
        """The n_obj actor losses -mean f_k Q1_k, from one batch (parts) or one per task."""
        if not self.tasks:
            obs, obs_p = buf.sample(rng, self.cfg.batch, reward_matrix)[:2]
            q = self.critics[0](obs_p, self.actor(obs)) * self.factors
            return [-q[:, k].mean() for k in range(self.k)]
        losses = []
        for t in range(self.tasks):
            idx = buf.indices(rng, self.cfg.batch, task=t)
            if idx is None:  # not in the buffer yet: a zero gradient
                idx = buf.indices(rng, self.cfg.batch)
                scale = 0.0
            else:
                scale = 1.0
            obs, obs_p = buf.batch(idx, reward_matrix)[:2]
            losses.append(-scale * self.critics[0](obs_p, self.actor(obs)).mean())
        return losses

    def task_share_weights(self, buf):
        """n_tasks * each task's share of 4096 random buffer entries."""
        tasks = buf.task[self.share_rng.integers(0, buf.n, 4096)]
        return torch.tensor([self.tasks * np.mean(tasks == t) for t in range(self.tasks)], dtype=torch.float32)

    def train_actor(self, buf, rng, reward_matrix, step):
        """One actor step along the aggregated per-objective gradients."""
        params = list(self.actor.parameters())
        losses = self.objective_losses(buf, rng, reward_matrix)
        self.actor_opt.zero_grad(set_to_none=True)
        backward(losses, inputs=params)
        self.updates += 1
        log_now = self.updates % self.log_every == 0
        aggregator = aggregation.Constant(self.task_share_weights(buf)) if self.aggregator is None else self.aggregator
        warmup = step < self.agg_start
        if warmup:
            aggregator = self.warmup_aggregator
        weights = jac_to_grad(params, aggregator, retain_jac=log_now)
        if log_now:
            self.log_row(step, params, weights, 'sum' if warmup else self.agg_name)
        self.actor_opt.step()

    def log_row(self, step, params, weights, agg_name=None):
        """Per-objective gradient norms, pairwise cosines and the weights; checks J^T w = step."""
        self.rows.append([step, agg_name or self.agg_name] + self.gradient_stats(params, weights, self.n_obj))

    @staticmethod
    def gradient_stats(params, weights, m):
        """Norms, weights, pairwise cosines of the m stored gradient rows (.jac, deleted here), step
        norm and the relative error of J^T w against the step in .grad."""
        J = torch.cat([p.jac.reshape(m, -1) for p in params], dim=1).double()
        for p in params:
            del p.jac
        G = J @ J.T
        norm = torch.sqrt(torch.clamp(torch.diag(G), min=0))
        w = weights.detach().double() if weights is not None else torch.full((m,), float('nan'))
        step_vec = torch.cat([p.grad.reshape(-1) for p in params]).double()
        check = float(torch.linalg.norm(J.T @ w - step_vec) / max(float(torch.linalg.norm(step_vec)), 1e-30))
        row = [float(x) for x in norm] + [float(x) for x in w]
        for a in range(m):
            for b in range(a + 1, m):
                row.append(float(G[a, b] / max(float(norm[a] * norm[b]), 1e-30)))
        return row + [float(torch.linalg.norm(step_vec)), check]

    def log_fields(self):
        """CSV header for the rows of log_row."""
        n = self.names
        return (['step', 'aggregator'] + [f'norm_{x}' for x in n] + [f'weight_{x}' for x in n]
                + [f'cos_{n[a]}_{n[b]}' for a in range(len(n)) for b in range(a + 1, len(n))]
                + ['step_norm', 'check_rel_error'])
