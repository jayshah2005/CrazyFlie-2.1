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
"""
import numpy as np
import torch
from torchjd import aggregation
from torchjd.autojac import backward, jac_to_grad

from td3 import TD3

OBJECTIVES = ['survival', 'position', 'orientation', 'linear_velocity', 'action', 'smoothing',
              'angular_acceleration', 'landing_speed']
GROUP_OF = {'survival': 0, 'position': 1, 'orientation': 2, 'linear_velocity': 3, 'action': 4,
            'action_rate': 5, 'zigzag': 5, 'angular_acceleration': 6, 'landing_speed': 7,
            'time': 1, 'power': 4, 'speed_limit': 7, 'energy': 4}
# parts=3: the groups tracking, smoothness, safety; untested before 2026-11-22
GROUPS3 = ['tracking', 'smoothness', 'safety']
GROUP3_OF = {'position': 0, 'linear_velocity': 0, 'action': 1, 'action_rate': 1, 'zigzag': 1,
             'angular_acceleration': 1, 'survival': 2, 'orientation': 2, 'landing_speed': 2,
             'time': 0, 'power': 1, 'speed_limit': 2, 'energy': 1}
# parts=4 (time-to-target study, 2026-11-22): time, precision, efficiency, safety
GROUPS4 = ['time', 'precision', 'efficiency', 'safety']
GROUP4_OF = {'time': 0, 'position': 1, 'linear_velocity': 1, 'action': 2, 'action_rate': 2, 'zigzag': 2,
             'angular_acceleration': 2, 'power': 2, 'survival': 3, 'orientation': 3, 'landing_speed': 3,
             'speed_limit': 3, 'energy': 2}
# regrouped by measured conflict (2026-11-23, ungrouped run: survival fights the dense costs,
# time agrees with position): --grouping regroup with parts=4 (crash, progress, effort, limits)
# or parts=5 (+ energy per leg as its own objective)
REGROUPS = {4: ['crash', 'progress', 'effort', 'limits'], 5: ['crash', 'progress', 'effort', 'limits', 'energy']}
REGROUP_OF = {4: {'survival': 0, 'position': 1, 'linear_velocity': 1, 'time': 1, 'energy': 1,
                  'action': 2, 'action_rate': 2, 'zigzag': 2, 'angular_acceleration': 2, 'power': 2,
                  'orientation': 3, 'speed_limit': 3, 'landing_speed': 3}}
REGROUP_OF[5] = {**REGROUP_OF[4], 'energy': 4}
TASK_NAMES = ['free_flight', 'takeoff', 'landing', 'circle', 'line', 'eight', 'fast_circle', 'p2p', 'gust', 'waypoint']
AGGREGATORS = ['sum', 'mean', 'upgrad', 'aligned_mtl', 'dualproj', 'pcgrad', 'mgda', 'svd_pga', 'osnmf_pga', 'task_share']


def make_aggregator(name, m):
    """TorchJD aggregator `name` for m objectives (preference vectors all ones)."""
    ones = torch.ones(m)
    makers = {'sum': aggregation.Sum, 'mean': aggregation.Mean,
              'upgrad': lambda: aggregation.UPGrad(pref_vector=ones),
              'aligned_mtl': lambda: aggregation.AlignedMTL(pref_vector=ones),
              'dualproj': lambda: aggregation.DualProj(pref_vector=ones),
              'pcgrad': aggregation.PCGrad,  # added 2026-11-22 (MOO round 2): stays near the sum
              'mgda': aggregation.MGDA, 'svd_pga': aggregation.SVDPGA, 'osnmf_pga': aggregation.OSNMFPGA}
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
                 log_every=50, objectives=None, grouping='default'):
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
        self.agg_name = agg
        if agg == 'task_share' and not tasks:
            raise ValueError('task_share needs tasks')
        self.aggregator = None if agg == 'task_share' else make_aggregator(agg, self.n_obj)
        self.total, self.norm = total, norm
        self.scale = np.ones(k, np.float32)
        self.group_weights = np.ones(k, np.float32)
        self.log_every, self.updates, self.rows = log_every, 0, []
        self.share_rng = np.random.default_rng(54321)
        self.names = [TASK_NAMES[t] for t in range(tasks)] if tasks else \
            (self.objectives or (REGROUPS[k] if grouping == 'regroup' else {8: OBJECTIVES, 4: GROUPS4, 3: GROUPS3, 1: ['total']}[k]))

    def reward_matrix(self, part_names, weights):
        """R for the critic outputs (see td3.py); also sets the value factors f."""
        U, W = group_parts(part_names, weights, self.k, self.objectives, self.grouping)
        self.group_weights = W
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
        weights = jac_to_grad(params, aggregator, retain_jac=log_now)
        if log_now:
            self.log_row(step, params, weights)
        self.actor_opt.step()

    def log_row(self, step, params, weights):
        """Per-objective gradient norms, pairwise cosines and the weights; checks J^T w = step."""
        J = torch.cat([p.jac.reshape(self.n_obj, -1) for p in params], dim=1).double()
        for p in params:
            del p.jac
        G = J @ J.T
        norm = torch.sqrt(torch.clamp(torch.diag(G), min=0))
        w = weights.detach().double() if weights is not None else torch.full((self.n_obj,), float('nan'))
        step_vec = torch.cat([p.grad.reshape(-1) for p in params]).double()
        check = float(torch.linalg.norm(J.T @ w - step_vec) / max(float(torch.linalg.norm(step_vec)), 1e-30))
        row = [step, self.agg_name] + [float(x) for x in norm] + [float(x) for x in w]
        for a in range(self.n_obj):
            for b in range(a + 1, self.n_obj):
                row.append(float(G[a, b] / max(float(norm[a] * norm[b]), 1e-30)))
        row += [float(torch.linalg.norm(step_vec)), check]
        self.rows.append(row)

    def log_fields(self):
        """CSV header for the rows of log_row."""
        n = self.names
        return (['step', 'aggregator'] + [f'norm_{x}' for x in n] + [f'weight_{x}' for x in n]
                + [f'cos_{n[a]}_{n[b]}' for a in range(len(n)) for b in range(a + 1, len(n))]
                + ['step_norm', 'check_rel_error'])
