"""TD3 in PyTorch, the same algorithm and schedule as the C++ training (plan A4).

Mirrors rl_tools (rl/algorithms/td3/loop.h, operations_generic.h, off_policy_runner) and the
learning-to-fly configuration (src/config/config.h, src/steps/curriculum.h, checkpoint.h):

  every step t        (curriculum and checkpoint first, as in training.h, then)
                      act = clip(actor(obs) + N(0, sigma_explore), -1, 1); one 10 ms env step;
                      store (obs, critic obs, act, reward parts, next obs, next critic obs,
                      terminated, task); new episode after a termination or 500 steps
  t > 15k, t % 10     each critic, on its own batch of 256 and its own target noise:
                      y = r + gamma * min(Q1', Q2')(s', clip(actor'(s') + clip(N(0, 0.2), +-0.5), -1, 1))
                      (no bootstrap where terminated; truncated episodes bootstrap), MSE loss;
                      then Polyak 0.995 on both critic targets
  t > 30k, t % 20     actor on its own batch: maximise Q1(s, actor(s)); Polyak 0.995 on its target
  curriculum          every 10k: mission progress = min(1, t / L2F_MISSION_RAMP); every 100k (not 0,
                      not the last step): action weight x1.4 (cap 1), position x1.2 (cap 40),
                      linear velocity x1.4 (cap 1); from 500k, every 100k: exploration noise, target
                      noise std and clip x0.9
  checkpoints         actor every 100k (step 0 too), in rl_tools' .h5 layout (export.py)

The replay buffer stores the unweighted reward parts u_i. A batch's rewards are parts @ R, where
R (n_parts x outputs) holds the current curriculum weights: R = w as one column for the scalar
critic, so r = sum_i w_i u_i, the same as rl_tools' recalculation of every stored reward at each
curriculum step. A critic with several outputs (moo.py) gets one column per output. Its twin
minimum is taken on the value sum_k f_k Q_k (f = 1 unless moo.py scales the outputs); with one
output and f = 1 this is TD3's min(Q1', Q2'). No observation normalization (as in C++ training).
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from networks import Actor, Critic, make_optimizer


@dataclass
class TD3Config:
    """Settings of the C++ recipe (src/config/config.h, rl_tools defaults)."""
    steps: int = 2_000_001
    batch: int = 256
    gamma: float = 0.99
    critic_interval: int = 10
    actor_interval: int = 20
    warmup_critic: int = 15_000
    warmup_actor: int = 30_000
    polyak: float = 0.995
    explore_std: float = 0.1
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5
    replay_cap: int = 700_000
    step_limit: int = 500
    checkpoint_interval: int = 100_000
    mission_ramp: int = 800_000
    hidden: int = 64
    weight_caps: dict = field(default_factory=lambda: {'action': (1.4, 1.0), 'position': (1.2, 40.0),
                                                       'linear_velocity': (1.4, 1.0)})


class ReplayBuffer:
    """Ring buffer of transitions; rewards stay as unweighted parts until sampled."""

    def __init__(self, cap, obs_dim, obs_dim_p, act_dim, n_parts):
        self.cap, self.n, self.i = cap, 0, 0
        self.obs = np.zeros((cap, obs_dim), np.float32)
        self.obs_p = np.zeros((cap, obs_dim_p), np.float32)
        self.act = np.zeros((cap, act_dim), np.float32)
        self.parts = np.zeros((cap, n_parts), np.float32)
        self.next_obs = np.zeros((cap, obs_dim), np.float32)
        self.next_obs_p = np.zeros((cap, obs_dim_p), np.float32)
        self.terminated = np.zeros((cap,), np.float32)
        self.task = np.zeros((cap,), np.int8)

    def add(self, obs, obs_p, act, parts, next_obs, next_obs_p, terminated, task):
        """Store one transition per environment (arrays with a leading env axis)."""
        for k in range(obs.shape[0]):
            j = self.i
            self.obs[j], self.obs_p[j], self.act[j], self.parts[j] = obs[k], obs_p[k], act[k], parts[k]
            self.next_obs[j], self.next_obs_p[j] = next_obs[k], next_obs_p[k]
            self.terminated[j], self.task[j] = terminated[k], task[k]
            self.i = (self.i + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def indices(self, rng, batch, task=None):
        """Uniform batch indices; with `task`, only that episode type (rejection sampling); None
        if the task has no transitions in the buffer yet."""
        if task is None:
            return rng.integers(0, self.n, batch)
        picked = []
        for _ in range(64):
            cand = rng.integers(0, self.n, 4 * batch)
            picked.extend(cand[self.task[cand] == task].tolist())
            if len(picked) >= batch:
                return np.array(picked[:batch])
        return None

    def batch(self, idx, reward_matrix):
        """Tensors for indices idx; rewards (batch, outputs) = parts @ reward_matrix."""
        t = lambda a: torch.from_numpy(a[idx])
        reward = torch.from_numpy(self.parts[idx] @ reward_matrix.astype(np.float32))
        return t(self.obs), t(self.obs_p), t(self.act), reward, t(self.next_obs), t(self.next_obs_p), t(self.terminated)

    def sample(self, rng, batch, reward_matrix):
        """A uniform batch (see batch())."""
        return self.batch(self.indices(rng, batch), reward_matrix)


def polyak(source, target, keep):
    """target <- keep * target + (1 - keep) * source."""
    with torch.no_grad():
        for p, q in zip(source.parameters(), target.parameters()):
            q.mul_(keep).add_(p, alpha=1 - keep)


class TD3:
    """Actor, twin critics, their targets and optimizers, and the update steps."""

    def __init__(self, cfg, obs_dim, obs_dim_p, act_dim, n_out=1):
        self.cfg = cfg
        self.n_out = n_out
        self.actor = Actor(obs_dim, act_dim, cfg.hidden)
        self.critics = [Critic(obs_dim_p, act_dim, cfg.hidden, n_out) for _ in range(2)]
        self.actor_target = Actor(obs_dim, act_dim, cfg.hidden)
        self.critic_targets = [Critic(obs_dim_p, act_dim, cfg.hidden, n_out) for _ in range(2)]
        self.actor_target.load_state_dict(self.actor.state_dict())
        for c, ct in zip(self.critics, self.critic_targets):
            ct.load_state_dict(c.state_dict())
        self.actor_opt = make_optimizer(self.actor)
        self.critic_opts = [make_optimizer(c) for c in self.critics]
        self.explore_std = cfg.explore_std
        self.target_noise_std = cfg.target_noise_std
        self.target_noise_clip = cfg.target_noise_clip
        self.factors = torch.ones(n_out)  # f_k: output k's share of the value is f_k Q_k

    def act(self, obs, rng):
        """Exploration action for observations (n, obs_dim)."""
        with torch.no_grad():
            a = self.actor(torch.from_numpy(obs)).numpy()
        a = a + rng.normal(0, self.explore_std, a.shape).astype(np.float32)
        return np.clip(a, -1, 1).astype(np.float32)

    def critic_loss(self, q, y):
        """Sum over outputs of the batch-mean squared TD error (TD3's MSE for one output)."""
        return torch.mean((q - y) ** 2) * self.n_out

    def td_target(self, batch, noise=None):
        """TD target per output, from the target twin with the smaller summed value; noise: the
        target-smoothing noise (drawn here if None)."""
        obs, obs_p, act, reward, next_obs, next_obs_p, terminated = batch
        with torch.no_grad():
            if noise is None:
                noise = (torch.randn_like(act) * self.target_noise_std).clamp(-self.target_noise_clip, self.target_noise_clip)
            next_act = (self.actor_target(next_obs) + noise).clamp(-1, 1)
            q1 = self.critic_targets[0](next_obs_p, next_act)
            q2 = self.critic_targets[1](next_obs_p, next_act)
            pick_1 = ((q1 * self.factors).sum(1) < (q2 * self.factors).sum(1))[:, None]
            q_next = torch.where(pick_1, q1, q2)
            return reward + self.cfg.gamma * (1 - terminated[:, None]) * q_next

    def train_critic(self, i, batch, noise=None):
        """One step of critic i towards the target of the twin with the smaller value; noise:
        the target-smoothing noise (drawn here if None)."""
        obs_p, act = batch[1], batch[2]
        y = self.td_target(batch, noise)
        loss = self.critic_loss(self.critics[i](obs_p, act), y)
        self.critic_opts[i].zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opts[i].step()
        return float(loss.detach())

    def train_actor(self, buf, rng, reward_matrix, step):
        """One step of the actor up the first critic's value."""
        obs, obs_p = buf.sample(rng, self.cfg.batch, reward_matrix)[:2]
        return self.actor_step(obs, obs_p)

    def actor_step(self, obs, obs_p):
        """The actor update on one batch of observations."""
        value = (self.critics[0](obs_p, self.actor(obs)) * self.factors).sum(1).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        (-value).backward()
        self.actor_opt.step()
        return float(value.detach())

    def before_critic_training(self, buf, rng, reward_matrix):
        """Hook run once, when critic training starts (moo.py estimates output scales here)."""

    def update(self, t, buf, rng, part_names, weights):
        """The training part of rl_tools' loop step t (after the environment step), with the
        current curriculum weights of the reward parts."""
        cfg = self.cfg
        critic_now = t > cfg.warmup_critic and t % cfg.critic_interval == 0
        actor_now = t > cfg.warmup_actor and t % cfg.actor_interval == 0
        if not (critic_now or actor_now):
            return
        reward_matrix = self.reward_matrix(part_names, weights)
        if critic_now:
            if t - cfg.critic_interval <= cfg.warmup_critic:
                self.before_critic_training(buf, rng, reward_matrix)
                reward_matrix = self.reward_matrix(part_names, weights)
            for i in range(2):
                self.train_critic(i, buf.sample(rng, cfg.batch, reward_matrix))
        if actor_now:
            self.train_actor(buf, rng, reward_matrix, t)
        if critic_now:
            for c, ct in zip(self.critics, self.critic_targets):
                polyak(c, ct, cfg.polyak)
        if actor_now:
            polyak(self.actor, self.actor_target, cfg.polyak)

    def reward_matrix(self, part_names, weights):
        """R for the scalar critic: one column of weights."""
        return np.asarray(weights, np.float32)[:, None]

    def decay_noise(self):
        """Exploration and target-smoothing noise x0.9 (curriculum, from 500k)."""
        self.explore_std *= 0.9
        self.target_noise_std *= 0.9
        self.target_noise_clip *= 0.9
