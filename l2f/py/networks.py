"""PyTorch actor, critic and optimizer matching the C++ training (plan A3).

Same as src/config/actor_and_critic.h and rl_tools:
  - actor: obs -> 64 -> 64 -> 4, FAST_TANH on every layer (the output too)
  - critic: (critic obs, action) -> 64 -> 64 -> outputs (1, or one per reward part), FAST_TANH
    hidden layers, linear output; TD3 uses two
  - FAST_TANH (rl_tools math::fast_tanh): x clipped to [-3, 3], then x (27 + x^2) / (27 + 9 x^2).
    Its backward pass in rl_tools is the tanh-style 1 - y^2 (nn/activation_functions.h), not the
    derivative of that rational function; FastTanh below uses the same.
  - initialization: rl_tools' init_kaiming is PyTorch's default nn.Linear initialization
    (uniform, bound 1 / sqrt(fan_in) for weights and biases)
  - Adam (nn/optimizers/adam DefaultParameters): lr 1e-3, betas 0.9 / 0.999, eps 1e-8 added after the square root of
    the bias-corrected second moment, as PyTorch does. Weight decay 1e-4 is decoupled, on weights
    only, and does not scale with the learning rate: p <- p - lr * adam_step - p * wd / 2. AdamW
    decays p <- p (1 - lr * wd'), so the same decay is wd' = wd / (2 lr) on the weight tensors.
"""
import torch
from torch import nn


class FastTanhFunction(torch.autograd.Function):
    """rl_tools FAST_TANH with rl_tools' backward pass, dy/dx = 1 - y^2."""

    @staticmethod
    def forward(ctx, x):
        """y = x (27 + x^2) / (27 + 9 x^2), with x clipped to [-3, 3]."""
        x = x.clamp(-3.0, 3.0)
        x2 = x * x
        y = x * (27 + x2) / (27 + 9 * x2)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad):
        """grad * (1 - y^2), as rl_tools' d_activation_d_x (0 where |x| >= 3, since y = +-1)."""
        (y,) = ctx.saved_tensors
        return grad * (1 - y * y)


def fast_tanh(x):
    """rl_tools FAST_TANH (see FastTanhFunction)."""
    return FastTanhFunction.apply(x)


class MLP(nn.Module):
    """Three dense layers, FAST_TANH between them; the output activation is FAST_TANH or none."""

    def __init__(self, n_in, n_out, hidden=64, output_tanh=False):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(n_in, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, n_out)])
        self.output_tanh = output_tanh

    def forward(self, x):
        """Apply the three layers."""
        x = fast_tanh(self.layers[0](x))
        x = fast_tanh(self.layers[1](x))
        x = self.layers[2](x)
        return fast_tanh(x) if self.output_tanh else x


class Actor(MLP):
    """obs -> motor commands in [-1, 1]."""

    def __init__(self, obs_dim, action_dim=4, hidden=64):
        super().__init__(obs_dim, action_dim, hidden, output_tanh=True)


class Critic(MLP):
    """(critic obs, action) -> value(s); n_out = 1, or one per reward part (MOO)."""

    def __init__(self, obs_dim_privileged, action_dim=4, hidden=64, n_out=1, layer_norm=False):
        super().__init__(obs_dim_privileged + action_dim, n_out, hidden, output_tanh=False)
        self.out_scale = 1.0  # MooTD3 head_scale: the outputs are multiplied by this
        # layer_norm (MooTD3 --critic-layer-norm, 2026-11-27): LayerNorm on each hidden layer's
        # pre-activation, before FAST_TANH, so at most ~1/9 of the units sit in the clipped region
        # (|x| >= 3, zero gradient); the shared critic had 39% of its layer-2 units dead at 2M
        self.norms = nn.ModuleList([nn.LayerNorm(hidden), nn.LayerNorm(hidden)]) if layer_norm else None

    def forward(self, obs_privileged, action):
        """Value of `action` in the critic's (noise-free) observation."""
        x = torch.cat([obs_privileged, action], dim=-1)
        if self.norms is None:
            q = super().forward(x)
        else:
            x = fast_tanh(self.norms[0](self.layers[0](x)))
            x = fast_tanh(self.norms[1](self.layers[1](x)))
            q = self.layers[2](x)
        return q if self.out_scale == 1.0 else q * self.out_scale


class SplitCritic(nn.Module):
    """k separate critics (critic obs, action) -> 64 -> 64 -> 1, one per reward group, stacked into
    (batch, k) values: no shared hidden layers, so one group's error cannot reshape another group's
    features (MooTD3 --split-critic; the shared 4-head critic with head scaling collapsed the group
    values into copies of one, 2026-11-24 snapshots)."""

    def __init__(self, obs_dim_privileged, action_dim=4, hidden=64, n_out=1):
        super().__init__()
        self.nets = nn.ModuleList([Critic(obs_dim_privileged, action_dim, hidden, 1) for _ in range(n_out)])

    def forward(self, obs_privileged, action):
        """Values of `action`, one column per group."""
        return torch.cat([net(obs_privileged, action) for net in self.nets], dim=-1)


class TowerCritic(nn.Module):
    """Partly shared critic: (critic obs, action) -> 64 shared -> per group 64 -> 1, stacked into
    (batch, k) values. The first layer (common state features) is shared; each group has its own
    second layer and output, so one group's error cannot reshape the features another group's head
    reads (MooTD3 --critic-towers; 2026-11-26, between the shared and the split critic)."""

    def __init__(self, obs_dim_privileged, action_dim=4, hidden=64, n_out=1):
        super().__init__()
        self.shared = nn.Linear(obs_dim_privileged + action_dim, hidden)
        self.towers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_out)])
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_out)])

    def forward(self, obs_privileged, action):
        """Values of `action`, one column per group."""
        h = fast_tanh(self.shared(torch.cat([obs_privileged, action], dim=-1)))
        return torch.cat([head(fast_tanh(tower(h))) for tower, head in zip(self.towers, self.heads)], dim=-1)


def make_optimizer(module, lr=1e-3, weight_decay=1e-4, eps=1e-8):
    """rl_tools' Adam for `module` (see the module docstring): AdamW with the decoupled decay
    wd / (2 lr) on weight tensors only."""
    weights = [p for name, p in module.named_parameters() if name.endswith('weight')]
    biases = [p for name, p in module.named_parameters() if not name.endswith('weight')]
    return torch.optim.AdamW([{'params': weights, 'weight_decay': weight_decay / (2 * lr)},
                              {'params': biases, 'weight_decay': 0.0}], lr=lr, betas=(0.9, 0.999), eps=eps)
