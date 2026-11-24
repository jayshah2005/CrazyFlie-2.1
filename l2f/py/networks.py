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

    def __init__(self, obs_dim_privileged, action_dim=4, hidden=64, n_out=1):
        super().__init__(obs_dim_privileged + action_dim, n_out, hidden, output_tanh=False)
        self.out_scale = 1.0  # MooTD3 head_scale: the outputs are multiplied by this

    def forward(self, obs_privileged, action):
        """Value of `action` in the critic's (noise-free) observation."""
        q = super().forward(torch.cat([obs_privileged, action], dim=-1))
        return q if self.out_scale == 1.0 else q * self.out_scale


def make_optimizer(module, lr=1e-3, weight_decay=1e-4, eps=1e-8):
    """rl_tools' Adam for `module` (see the module docstring): AdamW with the decoupled decay
    wd / (2 lr) on weight tensors only."""
    weights = [p for name, p in module.named_parameters() if name.endswith('weight')]
    biases = [p for name, p in module.named_parameters() if not name.endswith('weight')]
    return torch.optim.AdamW([{'params': weights, 'weight_decay': weight_decay / (2 * lr)},
                              {'params': biases, 'weight_decay': 0.0}], lr=lr, betas=(0.9, 0.999), eps=eps)
