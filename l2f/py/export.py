"""Actor checkpoints in rl_tools' formats (plan A3), so PyTorch-trained actors work with
evaluate_robustness, l2f/firmware/build_firmware.sh and rl_trigger.py unchanged.

  .h5  the layout C++ training writes (src/steps/checkpoint.h): actor/layers/<i>/{weights,biases}/
       {parameters, gradient, gradient_first_order_moment, gradient_second_order_moment} and the
       layer buffers output, pre_activations (batch 256). Only `parameters` carries the actor; the
       rest is written as zeros so rl_tools' loader finds every dataset. Weights are (out, in), as
       in torch.nn.Linear; biases (1, out).
  .h   the firmware's actor header, made from the .h5 by the C++ tool h5_to_header (built with the
       same flags as the checkpoint's variant), so it is exactly what training would have written.
"""
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

BATCH = 256  # rl_tools layer buffers in the training checkpoints


def save_actor_h5(actor, path):
    """Write `actor` (networks.Actor) as an rl_tools actor checkpoint."""
    with h5py.File(path, 'w') as f:
        layers = f.create_group('actor').create_group('layers')
        for i, lin in enumerate(actor.layers):
            g = layers.create_group(str(i))
            for kind, tensor in (('weights', lin.weight), ('biases', lin.bias[None, :])):
                arr = tensor.detach().cpu().numpy().astype(np.float32)
                p = g.create_group(kind)
                p['parameters'] = arr
                for extra in ('gradient', 'gradient_first_order_moment', 'gradient_second_order_moment'):
                    p[extra] = np.zeros_like(arr)
            g['output'] = np.zeros((BATCH, lin.out_features), np.float32)
            g['pre_activations'] = np.zeros((BATCH, lin.out_features), np.float32)


def load_actor_h5(actor, path):
    """Copy the parameters of an rl_tools actor checkpoint into `actor`."""
    with h5py.File(path, 'r') as f, torch.no_grad():
        for i, lin in enumerate(actor.layers):
            g = f['actor']['layers'][str(i)]
            w = torch.from_numpy(np.array(g['weights']['parameters']))
            b = torch.from_numpy(np.array(g['biases']['parameters'])).reshape(-1)
            if w.shape != lin.weight.shape or b.shape != lin.bias.shape:
                raise ValueError(f'layer {i}: checkpoint {tuple(w.shape)} / {tuple(b.shape)}, actor '
                                 f'{tuple(lin.weight.shape)} / {tuple(lin.bias.shape)} (wrong variant?)')
            lin.weight.copy_(w)
            lin.bias.copy_(b)
    return actor


def h5_to_header(h5_path, header_path, tool, name=None):
    """Firmware header from a .h5 checkpoint, with the C++ tool h5_to_header of the right build."""
    cmd = [str(tool), str(h5_path), str(header_path)] + ([name] if name else [])
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return Path(header_path)
