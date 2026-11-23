"""Check the PyTorch networks and checkpoint export against rl_tools (plan A3).

1. A C++ training checkpoint loaded into the PyTorch actor gives the same actions as
   rl_tools' own evaluation (l2f_sim.evaluate_actor), to float32 rounding.
2. Exported back to .h5, it evaluates exactly like the original in rl_tools.
3. The firmware headers made from both files by h5_to_header carry the same weights as
   the header C++ training wrote for that checkpoint, byte for byte.
4. FAST_TANH's backward pass is rl_tools' 1 - y^2, and the optimizer's weight decay is
   rl_tools' p * wd / 2 per update on weights only.

Usage: python l2f/py/test_networks.py --build l2f/learning_to_fly/build_py_E/src --checkpoint <actor.h5>
"""
import argparse
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for run_log
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_log  # noqa: E402
from export import h5_to_header, load_actor_h5, save_actor_h5  # noqa: E402
from networks import Actor, Critic, fast_tanh, make_optimizer  # noqa: E402

log = run_log.setup(__file__)
MEMORY = re.compile(r'memory\[\] = \{([0-9, ]+)\}')


def header_arrays(path, n=6):
    """The first n byte arrays (layer weights and biases) of a firmware actor header."""
    return [m.group(1).replace(' ', '') for m in MEMORY.finditer(Path(path).read_text())][:n]


def main(args):
    """Run the four checks; exit 1 if any fails."""
    sys.path.insert(0, str(Path(args.build).resolve()))
    import l2f_sim
    info = l2f_sim.info()
    ok = True
    ckpt = Path(args.checkpoint)
    actor = load_actor_h5(Actor(info['obs_dim'], info['action_dim']), ckpt)
    rng = np.random.default_rng(0)
    obs = np.concatenate([rng.uniform(-1, 1, (1000, info['obs_dim'])),
                          rng.normal(0, 3, (1000, info['obs_dim']))]).astype(np.float32)  # some saturate
    ref = l2f_sim.evaluate_actor(str(ckpt), obs)
    with torch.no_grad():
        mine = actor(torch.from_numpy(obs)).numpy()
    diff = float(np.max(np.abs(mine - ref)))
    log.info('1. PyTorch vs rl_tools actor on 2000 observations: max |diff| %.2g', diff)
    if diff > 1e-5:
        log.error('FAIL: the PyTorch actor does not match rl_tools')
        ok = False

    with tempfile.TemporaryDirectory() as tmp:
        exported = Path(tmp) / 'exported.h5'
        save_actor_h5(actor, exported)
        again = l2f_sim.evaluate_actor(str(exported), obs)
        diff2 = float(np.max(np.abs(again - ref)))
        log.info('2. exported .h5 vs original in rl_tools: max |diff| %.2g', diff2)
        if diff2 != 0:
            log.error('FAIL: the exported checkpoint does not evaluate like the original')
            ok = False
        tool = Path(args.build) / 'h5_to_header'
        h_orig = h5_to_header(ckpt, Path(tmp) / 'orig.h', tool)
        h_exp = h5_to_header(exported, Path(tmp) / 'exported.h', tool)
        trained_h = ckpt.with_suffix('.h')
        same = header_arrays(h_orig) == header_arrays(h_exp)
        same_trained = trained_h.exists() and header_arrays(h_exp) == header_arrays(trained_h)
        log.info('3. firmware header weights: from original == from export %s; == training\'s own header %s',
                 same, same_trained if trained_h.exists() else 'no header next to the checkpoint')
        if not same or (trained_h.exists() and not same_trained):
            log.error('FAIL: the firmware header weights differ')
            ok = False

    x = torch.linspace(-4, 4, 81, requires_grad=True)
    y = fast_tanh(x)
    y.sum().backward()
    grad_ok = torch.allclose(x.grad, 1 - y.detach() ** 2)
    edge_ok = abs(float(fast_tanh(torch.tensor(3.0))) - 1.0) < 1e-7
    net = Critic(info['obs_dim_privileged'], info['action_dim'])
    opt = make_optimizer(net)
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    for p in net.parameters():
        p.grad = torch.zeros_like(p)  # no gradient: only the weight decay acts
    opt.step()
    decay_ok = all(torch.allclose(p, before[n] * (1 - 1e-4 / 2) if n.endswith('weight') else before[n], atol=1e-9)
                   for n, p in net.named_parameters())
    log.info('4. FAST_TANH backward = 1 - y^2: %s, fast_tanh(3) = 1: %s, weight decay p * (1 - wd/2) on weights '
             'only: %s', grad_ok, edge_ok, decay_ok)
    if not (grad_ok and edge_ok and decay_ok):
        log.error('FAIL: activation or optimizer differs from rl_tools')
        ok = False
    log.info('all checks passed' if ok else 'checks FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--build', required=True, help='folder holding l2f_sim and h5_to_header')
    parser.add_argument('--checkpoint', required=True, help='an actor .h5 from C++ training of the same variant')
    sys.exit(main(parser.parse_args()))
