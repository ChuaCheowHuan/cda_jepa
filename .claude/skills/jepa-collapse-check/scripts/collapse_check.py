"""Measure `latent_std` and the variance hinge on inputs whose answer is known.

Three batches through a freshly built `jepa` module, each with the same mask:

  random     distinct observations: a healthy encoder, the reference
  identical  one observation repeated: nothing differs between rows
  blind      distinct observations, but the online trunk ignores its input

A metric that detects collapse must read far lower on `identical` and `blind` than on
`random`. `--check` exits 1 when it does not (issue #4).

    python .claude/skills/jepa-collapse-check/scripts/collapse_check.py
    python .claude/skills/jepa-collapse-check/scripts/collapse_check.py --check
"""

import argparse
import sys

import numpy as np
import torch

from gym_continuousDoubleAuction.envs.continuousDoubleAuction_env import (
    continuousDoubleAuctionEnv,
)
from gym_continuousDoubleAuction.train.model.encoders import ENCODER_REGISTRY, ensure_registered
from gym_continuousDoubleAuction.train.model.model_handler import build_trainable_module_spec

from cda_jepa import encoder as jepa_module

#: A collapse reading must be at most this fraction of the healthy one.
RATIO = 0.5


def _encoder_type():
    """`jepa`, or this package's encoder under a private name on a gym-CDA that ships its own."""
    ensure_registered("jepa")
    if ENCODER_REGISTRY["jepa"].__module__ == jepa_module.__name__:
        return "jepa"
    if "_jepa_check" not in ENCODER_REGISTRY:
        jepa_module.register_encoder("_jepa_check", extension=None)
    return "_jepa_check"


def _stats(obs, blind, seed):
    env = continuousDoubleAuctionEnv({})
    agent = env.agents[0]
    torch.manual_seed(seed)
    module = build_trainable_module_spec(
        env.get_observation_space(agent), env.get_action_space(agent), encoder_type=_encoder_type()
    ).build()
    encoder = module.encoder.actor_encoder
    if blind:
        with torch.no_grad():
            encoder.trunk.project.weight.zero_()
            encoder.trunk.project.bias.zero_()
    # The same mask in every case, so only the input differs.
    encoder._mask_generator = torch.Generator().manual_seed(seed)
    module.train()
    with torch.no_grad():
        encoder({"obs": torch.from_numpy(obs)})
    stats = encoder.take_jepa_stats()
    hinge = float(stats["aux_loss"] - stats["predict_loss"]) / max(encoder.variance_coeff, 1e-12)
    return float(stats["latent_std"]), float(stats["predict_loss"]), hinge


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="exit 1 if a collapse reads healthy")
    args = parser.parse_args(argv)

    env = continuousDoubleAuctionEnv({})
    space = env.get_observation_space(env.agents[0])
    space.seed(args.seed)
    distinct = np.stack([space.sample() for _ in range(args.batch)])
    repeated = np.repeat(distinct[:1], args.batch, axis=0)

    rows = {
        "random": _stats(distinct, blind=False, seed=args.seed),
        "identical": _stats(repeated, blind=False, seed=args.seed),
        "blind": _stats(distinct, blind=True, seed=args.seed),
    }
    print(f"{'input':<10} {'latent_std':>10} {'predict':>8} {'hinge':>7}")
    for name, (std, predict, hinge) in rows.items():
        print(f"{name:<10} {std:>10.3f} {predict:>8.3f} {hinge:>7.3f}")

    healthy = rows["random"][0]
    missed = [n for n in ("identical", "blind") if rows[n][0] > RATIO * healthy]
    if missed:
        print(f"collapse NOT detected for: {', '.join(missed)} (> {RATIO} x random)")
    else:
        print("collapse detected in both cases")
    return 1 if args.check and missed else 0


if __name__ == "__main__":
    sys.exit(main())
