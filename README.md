# cda_jepa

A JEPA encoder for [gym-continuousDoubleAuction](https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction),
as an installable extension, built on the I-JEPA and LeJEPA authors' code.

A Joint-Embedding Predictive Architecture predicts the latents of masked parts of its input from
the visible parts, scored against an EMA copy of itself, and reconstructs nothing. Here the input
is gym-CDA's order-book observation, and the objective is an auxiliary loss on PPO's encoder, with
an optional action-conditioned world model and an offline pretrainer. It was part of gym-CDA until
it moved here; [docs/migration.md](docs/migration.md) covers what changed.

- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive
  Architecture* (I-JEPA), CVPR 2023. Code: [ChuaCheowHuan/ijepa](https://github.com/ChuaCheowHuan/ijepa).
- Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning Without the
  Heuristics*, 2025. Code: [ChuaCheowHuan/lejepa](https://github.com/ChuaCheowHuan/lejepa).

## Install

gym-CDA is not on PyPI. Install it first, then this package:

```bash
pip install "gym_continuousDoubleAuction @ git+https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction"
pip install "git+https://github.com/ChuaCheowHuan/cda_jepa"
```

pip fetches the two forks (git submodules) itself. Installing changes no run.

## Enable

In the train config:

```json
"extensions_enabled": ["jepa"],
"encoder_type": "jepa",
"encoder_specs": {"jepa": {"mask_axis": "level"}}
```

The encoder's settings stay where gym-CDA's encoder settings live, under `encoder_specs.jepa`.
Every one has a default, so `{}` is enough; [docs/settings.md](docs/settings.md) lists them.
Selecting `jepa` without `"jepa"` in `extensions_enabled` is refused.

| | |
|---|---|
| Tokenisation | either gym-CDA book mode (`grid`, `levels`); `tokenization` `both`, `time` or `level` |
| Learners | any number; the loss is computed on the actor's encoder only |
| Composition | `CDAJEPALearner` subclasses gym-CDA's PPO Learner and keeps the MoE term; other extensions' mixins (CBP) wrap it |
| World model | `"world_model": true` adds `Columns.NEXT_OBS` to the train batch; nothing else does |
| Pretraining | `python -m cda_jepa.pretrain`, then `pretrained_encoder_path` ([docs/pretraining.md](docs/pretraining.md)) |
| Resumes | checkpoints written with gym-CDA's built-in `jepa` resume with this package installed ([docs/migration.md](docs/migration.md)) |

## Read the collapse metric before believing it works

A collapsed JEPA maps every observation to one latent, which makes its prediction loss zero and
reads as success. Watch `jepa_latent_std` (near 1, never near 0) and `jepa_offdiag_cov`
([docs/metrics.md](docs/metrics.md)).

## What comes from the forks

The forks are installed unchanged, byte for byte from pinned commits: I-JEPA's `src` package
(a top-level package named `src`, because that is how the authors' code imports itself) and the
`lejepa` package. Today the encoder uses I-JEPA's `apply_masks` to gather tokens; the LeJEPA
objective and I-JEPA's block masks are planned as opt-in settings. [docs/provenance.md](docs/provenance.md)
says what is used, what is not, and why.

## Documentation

[docs/README.md](docs/README.md) indexes the rest.

## Compatibility

Needs gym-CDA 2.2 or later (`gym_continuousDoubleAuction>=2.2,<3`): 2.2.0 finds encoders through
the `gym_cda.encoders` entry point. [docs/development.md](docs/development.md) lists what this
package uses from gym-CDA.

## License

This package's own code is MIT, the same as gym-CDA, from which it came. The I-JEPA and LeJEPA
code it installs is CC BY-NC 4.0 (non-commercial); see [NOTICE](NOTICE).
