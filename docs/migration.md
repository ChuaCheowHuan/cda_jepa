# Moving from gym-CDA's built-in `jepa`

The JEPA encoder was part of gym-CDA (`train/model/encoders/jepa.py`,
`train/model/jepa_learner.py`, `train/pretrain/`, the `encoder_specs.jepa` block and
`cli_defaults.cda_pretrain`) until it moved into this package. This page is what changes.

## A config

| Before (gym-CDA) | Now |
|---|---|
| `"encoder_type": "jepa"` | the same, plus `"extensions_enabled": ["jepa"]` |
| `"encoder_specs": {"jepa": {...}}` | unchanged: same keys, same defaults ([settings.md](settings.md)) |
| `"pretrained_encoder_path": "..."` | unchanged |

gym-CDA refuses `"encoder_type": "jepa"` without `"jepa"` in `extensions_enabled`, and names this
package when it is not installed. Installing it changes no run.

## A checkpoint

A checkpoint stores its module and Learner classes by module path. A gym-CDA without the built-in
encoder keeps `train/model/jepa_learner.py` as a forwarding module, so a checkpoint written with
the built-in `jepa` resumes with this package installed: the classes resolve to this package's,
the weights and optimiser state load as they were, and the encoder fingerprint still says `jepa`.
Without the package the resume is refused with a message naming it.

## A pretrained encoder

The directory format (`encoder.pt`, `encoder_fingerprint.json`, `rl_module/`) is unchanged, and it
stayed in gym-CDA (`train/model/encoders/pretrained.py`). Weights pretrained before the move load.

| Before | Now |
|---|---|
| `python -m gym_continuousDoubleAuction.train.pretrain ...` | `python -m cda_jepa.pretrain ...`, same flags |
| defaults in gym-CDA's `cli_defaults.json` → `cda_pretrain` | `PRETRAIN_DEFAULTS` in `cda_jepa/config.py`, same values |
| `from gym_continuousDoubleAuction.train.pretrain import pretrain` | `from cda_jepa.pretrain import pretrain` |

The probe stays in gym-CDA: `python -m gym_continuousDoubleAuction.train.probe --encoders jepa
--pretrained <dir>` works with this package installed.

## Metrics

The `jepa_*` names are unchanged ([metrics.md](metrics.md)).

## Training is the same

Moving changed one thing in the encoder: tokens are gathered with I-JEPA's `apply_masks` instead of
an inline `torch.gather`, which gives identical tensors. An equivalence test in 0.1.0 showed the two
were the same: from the same state, on the same episodes, this package and gym-CDA's built-in `jepa`
produced identical weights, EMA target, optimiser state and `jepa_*` metrics, with the world model
off and on and the trunk shared or not. It was removed after 0.1.0: it could run only against
gym-CDA 2.2, and later gym-CDA versions have no `jepa` of their own to compare with.
