"""Defaults for the offline pretrainer's command line (`python -m cda_jepa.pretrain`).

They were the `cda_pretrain` block of gym-CDA's `config/cli_defaults.json`, with the
same values. The encoder's own hyperparameters are not here: they come from the run's
`encoder_specs.jepa` block (defaults in `encoder.JEPA_DEFAULTS`), so a pretrained
encoder is built from the same numbers the training run will use.
"""

from typing import Any, Dict

PRETRAIN_DEFAULTS: Dict[str, Any] = {
    # Where the weights, the fingerprint and rl_module/ are written.
    "out_dir": "pretrained/jepa",
    "encoder_type": "jepa",
    "steps": 300,
    "batch_size": 128,
    "lr": 0.0003,
    "seed": 0,
    "log_every": 50,
    # The random-agent rollouts used when --parquet is not given. A book made by
    # random agents does not look like one made by a trained league: prefer a run's
    # episode record once one exists.
    "num_episodes": 6,
    "max_step": 512,
    "num_agents": 4,
    "log_level": None,
}
