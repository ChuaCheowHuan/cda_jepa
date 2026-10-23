"""Shared helpers: the env's spaces, built modules, and the name this package's encoder runs under.

`JEPA` is the `encoder_type` the tests build. On a gym-CDA that no longer ships its own
`jepa` it is `"jepa"`, served by this package. On gym-CDA 2.2, which still does, this
package's encoder is registered beside it as the test fixture `_jepa_ext`, so every test
here exercises the package's code and never gym-CDA's copy, and `test_equivalence.py` can
compare the two.
"""

import numpy as np
import pytest
import torch
import tree
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.columns import Columns

from gym_continuousDoubleAuction.envs.continuousDoubleAuction_env import (
    continuousDoubleAuctionEnv,
)
from gym_continuousDoubleAuction.train.model.encoders import ENCODER_REGISTRY, ensure_registered
from gym_continuousDoubleAuction.train.model.encoders.obs_layout import ObsLayout
from gym_continuousDoubleAuction.train.model.model_handler import build_trainable_module_spec

from cda_jepa import encoder as jepa_module

#: The fixture name this package's encoder takes when gym-CDA still ships its own.
FIXTURE = "_jepa_ext"

ensure_registered("jepa")
#: Whether the `jepa` gym-CDA resolves is gym-CDA's own (2.2) rather than this package's.
GYM_CDA_HAS_JEPA = ENCODER_REGISTRY["jepa"].__module__ != jepa_module.__name__
if GYM_CDA_HAS_JEPA and FIXTURE not in ENCODER_REGISTRY:
    jepa_module.register_encoder(FIXTURE, extension=None)

#: The `encoder_type` that builds this package's encoder.
JEPA = FIXTURE if GYM_CDA_HAS_JEPA else "jepa"

#: Time steps per sequence in a stateful forward batch.
SEQ_LEN = 3

#: A training config small enough for a test iteration.
TEST_CFG = dict(
    num_agents=4,
    num_trained_agents=2,
    max_step=64,
    num_episodes_per_iter=2,
    num_epochs=1,
    minibatch_size=64,
    num_env_runners=0,
    num_learners=0,
    num_gpus_per_learner=0,
    episode_data_dir=None,
    log_level="ERROR",
)


def config_with_encoder(encoder_type, encoder_specs=None, **overrides):
    """A TrainConfig for `encoder_type`, the fixture name included.

    `TrainConfig.__post_init__` refuses a fixture, so it is assigned after
    construction. `"jepa"` from this package needs its extension enabled.
    """
    from gym_continuousDoubleAuction.train.train import TrainConfig

    if encoder_type == "jepa" and not GYM_CDA_HAS_JEPA:
        overrides.setdefault("extensions_enabled", ["jepa"])
    cfg = TrainConfig(**{**TEST_CFG, **overrides})
    cfg.encoder_type = encoder_type
    if encoder_specs is not None:
        cfg.encoder_specs = encoder_specs
    return cfg


@pytest.fixture(scope="module")
def spaces():
    env = continuousDoubleAuctionEnv({})
    agent_id = env.agents[0]
    return env.get_observation_space(agent_id), env.get_action_space(agent_id)


@pytest.fixture(scope="module")
def layout_for_spaces(spaces):
    """The shipped `ObsLayout`, for tests that work on the grid directly."""
    obs_space, _ = spaces
    return ObsLayout.from_obs_space(obs_space)


def build_module(spaces, encoder_type, spec=None, **kwargs):
    obs_space, act_space = spaces
    module_spec = build_trainable_module_spec(
        obs_space,
        act_space,
        encoder_type=encoder_type,
        encoder_specs={encoder_type: spec} if spec else {},
        **kwargs,
    )
    if module_spec.module_class is None:
        module_spec.module_class = DefaultPPOTorchRLModule
    return module_spec.build()


def sample_batch(obs_space, n=4, seed=0, module=None, seq_len=SEQ_LEN):
    """A forward batch shaped for `module`, stateful or not."""
    rng = np.random.default_rng(seed)
    obs_space.seed(int(rng.integers(1 << 30)))

    if module is None or not module.is_stateful():
        obs = np.stack([obs_space.sample() for _ in range(n)])
        return {Columns.OBS: torch.from_numpy(obs)}

    obs = np.stack([[obs_space.sample() for _ in range(seq_len)] for _ in range(n)])
    state_in = tree.map_structure(
        lambda s: s.unsqueeze(0).expand(n, *s.shape).contiguous(),
        module.get_initial_state(),
    )
    return {Columns.OBS: torch.from_numpy(obs), Columns.STATE_IN: state_in}
