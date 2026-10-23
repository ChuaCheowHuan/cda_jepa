"""Checks of this encoder that lived in other gym-CDA test files until it moved.

- #242 (gym-CDA `test_s4_minor_fixes.py`): the unbiased std of a single row is NaN, and
  the NaN reached the auxiliary loss and every statistic beside it.
- One pooling definition (`test_s4_hardening.py`): the encoder uses gym-CDA's
  tokeniser tuple rather than a copy.
- #187 (`test_obs_layout_book_mode.py`): pretraining builds the encoder in the corpus's
  book mode, at a width both modes fit.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from gym_continuousDoubleAuction.envs.continuousDoubleAuction_env import (
    continuousDoubleAuctionEnv,
)
from gym_continuousDoubleAuction.train.model.encoders import tokenize
from gym_continuousDoubleAuction.train.model.encoders.obs_layout import ObsLayout
from gym_continuousDoubleAuction.train.probe import __main__ as probe_main
from gym_continuousDoubleAuction.train.probe.corpus import ProbeCorpus

from cda_jepa import encoder as jepa_module
from cda_jepa.pretrain import pretrain
from cda_jepa.pretrain.pretrain import _evaluate
from conftest import JEPA, build_module, sample_batch


class TestVarianceTermWithOneRow:
    @staticmethod
    def _stats(batch):
        env = continuousDoubleAuctionEnv({})
        agent = env.agents[0]
        spaces = (env.get_observation_space(agent), env.get_action_space(agent))
        # One token hidden per observation: with a batch of one that is one row.
        module = build_module(spaces, JEPA, spec={"mask_ratio": 0.01, "mask_axis": "random"})
        module.train()
        torch.manual_seed(0)
        module.encoder(sample_batch(spaces[0], n=batch, module=module))
        return module.encoder.actor_encoder.take_jepa_stats()

    def test_one_row_gives_a_finite_aux_loss(self):
        stats = self._stats(1)
        for key in ("aux_loss", "predict_loss", "offdiag_cov"):
            assert torch.isfinite(stats[key]), key

    def test_one_row_reports_the_spread_as_not_measured(self):
        assert torch.isnan(self._stats(1)["latent_std"])

    def test_one_row_aux_loss_is_just_the_prediction_loss(self):
        stats = self._stats(1)
        assert float(stats["aux_loss"]) == pytest.approx(float(stats["predict_loss"]))

    def test_two_rows_are_unchanged(self):
        stats = self._stats(2)
        assert all(torch.isfinite(v) for v in stats.values())
        assert float(stats["latent_std"]) > 0

    def test_a_nan_chunk_does_not_poison_the_pretraining_average(self):
        stats = iter(
            [
                {"predict_loss": torch.tensor(0.5), "latent_std": torch.tensor(0.8)},
                {"predict_loss": torch.tensor(0.5), "latent_std": torch.tensor(float("nan"))},
            ]
        )

        class Encoder:
            def __call__(self, batch):
                return None

            def take_jepa_stats(self):
                return next(stats)

        class Corpus:
            obs = np.zeros((3, 4), dtype=np.float32)

        out = _evaluate(Encoder(), None, np.arange(3), Corpus(), batch_size=2)
        assert out["latent_std"] == pytest.approx(0.8)
        assert out["loss"] == pytest.approx(0.5)


def test_the_encoder_shares_the_tokenizers_poolings():
    assert jepa_module.POOLINGS is tokenize.POOLINGS


#: 11 x 48 + 41 (grid) == 8 x 66 + 41 (levels)
AMBIGUOUS = 569


def test_pretraining_builds_the_encoder_in_the_corpus_mode():
    space = gym.spaces.Box(-np.inf, np.inf, shape=(AMBIGUOUS,), dtype=np.float32)
    layout = ObsLayout.from_obs_space(space, book_mode="levels")
    rng = np.random.default_rng(0)
    corpus = ProbeCorpus(
        obs=rng.normal(size=(24, AMBIGUOUS)).astype(np.float32),
        episode_index=np.repeat(np.arange(4), 6).astype(np.int32),
        layout=layout,
    )
    env = continuousDoubleAuctionEnv({**probe_main.env_config_for(corpus), "num_of_agents": 2})
    agent = env.agents[0]

    module, _report = pretrain(
        corpus,
        env.get_observation_space(agent),
        env.get_action_space(agent),
        encoder_type=JEPA,
        steps=2,
        batch_size=8,
    )

    assert module is not None
