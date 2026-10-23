"""This package's encoder through a real `Algorithm`: the claims only a built one can make.

Moved from gym-CDA's `test/integration/test_encoder_wiring.py` with the encoder. The
terms are computed several layers from the loss - the encoder stages them,
`JEPARLModule` moves them into `fwd_out`, `CDAJEPALearner` adds them to PPO's total - and
a break anywhere leaves training running normally with an encoder nothing is teaching,
which looks exactly like a working run.

Selecting this package's encoder must swap in *both* its module and its learner class,
and selecting anything else must leave both at gym-CDA's defaults.
"""

import pytest
import ray
import torch

from gym_continuousDoubleAuction.train.model.encoders import (
    ENCODER_LEARNER_CLASSES,
    ENCODER_MODULE_CLASSES,
    learner_class_for,
    module_class_for,
    needs_next_obs,
)
from gym_continuousDoubleAuction.train.model.moe_learner import (
    CDAPPOTorchLearner,
    CDAPPOTorchRLModule,
)
from gym_continuousDoubleAuction.train.policy.policy_handler import trainable_policy_ids
from gym_continuousDoubleAuction.train.train import build_config

from cda_jepa.learner import (
    JEPA_AUX_LOSS_KEY,
    JEPA_LATENT_STD_KEY,
    JEPA_OFFDIAG_COV_KEY,
    JEPA_PREDICT_LOSS_KEY,
    JEPA_WORLD_LOSS_KEY,
    CDAJEPALearner,
    JEPARLModule,
)
from conftest import JEPA, config_with_encoder

OTHER_ENCODERS = ["transformer", "lstm", "moe_transformer"]


def _build_and_train(encoder_specs=None):
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=2)
    cfg = config_with_encoder(JEPA, encoder_specs=encoder_specs)
    ppo_config, _callback = build_config(cfg)
    algo = ppo_config.build_algo()
    pid = trainable_policy_ids(cfg.num_trained_agents)[0]
    return algo, pid


class TestTheAuxLossReachesTheOptimiser:
    @classmethod
    def setup_class(cls):
        cls.algo, cls.pid = _build_and_train()
        cls.before = {
            k: v.detach().clone() for k, v in cls.algo.env_runner.module[cls.pid].named_parameters()
        }
        cls.result = cls.algo.train()

    @classmethod
    def teardown_class(cls):
        cls.algo.stop()
        ray.shutdown()

    def test_the_jepa_learner_is_in_use(self):
        assert self.algo.config.learner_class is CDAJEPALearner

    def test_the_module_class_is_the_jepa_one(self):
        assert isinstance(self.algo.env_runner.module[self.pid], JEPARLModule)

    def test_the_aux_loss_is_logged(self):
        """Absent if the module stopped forwarding it - the learner would then
        add nothing and training would look entirely normal."""
        learner_results = self.result["learners"][self.pid]

        assert JEPA_AUX_LOSS_KEY in learner_results
        assert learner_results[JEPA_AUX_LOSS_KEY] > 0
        assert JEPA_PREDICT_LOSS_KEY in learner_results

    def test_the_collapse_metrics_are_reported(self):
        """A collapsed JEPA drives its prediction loss to zero, which reads as
        success. These two numbers are the only difference."""
        learner_results = self.result["learners"][self.pid]

        assert JEPA_LATENT_STD_KEY in learner_results
        assert JEPA_OFFDIAG_COV_KEY in learner_results
        assert learner_results[JEPA_LATENT_STD_KEY] > 0.1

    def test_the_encoder_is_trained(self):
        after = dict(self.algo.env_runner.module[self.pid].named_parameters())
        changed = {k for k, v in after.items() if not torch.equal(self.before[k], v)}
        assert any("actor_encoder.trunk" in k for k in changed), (
            "the online trunk took no gradient at all"
        )

    def test_the_predictor_is_trained_but_the_target_is_not(self):
        """The target moves by EMA, never by gradient - that asymmetry is the
        anti-collapse mechanism."""
        encoder = self.algo.env_runner.module[self.pid].encoder.actor_encoder

        assert all(not p.requires_grad for p in encoder.target_trunk.parameters())
        assert any(p.requires_grad for p in encoder.predictor.parameters())

    def test_the_training_only_parts_are_declared(self):
        """Or every champion snapshot carries a second encoder it never runs."""
        declared = self.algo.env_runner.module[self.pid].get_non_inference_attributes()

        assert "encoder.actor_encoder.target_trunk" in declared
        assert "encoder.actor_encoder.predictor" in declared


class TestTheWorldModelReachesTheOptimiser:
    """It needs `Columns.NEXT_OBS`, so this also pins gym-CDA's conditional connector."""

    @classmethod
    def setup_class(cls):
        cls.algo, cls.pid = _build_and_train({JEPA: {"world_model": True}})
        cls.result = cls.algo.train()

    @classmethod
    def teardown_class(cls):
        cls.algo.stop()
        ray.shutdown()

    def test_the_world_loss_is_logged(self):
        learner_results = self.result["learners"][self.pid]

        assert JEPA_WORLD_LOSS_KEY in learner_results
        assert learner_results[JEPA_WORLD_LOSS_KEY] > 0

    def test_the_masked_objective_still_runs_alongside_it(self):
        learner_results = self.result["learners"][self.pid]

        assert JEPA_AUX_LOSS_KEY in learner_results
        assert JEPA_LATENT_STD_KEY in learner_results

    def test_the_action_embedding_is_trained(self):
        world = self.algo.env_runner.module[self.pid].encoder.actor_encoder.world_model

        assert world is not None
        assert any(p.requires_grad for p in world.action_embed.parameters())


class TestOtherEncodersAreUnaffected:
    @pytest.mark.parametrize("encoder_type", ["mlp", *OTHER_ENCODERS])
    def test_the_learner_is_still_the_default(self, encoder_type):
        assert learner_class_for(encoder_type, CDAPPOTorchLearner) is CDAPPOTorchLearner

    @pytest.mark.parametrize("encoder_type", OTHER_ENCODERS)
    def test_the_module_class_is_still_the_default(self, encoder_type):
        assert module_class_for(encoder_type, CDAPPOTorchRLModule) is CDAPPOTorchRLModule

    def test_this_package_registers_its_own_classes(self):
        assert module_class_for(JEPA, CDAPPOTorchRLModule) is JEPARLModule
        assert learner_class_for(JEPA, CDAPPOTorchLearner) is CDAJEPALearner
        assert ENCODER_MODULE_CLASSES[JEPA][0] == "cda_jepa.learner"
        assert ENCODER_LEARNER_CLASSES[JEPA][0] == "cda_jepa.learner"


class TestTheNextObsConnectorIsConditional:
    @pytest.mark.parametrize("encoder_type", ["mlp", *OTHER_ENCODERS])
    def test_other_encoders_do_not_ask_for_it(self, encoder_type):
        assert not needs_next_obs(encoder_type, None)

    def test_without_the_world_model_it_is_not_asked_for(self):
        assert not needs_next_obs(JEPA, {"world_model": False})

    def test_with_the_world_model_it_is(self):
        assert needs_next_obs(JEPA, {"world_model": True})

    def test_the_default_is_read_when_the_spec_omits_it(self):
        assert not needs_next_obs(JEPA, {})
        assert not needs_next_obs(JEPA, None)
