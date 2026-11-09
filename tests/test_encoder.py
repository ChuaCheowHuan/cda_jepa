"""The JEPA encoder: the contract every gym-CDA encoder meets, then its own specifics."""

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, ENCODER_OUT

from conftest import JEPA, build_module, sample_batch

from gym_continuousDoubleAuction.config_loader import group
from gym_continuousDoubleAuction.train.model.encoders.obs_layout import ObsLayout
from gym_continuousDoubleAuction.train.model.model_handler import build_trainable_module_spec
from gym_continuousDoubleAuction.train.model.encoders.tokenize import (
    TOKENIZATIONS,
    token_shape,
)

from cda_jepa import encoder as jepa_module
from cda_jepa.encoder import MASK_AXES, sample_mask
from cda_jepa.learner import JEPA_AUX_LOSS, JEPA_WORLD_LOSS, CDAJEPALearner


class TestTheEncoderContract:
    """The contract every gym-CDA encoder meets, for this one."""

    def test_produces_usable_policy_and_value_outputs(self, spaces):
        obs_space, _ = spaces
        module = build_module(spaces, JEPA)
        batch = sample_batch(obs_space, module=module)

        out = module.forward_train(batch)
        values = module.compute_values(batch)

        assert out[Columns.ACTION_DIST_INPUTS].shape[0] == 4
        assert tuple(values.shape) == (4,)
        assert torch.isfinite(out[Columns.ACTION_DIST_INPUTS]).all()
        assert torch.isfinite(values).all()

    def test_inference_forward_works(self, spaces):
        """The rollout path, which is a different method from the train one."""
        obs_space, _ = spaces
        module = build_module(spaces, JEPA)

        out = module.forward_inference(sample_batch(obs_space, module=module))

        assert torch.isfinite(out[Columns.ACTION_DIST_INPUTS]).all()

    def test_state_round_trips(self, spaces):
        """Checkpointing and champion snapshots both depend on this."""
        obs_space, _ = spaces
        source = build_module(spaces, JEPA)
        target = build_module(spaces, JEPA)
        batch = sample_batch(obs_space, module=source)

        target.set_state(source.get_state())

        assert torch.allclose(
            source.forward_train(batch)[Columns.ACTION_DIST_INPUTS],
            target.forward_train(batch)[Columns.ACTION_DIST_INPUTS],
        )
        assert torch.allclose(source.compute_values(batch), target.compute_values(batch))

    def test_non_inference_attributes_extend_the_base_list(self, spaces):
        """`vf` and the critic encoder are training-only for every encoder. This one appends
        its EMA target trunk and predictor, so an inference-only copy and every champion
        snapshot drop them rather than carrying a second encoder they never run. `world_model`
        is off by default, and a submodule that does not exist must not be declared: `setup`
        would then try to delete it."""
        module = build_module(spaces, JEPA, vf_share_layers=False)

        assert module.get_non_inference_attributes() == [
            "vf",
            "encoder.critic_encoder",
            "encoder.actor_encoder.target_trunk",
            "encoder.actor_encoder.predictor",
        ]

    @pytest.mark.parametrize("method", ["forward_inference", "forward_exploration"])
    def test_the_rollout_forward_is_deterministic(self, spaces, method):
        """PPO's ratio compares a rollout log-prob against a recomputed one.

        Any nondeterminism on the rollout path - a live dropout or a mask being the usual
        culprits - shows up as noise in that ratio rather than as an error. Dropout is on so
        the test can fail. (`forward_train` is not the path to test: RLlib puts the module in
        train mode for it.)
        """
        obs_space, _ = spaces
        module = build_module(spaces, JEPA, spec={"dropout": 0.1})
        module.eval()
        batch = sample_batch(obs_space, module=module)
        forward = getattr(module, method)

        first = forward(batch)[Columns.ACTION_DIST_INPUTS]
        second = forward(batch)[Columns.ACTION_DIST_INPUTS]

        assert torch.equal(first, second)

    def test_gradients_reach_the_encoder(self, spaces):
        """An encoder detached from the loss would train silently as a constant."""
        obs_space, _ = spaces
        module = build_module(spaces, JEPA)
        batch = sample_batch(obs_space, module=module)

        module.forward_train(batch)[Columns.ACTION_DIST_INPUTS].sum().backward()

        trunk = module.encoder.actor_encoder.trunk
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trunk.parameters())


# --- JEPA --------------------------------------------------------------------


class TestJEPA:
    """The latent-prediction objective and the collapse metrics.

    What is worth pinning is not that it runs - `TestTheEncoderContract` covers the
    encoder contract for it automatically - but the handful of properties that
    decide whether the objective is doing anything: that the policy latent is
    unaffected by masking, that the loss carries gradient to the online trunk
    and *not* to the EMA target, and that a collapse is visible in a metric
    rather than only in a loss that reads as success.
    """

    def test_the_policy_latent_ignores_the_mask(self, spaces):
        """The agent acts on everything it was given.

        Masking exists only for the auxiliary loss. If it reached the policy
        latent, the agent would be acting on a random subset of the book and
        PPO's ratio would compare log-probs computed under different masks.
        """
        obs_space, _ = spaces
        module = build_module(spaces, JEPA)
        batch = sample_batch(obs_space, n=6, module=module)

        module.train()
        with torch.no_grad():
            first = module.encoder(batch)[ENCODER_OUT][ACTOR]
            second = module.encoder(batch)[ENCODER_OUT][ACTOR]

        # Two training forwards draw two different masks; the latent must not
        # move between them.
        assert torch.allclose(first, second, atol=1e-6)

    def test_the_objective_runs_only_in_train_mode(self, spaces):
        """`eval()` must produce no stats at all.

        Mask sampling is stochastic, so an objective that ran on the inference
        path would put noise into PPO's ratio rather than raise - the failure
        `test_the_rollout_forward_is_deterministic` exists to catch.
        """
        obs_space, _ = spaces
        module = build_module(spaces, JEPA)
        batch = sample_batch(obs_space, n=4, module=module)
        encoder = module.encoder.actor_encoder

        module.eval()
        with torch.no_grad():
            module.encoder(batch)
        assert encoder.take_jepa_stats() is None

        module.train()
        module.encoder(batch)
        assert encoder.take_jepa_stats() is not None

    def test_the_aux_loss_reaches_fwd_out_and_carries_gradient(self, spaces):
        obs_space, _ = spaces
        module = build_module(spaces, JEPA)
        module.train()
        out = module._forward_train(sample_batch(obs_space, n=6, module=module))

        assert JEPA_AUX_LOSS in out
        assert out[JEPA_AUX_LOSS].requires_grad

        out[JEPA_AUX_LOSS].backward()
        trunk = module.encoder.actor_encoder.trunk
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trunk.parameters()), (
            "the online trunk took no gradient from the objective"
        )

    def test_the_target_trunk_takes_no_gradient(self, spaces):
        """It is an EMA copy, not a trained one. If it learned by gradient it
        would stop being a lagging target, and the asymmetry that discourages
        collapse would be gone."""
        obs_space, _ = spaces
        module = build_module(spaces, JEPA)
        module.train()
        out = module._forward_train(sample_batch(obs_space, n=6, module=module))
        out[JEPA_AUX_LOSS].backward()

        target = module.encoder.actor_encoder.target_trunk
        assert all(not p.requires_grad for p in target.parameters())
        assert all(p.grad is None for p in target.parameters())

    def test_the_target_trails_the_online_trunk(self, spaces):
        """One EMA step per training forward, and it must actually move."""
        obs_space, _ = spaces
        module = build_module(spaces, JEPA)
        encoder = module.encoder.actor_encoder

        # Push the online trunk somewhere the target is not.
        with torch.no_grad():
            for parameter in encoder.trunk.parameters():
                parameter.add_(1.0)

        before = next(encoder.target_trunk.parameters()).clone()
        module.train()
        module.encoder(sample_batch(obs_space, n=4, module=module))
        after = next(encoder.target_trunk.parameters())

        assert not torch.equal(before, after), "the target never moved"
        # It trails: one step of decay 0.996 closes 0.4% of a gap of 1.0.
        assert (after - before).abs().max() < 0.1

    def test_stats_are_cleared_when_taken(self, spaces):
        """Taking rather than reading. A stale auxiliary loss silently added to
        a later batch's gradient would be invisible; a None is not."""
        obs_space, _ = spaces
        module = build_module(spaces, JEPA)
        module.train()
        module.encoder(sample_batch(obs_space, n=4, module=module))

        encoder = module.encoder.actor_encoder
        assert encoder.take_jepa_stats() is not None
        assert encoder.take_jepa_stats() is None

    def test_collapse_is_visible_in_latent_std(self, spaces):
        """The metric that exists because the loss cannot tell.

        A collapsed JEPA maps every observation to the same latent, which makes
        the prediction *perfect* - loss near zero, which reads as success.
        `latent_std` goes to zero at the same time, and is the only signal that
        distinguishes the two.

        The collapse is forced on the ONLINE path, because that is what the
        statistic now measures. It used to be read off the no-grad `target`,
        which is docs/29 S2-9: a zeroed *target* trunk made the number fall
        while the thing being trained was perfectly healthy, and - worse - the
        hinge built from it carried no gradient at all.
        """
        obs_space, _ = spaces
        module = build_module(spaces, JEPA, spec={"ema_decay": 1.0})
        encoder = module.encoder.actor_encoder

        # A predictor whose weights are all zero emits the same vector for
        # every input, which is exactly what a collapsed online path does.
        with torch.no_grad():
            for parameter in encoder.predictor.parameters():
                parameter.zero_()

        module.train()
        module.encoder(sample_batch(obs_space, n=8, module=module))
        stats = encoder.take_jepa_stats()

        assert float(stats["latent_std"]) < 0.1, (
            "a collapsed encoder must show a near-zero latent_std"
        )
        # And the hinge pushes back, so the reported aux loss is NOT near zero
        # even though the prediction problem became trivial.
        assert float(stats["aux_loss"].detach()) > float(stats["predict_loss"])

    def test_identical_observations_are_visible_in_latent_std(self, spaces):
        """A constant input is the simplest collapse the metric must catch.

        `latent_std` used to be taken across batch x masked positions, so the
        predictor's position embeddings kept it near 0.9 even when every
        observation mapped to the same vector (#4). Taken across the batch at
        each masked position, repeated observations have no spread at all.
        """
        obs_space, _ = spaces
        module = build_module(spaces, JEPA, spec={"ema_decay": 1.0})
        encoder = module.encoder.actor_encoder
        batch = sample_batch(obs_space, n=8, module=module)
        batch[Columns.OBS] = batch[Columns.OBS][:1].repeat(8, 1)

        module.train()
        module.encoder(batch)
        stats = encoder.take_jepa_stats()

        assert float(stats["latent_std"]) < 0.1

    def test_a_blind_trunk_is_visible_in_latent_std(self, spaces):
        """A trunk that ignores its input is the realistic collapse.

        Zeroing the *predictor* removes its position embeddings too, which is
        why the old test passed while a trunk that still had positional
        variation did not (#4). Here the online trunk's projection is zeroed
        but the predictor is untouched; only spread across observations can
        show the collapse.
        """
        obs_space, _ = spaces
        module = build_module(spaces, JEPA, spec={"ema_decay": 1.0})
        encoder = module.encoder.actor_encoder
        with torch.no_grad():
            encoder.trunk.project.weight.zero_()
            encoder.trunk.project.bias.zero_()

        module.train()
        module.encoder(sample_batch(obs_space, n=8, module=module))
        stats = encoder.take_jepa_stats()

        assert float(stats["latent_std"]) < 0.1

    def test_the_mask_stream_continues_from_a_checkpoint(self, spaces):
        """A resumed run must draw the same next mask as an uninterrupted one.

        `_mask_generator` lived outside the module state (#13), so a restored
        encoder started its stream over instead of continuing it. The state
        buffer is part of the encoder's `state_dict`, and `set_state` carries it.
        """
        obs_space, _ = spaces
        source = build_module(spaces, JEPA)
        source.train()
        source.encoder(sample_batch(obs_space, n=8, module=source))
        assert source.encoder.actor_encoder._mask_state.numel() > 0

        target = build_module(spaces, JEPA)
        target.set_state(source.get_state())

        def next_mask(module):
            encoder = module.encoder.actor_encoder
            return sample_mask(
                encoder._num_tokens,
                encoder.layout,
                encoder.tokenization,
                encoder.mask_axis,
                encoder.mask_ratio,
                generator=encoder._mask_rng(),
            ).tolist()

        assert next_mask(source) == next_mask(target)

    def test_the_objective_only_forward_skips_the_policy_pass(self, spaces, monkeypatch):
        """Pretraining and validation need the objective, not the policy latent."""
        obs_space, _ = spaces
        module = build_module(spaces, JEPA)
        encoder = module.encoder.actor_encoder
        module.train()

        def boom(*args, **kwargs):
            raise AssertionError("the policy pass ran")

        monkeypatch.setattr(encoder.pool, "forward", boom)
        stats = encoder.forward_objective(sample_batch(obs_space, n=8, module=module)[Columns.OBS])

        assert "aux_loss" in stats

    def test_the_variance_hinge_actually_carries_a_gradient(self, monkeypatch, spaces):
        """docs/29 S2-9, isolated so only the hinge can explain the result.

        The hinge was computed from `target`, built under `torch.no_grad()` by
        a trunk with `requires_grad_(False)`. `loss + coeff * penalty` still
        backpropagates - through `loss` - so nothing raised, and the term
        described everywhere as the only active defence against collapse was a
        constant.

        Asserting that "a gradient reaches the trunk" does not catch that: one
        reaches it from the prediction loss either way. So the same batch and
        the same mask are run twice, differing only in `variance_coeff`. If the
        hinge is in the graph the gradients must differ; if it is a constant
        they are identical, because a constant's derivative is zero however
        large the coefficient in front of it.

        `VARIANCE_TARGET` is raised so the hinge is off its `relu` floor on a
        healthy encoder. Collapsing the encoder instead would not do: the
        obvious way is to zero the predictor, and a zeroed predictor has a zero
        Jacobian back to the trunk.
        """
        monkeypatch.setattr(jepa_module, "VARIANCE_TARGET", 100.0)

        obs_space, _ = spaces
        # ema_decay 1.0 freezes the target, so the EMA step `_forward` takes
        # cannot make the second pass differ for a reason of its own.
        module = build_module(spaces, JEPA, spec={"ema_decay": 1.0})
        encoder = module.encoder.actor_encoder
        module.train()
        batch = sample_batch(obs_space, n=8, module=module)

        def gradients_with(coeff):
            encoder.variance_coeff = coeff
            # Masks come from the encoder's own generator, which `torch.manual_seed`
            # does not reset once made; re-seeding it is what makes the two passes
            # see the same mask, so only the hinge can explain a difference.
            encoder._mask_generator = torch.Generator().manual_seed(0)
            module.encoder(batch)
            stats = encoder.take_jepa_stats()
            encoder.zero_grad(set_to_none=True)
            stats["aux_loss"].backward()
            return (
                {
                    name: parameter.grad.clone()
                    for name, parameter in encoder.trunk.named_parameters()
                    if parameter.grad is not None
                },
                float(stats["aux_loss"].detach()),
                float(stats["predict_loss"]),
            )

        without, aux_off, predict_off = gradients_with(0.0)
        with_hinge, aux_on, _predict_on = gradients_with(50.0)

        assert without, "precondition: the prediction loss reaches the trunk"
        assert aux_off == pytest.approx(predict_off, rel=1e-6), (
            "precondition: at coeff 0 the aux loss is the prediction loss"
        )
        assert aux_on > aux_off, "precondition: the hinge is off its floor"

        assert any(not torch.allclose(without[name], with_hinge[name]) for name in without), (
            "the trunk's gradient is unchanged by variance_coeff, so the "
            "variance hinge is a constant again"
        )

    @pytest.mark.parametrize("mask_axis", MASK_AXES)
    def test_every_mask_axis_builds_and_runs(self, spaces, mask_axis):
        obs_space, _ = spaces
        module = build_module(spaces, JEPA, spec={"mask_axis": mask_axis})
        module.train()
        out = module._forward_train(sample_batch(obs_space, n=4, module=module))
        assert torch.isfinite(out[JEPA_AUX_LOSS])

    def test_an_out_of_range_mask_ratio_raises(self, spaces):
        for ratio in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="mask_ratio"):
                build_module(spaces, JEPA, spec={"mask_ratio": ratio})

    def test_unknown_mask_axis_raises(self, spaces):
        with pytest.raises(ValueError, match="mask_axis"):
            build_module(spaces, JEPA, spec={"mask_axis": "diagonal"})

    def test_unknown_spec_key_raises(self, spaces):
        with pytest.raises(ValueError, match="Unknown key"):
            build_module(spaces, JEPA, spec={"mask_rate": 0.5})

    def test_a_dense_encoder_produces_no_jepa_loss(self, spaces):
        """The plumbing is inert for every other architecture."""
        obs_space, _ = spaces
        module = build_module(spaces, "transformer")
        module.train()
        out = module._forward_train(sample_batch(obs_space, n=4, module=module))
        assert JEPA_AUX_LOSS not in out


class TestJEPAWorldModel:
    """`z_hat_{t+1} = P(z_t, a_t)`, the action-conditioned term.

    Off by default, so most of what matters is that turning it on adds a term
    that reaches the loss - and that leaving it off, or running on a batch
    without `Columns.NEXT_OBS`, adds nothing rather than raising. PPO's batch
    carries no NEXT_OBS, so every path except the connector-fed training one
    has to keep working.
    """

    @staticmethod
    def _batch(obs_space, act_space, n=6, with_next=True):
        rng = np.random.default_rng(0)
        obs_space.seed(int(rng.integers(1 << 30)))
        act_space.seed(int(rng.integers(1 << 30)))

        batch = {Columns.OBS: torch.from_numpy(np.stack([obs_space.sample() for _ in range(n)]))}
        if with_next:
            batch[Columns.NEXT_OBS] = torch.from_numpy(
                np.stack([obs_space.sample() for _ in range(n)])
            )
            sampled = [act_space.sample() for _ in range(n)]
            batch[Columns.ACTIONS] = {
                key: torch.from_numpy(np.stack([a[key] for a in sampled]))
                for key in act_space.spaces
            }
        return batch

    def test_the_term_reaches_fwd_out_and_carries_gradient(self, spaces):
        obs_space, act_space = spaces
        module = build_module(spaces, JEPA, spec={"world_model": True})
        module.train()

        out = module._forward_train(self._batch(obs_space, act_space))

        assert JEPA_WORLD_LOSS in out
        assert out[JEPA_WORLD_LOSS].requires_grad

        out[JEPA_WORLD_LOSS].backward()
        world = module.encoder.actor_encoder.world_model
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in world.parameters())

    def test_a_batch_without_next_obs_is_not_an_error(self, spaces):
        """PPO's own batch has none, and `compute_values` runs on one. Raising
        on a key PPO never promised would break every path but training."""
        obs_space, act_space = spaces
        module = build_module(spaces, JEPA, spec={"world_model": True})
        module.train()

        out = module._forward_train(self._batch(obs_space, act_space, with_next=False))

        assert JEPA_WORLD_LOSS not in out
        # The masked objective still ran; only the world-model term is absent.
        assert JEPA_AUX_LOSS in out

    def test_the_target_takes_no_gradient(self, spaces):
        """The world model's target comes from the EMA trunk, same as the
        masked objective's. A gradient reaching it would end the asymmetry."""
        obs_space, act_space = spaces
        module = build_module(spaces, JEPA, spec={"world_model": True})
        module.train()

        out = module._forward_train(self._batch(obs_space, act_space))
        out[JEPA_WORLD_LOSS].backward()

        target = module.encoder.actor_encoder.target_trunk
        assert all(p.grad is None for p in target.parameters())

    def test_the_prediction_depends_on_the_action(self, spaces):
        """Otherwise it is not action-conditioned at all - it would be
        predicting the next book from the current one and ignoring what the
        agent did, which is a different and much less interesting model."""
        obs_space, act_space = spaces
        module = build_module(spaces, JEPA, spec={"world_model": True})
        module.eval()
        world = module.encoder.actor_encoder.world_model

        batch = self._batch(obs_space, act_space, n=4)
        latent = torch.zeros(4, module.encoder.actor_encoder.config.d_model)

        actions = batch[Columns.ACTIONS]
        other = dict(actions)
        # A different category is a different order type entirely.
        other["category"] = (actions["category"] + 1) % int(act_space["category"].n)

        with torch.no_grad():
            assert not torch.allclose(world(latent, actions), world(latent, other))

    def test_the_policy_latent_is_unchanged_by_it(self, spaces):
        """The world model is an auxiliary term, not a change to what the agent
        acts on."""
        obs_space, act_space = spaces
        plain = build_module(spaces, JEPA)
        with_world = build_module(spaces, JEPA, spec={"world_model": True})
        with_world.encoder.actor_encoder.trunk.load_state_dict(
            plain.encoder.actor_encoder.trunk.state_dict()
        )
        with_world.encoder.actor_encoder.pool.load_state_dict(
            plain.encoder.actor_encoder.pool.state_dict()
        )
        if plain.encoder.actor_encoder.private_token is not None:
            with_world.encoder.actor_encoder.private_token.load_state_dict(
                plain.encoder.actor_encoder.private_token.state_dict()
            )

        batch = self._batch(obs_space, act_space, n=4)
        plain.eval()
        with_world.eval()
        with torch.no_grad():
            assert torch.allclose(
                plain.encoder(batch)[ENCODER_OUT][ACTOR],
                with_world.encoder(batch)[ENCODER_OUT][ACTOR],
                atol=1e-6,
            )


class TestJEPAReviewRegressions:
    """Findings from the review of the branch that introduced this encoder.

    Each one built, trained and reported plausible numbers while being wrong,
    which is why they are pinned individually rather than left to the contract
    tests above.
    """

    def test_an_inference_only_module_builds(self, spaces):
        """Champions are inference-only copies, so this failed a `jepa` league
        at its first snapshot.

        Two causes, both silent. `get_non_inference_attributes` read
        `self.encoder` unguarded, and RLlib calls it from `__init__` before
        `setup()` has created it - `RLModuleSpec.build` catches AttributeError
        to fall back to a deprecated constructor, so the real error was
        swallowed and resurfaced as a complaint about `RLModuleConfig`. Then
        RLlib's own stripping loop, for a dotted path whose target *exists*,
        traverses to the leaf and calls `delattr` on the **module** rather than
        the leaf's parent.
        """
        obs_space, act_space = spaces
        spec = build_trainable_module_spec(obs_space, act_space, encoder_type=JEPA)
        spec.inference_only = True

        module = spec.build()
        encoder = module.encoder.actor_encoder
        for part in ("target_trunk", "predictor"):
            assert getattr(encoder, part, None) is None, part

    def test_the_world_model_is_declared_only_when_it_exists(self, spaces):
        """It is an attribute set to None when off, so `hasattr` alone would
        declare a submodule that is not there - and `setup` would then try to
        delete it."""
        off = build_module(spaces, JEPA, vf_share_layers=False)
        on = build_module(spaces, JEPA, spec={"world_model": True}, vf_share_layers=False)

        assert "encoder.actor_encoder.world_model" not in (off.get_non_inference_attributes())
        assert "encoder.actor_encoder.world_model" in (on.get_non_inference_attributes())

    def test_an_inference_only_module_is_smaller(self, spaces):
        """The point of declaring them: a champion should not carry a second
        encoder it never runs."""
        obs_space, act_space = spaces
        full = build_trainable_module_spec(
            obs_space,
            act_space,
            encoder_type=JEPA,
            encoder_specs={JEPA: {"world_model": True}},
        ).build()

        spec = build_trainable_module_spec(
            obs_space,
            act_space,
            encoder_type=JEPA,
            encoder_specs={JEPA: {"world_model": True}},
        )
        spec.inference_only = True
        stripped = spec.build()

        assert sum(p.numel() for p in stripped.parameters()) < 0.5 * sum(
            p.numel() for p in full.parameters()
        )

    def test_only_one_branch_runs_the_objective(self, spaces):
        """`vf_share_layers` is false by default, so the critic gets its own
        encoder - and only the actor's stats are ever collected. The critic's
        copy would run a mask pass, a target pass, the predictor and the world
        model on every training forward, have all of it discarded, and hold the
        autograd graph until the next forward overwrote it."""
        obs_space, _ = spaces
        module = build_module(spaces, JEPA, vf_share_layers=False)
        module.train()
        module._forward_train(sample_batch(obs_space, n=4, module=module))

        assert module.encoder.actor_encoder.objective_enabled
        assert not module.encoder.critic_encoder.objective_enabled
        assert module.encoder.critic_encoder._jepa_stats is None

    def test_evaluation_does_not_step_the_ema(self, spaces):
        """The objective only exists in train mode, so evaluating it runs in
        train mode too - under `no_grad`, with no optimiser step to follow. An
        EMA step there makes the trained weights a function of how often
        validation ran and how much data it covered."""
        obs_space, _ = spaces
        module = build_module(spaces, JEPA)
        encoder = module.encoder.actor_encoder
        module.train()
        batch = sample_batch(obs_space, n=4, module=module)

        before = next(encoder.target_trunk.parameters()).clone()
        with torch.no_grad():
            for _ in range(5):
                encoder(batch)
        assert torch.equal(before, next(encoder.target_trunk.parameters()))

        # The target starts as a copy of the online trunk, so an EMA step can
        # only be visible after the online trunk has moved.
        with torch.no_grad():
            for parameter in encoder.trunk.parameters():
                parameter.add_(1.0)
        encoder(batch)
        assert not torch.equal(before, next(encoder.target_trunk.parameters()))

    @pytest.mark.parametrize("mask_axis", MASK_AXES)
    @pytest.mark.parametrize("n_hist", [1, 2, 3, "shipped"])
    def test_a_mask_never_hides_everything_or_nothing(self, layout_for_spaces, mask_axis, n_hist):
        """An empty mask leaves the objective nothing to predict; a full one leaves the
        context encoder no input. Both are silent degeneracies.

        `time` masking once clamped against `n_hist - 1`, which bounds SNAPSHOTS and not
        tokens: at `n_hist` 1 it masked the whole sequence. `n_hist: 1` is a setting the
        `lstm` block explicitly recommends, so it was reachable."""
        if n_hist == "shipped":
            layout = layout_for_spaces
        else:
            layout = ObsLayout(n_hist=n_hist, book_rows=4, k_rows=10, extra_dim=2, private_dim=9)
        generator = torch.Generator().manual_seed(0)
        for tokenization in TOKENIZATIONS:
            total = token_shape(layout, tokenization)[0]
            if total < 2:
                continue  # refused at construction; see the next test
            for ratio in (0.01, 0.5, 0.99):
                masked = sample_mask(total, layout, tokenization, mask_axis, ratio, generator)
                case = (tokenization, mask_axis, ratio, len(masked), total)
                assert 0 < len(masked) < total, case
                assert len(set(masked.tolist())) == len(masked), ("duplicate indices", case)

    def test_a_single_token_sequence_is_refused(self, spaces):
        """One token cannot be both hidden and visible, so the contract
        `sample_mask` documents is unsatisfiable there. Refuse it at build
        rather than return a degenerate mask."""
        import gymnasium as gym

        _obs_space, act_space = spaces
        layout = group("tunable_constants.json", "observation_layout")
        snapshot = layout["book_rows"] * layout["k_rows"] + layout["extra_dim"]
        one_frame = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(snapshot + layout["private_dim"],),
            dtype=np.float32,
        )

        with pytest.raises(ValueError, match="at least two"):
            build_trainable_module_spec(
                one_frame,
                act_space,
                encoder_type=JEPA,
                encoder_specs={JEPA: {"tokenization": "time"}},
            ).build()


class TestMasksFollowTheirAxis:
    """The mask is structural wherever the tokenisation has that axis (#9)."""

    def test_level_masking_under_level_tokenisation_is_a_contiguous_band(self, layout_for_spaces):
        """`level` tokenisation has `k_rows` level tokens plus a global one.

        The old branch fell back to a random draw for every tokenisation but
        `both`, so a `(tokenization x mask_axis)` ablation reported a random
        mask under the name "level".
        """
        layout = layout_for_spaces
        total = layout.k_rows + 1
        masked = sample_mask(
            total,
            layout,
            "level",
            "level",
            0.4,
            torch.Generator().manual_seed(0),
        ).tolist()

        assert layout.k_rows not in masked, "the global token must stay visible"
        assert 0 < len(masked) < layout.k_rows
        assert masked == list(range(masked[0], masked[-1] + 1)), "the band is not contiguous"

    def test_a_missing_axis_still_falls_back_to_a_random_mask(self, layout_for_spaces):
        """`time` tokenisation has no level axis; asking for one is not an error."""
        layout = layout_for_spaces
        total = layout.n_hist
        masked = sample_mask(
            total,
            layout,
            "time",
            "level",
            0.4,
            torch.Generator().manual_seed(0),
        )
        assert 0 < len(masked) < total


class TestTheLearnerReadsThroughADDPWrapper:
    """`num_learners > 1` wraps each module; `.encoder` lives on the inner one (#6)."""

    def test_the_encoder_setting_is_found_on_the_unwrapped_module(self, spaces):
        module = build_module(spaces, JEPA)

        class Wrapper:
            """The part of `TorchDDPRLModule` that matters here: no `.encoder`."""

            def __init__(self, inner):
                self.module = inner

            def unwrapped(self):
                return self.module

        # `Learner.module` is a read-only property, so a stand-in subclass owns
        # the mapping instead of assigning it on an instance.
        Learner = type("Learner", (CDAJEPALearner,), {"module": {"p": Wrapper(module)}})
        learner = Learner.__new__(Learner)

        assert (
            learner._encoder_setting("p", "aux_loss_coeff")
            == jepa_module.JEPA_DEFAULTS["aux_loss_coeff"]
        )
