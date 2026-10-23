"""The gate for removing JEPA from gym-CDA: this package must train identically.

Runs only against a gym-CDA that still ships its own `jepa`, so that both paths exist in
one process: gym-CDA's encoder under `jepa` and this package's under the fixture name
`_jepa_ext` (`conftest.py`).

A whole `algo.train()` is not reproducible run to run, so the comparison is made at the
Learner update, as cda_cbp's was. Two algorithms are built, one per encoder. The
second learner is given the first's exact state (weights, EMA target, optimiser), and
both are updated on the same sampled episodes, with every RNG and every encoder's own
mask generator reseeded before each update, for several rounds. Weights, the EMA target,
optimiser state and every logged `jepa_*` metric must match exactly. A control with a
different `mask_ratio` must differ, which shows the comparison can fail.

Against a gym-CDA without its own `jepa` the module is skipped: there is nothing left
to compare with.
"""

import copy
import random

import numpy as np
import pytest
import ray
import torch

from gym_continuousDoubleAuction.train.config import TrainConfig
from gym_continuousDoubleAuction.train.train import build_config

from conftest import FIXTURE, GYM_CDA_HAS_JEPA, TEST_CFG

pytestmark = pytest.mark.skipif(
    not GYM_CDA_HAS_JEPA, reason="this gym-CDA ships no jepa of its own to compare against"
)

ROUNDS = 3
MODULES = ("policy_0", "policy_1")

#: (case id, encoder settings, vf_share_layers)
CASES = [
    ("default", {}, False),
    ("shared-trunk", {}, True),
    ("world-model", {"world_model": True}, False),
    ("world-model-shared", {"world_model": True}, True),
    ("time-mask", {"mask_axis": "time", "tokenization": "both"}, False),
]


def _build(encoder_type, spec, vf_share_layers):
    cfg = TrainConfig(**TEST_CFG, vf_share_layers=vf_share_layers)
    # Assigned after construction: `TrainConfig` refuses a fixture name.
    cfg.encoder_type = encoder_type
    cfg.encoder_specs = {encoder_type: spec}
    ppo_config, _cb = build_config(cfg)
    return ppo_config.build_algo()


def _reseed(algo):
    """Reset every RNG an update draws from, the same way for both algorithms.

    The global ones, and each encoder's own mask generator: that is seeded per encoder
    from a counter in its module, so the two would otherwise draw different masks.
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    learner = algo.learner_group._learner
    for m in MODULES:
        for i, (_name, sub) in enumerate(learner.module[m].named_modules()):
            if hasattr(sub, "_mask_rng"):
                sub._mask_generator = torch.Generator().manual_seed(1000 + i)


def _update(algo, episodes):
    _reseed(algo)
    results = algo.learner_group.update(
        episodes=copy.deepcopy(episodes),
        timesteps={},
        num_epochs=algo.config.num_epochs,
        minibatch_size=algo.config.minibatch_size,
        shuffle_batch_per_epoch=algo.config.shuffle_batch_per_epoch,
    )
    result = results[0] if isinstance(results, list) else results
    return {
        m: {k: float(v.peek()) for k, v in result[m].items() if k.startswith("jepa_")}
        for m in MODULES
    }


def _snapshot(learner):
    weights = {
        m: {k: v.detach().clone() for k, v in learner.module[m].state_dict().items()}
        for m in MODULES
    }
    optim = {
        m: [
            {k: v.clone() for k, v in slots.items() if isinstance(v, torch.Tensor)}
            for slots in learner.get_optimizer(module_id=m).state.values()
        ]
        for m in MODULES
    }
    return weights, optim


def _run_pair(spec, vf_share_layers, candidate_spec=None):
    reference = _build("jepa", spec, vf_share_layers)
    candidate = _build(FIXTURE, spec if candidate_spec is None else candidate_spec, vf_share_layers)
    try:
        ref, cand = reference.learner_group._learner, candidate.learner_group._learner
        cand.set_state(ref.get_state())
        ref_metrics, cand_metrics = [], []
        for _ in range(ROUNDS):
            episodes = reference.env_runner_group.local_env_runner.sample(
                num_episodes=TEST_CFG["num_episodes_per_iter"]
            )
            ref_metrics.append(_update(reference, episodes))
            cand_metrics.append(_update(candidate, episodes))
        return (_snapshot(ref), ref_metrics), (_snapshot(cand), cand_metrics), cand
    finally:
        reference.stop()
        candidate.stop()


def _assert_same(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert set(a) == set(b), path
        for key in a:
            _assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and np.isnan(a):
        assert np.isnan(b), path
    else:
        assert a == b, path


@pytest.fixture(scope="module", autouse=True)
def _ray():
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=2)
    yield
    ray.shutdown()


@pytest.mark.parametrize("spec, vf_share_layers", [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_the_package_trains_exactly_like_gym_cdas_jepa(spec, vf_share_layers):
    (ref_state, ref_metrics), (cand_state, cand_metrics), cand = _run_pair(spec, vf_share_layers)

    # The candidate really ran this package's code, not gym-CDA's under another name.
    encoder = cand.module[MODULES[0]].encoder
    sub = encoder.encoder if vf_share_layers else encoder.actor_encoder
    assert type(sub).__module__ == "cda_jepa.encoder"

    assert all(ref_metrics[0][m] for m in MODULES), "no jepa_* metric was logged"
    if spec.get("world_model"):
        assert "jepa_world_loss" in ref_metrics[0][MODULES[0]]
    _assert_same(ref_metrics, cand_metrics, "metrics")
    _assert_same(ref_state, cand_state, "state")
    # The EMA target is in the weights; say so rather than leave it implied.
    assert any("target_trunk" in k for k in ref_state[0][MODULES[0]])


def test_the_comparison_can_fail():
    """A different mask ratio on the candidate must show up as a difference."""
    (ref_state, ref_metrics), (cand_state, cand_metrics), _ = _run_pair(
        {}, False, candidate_spec={"mask_ratio": 0.6}
    )
    with pytest.raises(AssertionError):
        _assert_same(ref_metrics, cand_metrics, "metrics")
