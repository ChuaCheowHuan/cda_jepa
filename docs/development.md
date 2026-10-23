# Development

## Layout

```
cda_jepa/__init__.py         EXTENSION: what gym-CDA discovers through gym_cda.extensions
cda_jepa/encoder.py          the encoder; listed under gym_cda.encoders, registers `jepa` on import
cda_jepa/learner.py          JEPARLModule and CDAJEPALearner: the loss terms and metrics
cda_jepa/pretrain/           offline pretraining (python -m cda_jepa.pretrain)
cda_jepa/config.py           the pretrainer's CLI defaults
tests/                       pytest; `python -m pytest -q`
third_party/ijepa            I-JEPA, a git submodule, installed unchanged as `src`
third_party/lejepa           LeJEPA, a git submodule, installed unchanged as `lejepa`
.github/workflows/cda_jepa.yml  CI
```

## Install for development

gym-CDA is not on PyPI. Install it first (CPU torch is enough), then this package:

```bash
pip install torch==2.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install "gym_continuousDoubleAuction @ git+https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction"
git clone --recurse-submodules https://github.com/ChuaCheowHuan/cda_jepa
cd cda_jepa
pip install -e ".[test]"
python -m pytest -q
```

A clone made without `--recurse-submodules` needs `git submodule update --init` first, or the
build stops at the missing `third_party/*` directories.

To move a fork to a newer commit: `git -C third_party/ijepa fetch && git -C third_party/ijepa
checkout <sha>`, commit the submodule, and let CI run.

## Against gym-CDA 2.2

gym-CDA 2.3 and later ship no `jepa`, and this package registers it. gym-CDA 2.2, the oldest this
package supports, still ships its own, which wins; `tests/conftest.py` then registers this
package's encoder as the fixture `_jepa_ext`, so the tests exercise this package's code either
way. `JEPA` in conftest is the name to build.

## What this package uses from gym-CDA

| From gym-CDA | For |
|---|---|
| `train.model.encoders`: `register`, `encoder_settings`, `ENCODER_REGISTRY`, `ensure_registered` | registering the encoder and reading its settings |
| `train.model.encoders.pretrained` | the pretrained-weights format |
| `train.model.encoders`: `obs_layout`, `tokenize`, `blocks`, `transformer` | the book grid, its tokens, and the trunk shared with gym-CDA's `transformer` |
| `train.model.moe_learner`: `CDAPPOTorchRLModule`, `CDAPPOTorchLearner` | the classes the module and Learner subclass |
| `train.extensions.Extension` | `EXTENSION` |
| `train.probe` corpus and split | the pretrainer's data |

## Tests

| File | Covers |
|---|---|
| `test_encoder.py` | The encoder contract every gym-CDA encoder meets, masks, EMA, collapse metrics, the world model, review regressions |
| `test_jepa_value_pass.py` | The value pass does not add a second JEPA term |
| `test_world_model_actions.py` | The world model embeds every action head |
| `test_pretrain.py` | The pretrainer: the loop, collapse, the checkpoint, its guard, the CLI |
| `test_wiring.py` | A real gym-CDA `Algorithm`: the loss terms reach the optimiser, the NEXT_OBS connector is conditional, other encoders are unaffected |
| `test_moved_checks.py` | Checks from other gym-CDA test files: one-row batches give finite losses (#242), the pooling tuple is gym-CDA's, pretraining builds in the corpus's book mode (#187) |
| `test_packaging.py` | The installed forks equal the submodules; the entry points |
| `test_docs.py` | The settings table matches `JEPA_DEFAULTS`, and every relative link resolves |

## CI

`.github/workflows/cda_jepa.yml` checks out the submodules, installs gym-CDA at `GYM_CDA_REF` and
runs the tests.
