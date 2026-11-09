# Development

## Layout

```
cda_jepa/__init__.py         EXTENSION: what gym-CDA discovers through gym_cda.extensions
cda_jepa/encoder.py          the encoder; listed under gym_cda.encoders, registers `jepa` on import
cda_jepa/learner.py          JEPARLModule and CDAJEPALearner: the loss terms and metrics
cda_jepa/pretrain/           offline pretraining (python -m cda_jepa.pretrain)
cda_jepa/config.py           the pretrainer's CLI defaults
tests/                       pytest; `python -m pytest -q`
CLAUDE.md, .claude/skills/   rules and skills for coding agents working here
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
ruff check . && ruff format --check .
```

`scipy` is not installed by default: nothing imports LeJEPA yet (#19). Add the `lejepa` extra
(`pip install -e ".[test,lejepa]"`) when you want `tests/test_packaging.py` to check the LeJEPA
files; otherwise that one case skips.

If `download.pytorch.org` is unreachable, `pip install torch` from PyPI also works; it pulls in the
CUDA libraries and is several GB. The `jepa-dev-setup` skill (`.claude/skills/`) has the details.

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

A test belongs here when it fails if this package's code is wrong. Behaviour that is gym-CDA's own
(its registry, catalog, probe split, checkpoint format) is tested in gym-CDA.

| File | Covers |
|---|---|
| `test_encoder.py` | The encoder contract every gym-CDA encoder meets; masks, EMA, the collapse metrics and the variance hinge; the world model; inference-only modules |
| `test_jepa_value_pass.py` | The value pass runs no objective and takes no EMA step; masks have their own generator |
| `test_world_model_actions.py` | The world model embeds every action head |
| `test_pretrain.py` | The pretrainer: the loop, collapse, the checkpoint, its guard, the CLI |
| `test_wiring.py` | A real gym-CDA `Algorithm`: the loss terms reach the optimiser, the NEXT_OBS connector is conditional, other encoders are unaffected |
| `test_moved_checks.py` | Checks from other gym-CDA test files: one-row batches give finite losses (gym-CDA #242), pretraining builds in the corpus's book mode (#187), the Learner calls gym-CDA's loss seam |
| `test_packaging.py` | The installed forks equal the submodules; the entry points; the encoder imports with a shadowing `src/` |
| `test_docs.py` | The settings table matches `JEPA_DEFAULTS`, and every relative link resolves (docs, `CLAUDE.md`, skills) |
| `test_ci.py` | The workflow token is read-only, actions are commit-pinned, and Dependabot keeps the pins current |

## CI

`.github/workflows/cda_jepa.yml` checks out the submodules, installs gym-CDA at `GYM_CDA_REF`
(default `master`) and runs the tests. The job token is read-only, and each action is pinned to a
commit SHA that Dependabot updates. It does not yet run ruff
([#18](https://github.com/ChuaCheowHuan/cda_jepa/issues/18)) or test the gym-CDA 2.2 floor
([#23](https://github.com/ChuaCheowHuan/cda_jepa/issues/23)).
