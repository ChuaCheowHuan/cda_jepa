# Changelog

## Unreleased

- **Tests pruned and hardened.** The variance-hinge gradient test pinned its mask with
  `torch.manual_seed`, which stopped working when masks got their own generator, so the two passes
  it compares saw different masks, and it passed with the hinge detached (#25); it now reseeds the
  encoder's generator. The determinism test checked `forward_train`, which RLlib runs in train
  mode, with dropout off; it now checks the rollout methods with dropout on (#26).
  `test_packaging` works on the documented editable install (#8). Duplicates and tests of gym-CDA's own behaviour are gone: a second "target takes no
  gradient" test, a second "action space missing a head" test, the shipped-layout mask test
  (folded into the `n_hist` one), the episode-split tests (gym-CDA's `test_probe`), the pooling
  identity check, `world_model` off by default (the settings table covers it), the stateful and
  `vf_share_layers` branches of the encoder contract, and a second pretraining run that repeated
  the `saved` fixture.
- **ruff's lint rules are pinned** (`E4`, `E7`, `E9`, `F`) and `ruff` is in the `test` extra, so
  `ruff check .` and `ruff format --check .` give the same answer on any ruff version.
- **Docs say what the code does today.** `jepa_latent_std` does not yet detect a collapse (#4),
  the variance hinge is never fully off (#7), only one Learner works (#6), `level` masks under
  `level` tokenisation are random (#9). Stale references to gym-CDA paths and old names in the
  code are fixed (#15). `docs/pretraining.md` loses gym-CDA's section numbers, fixes a broken
  table, and says to load only trusted checkpoints. `docs/jepa.md` loses its review-fix log; the
  one note a user needs (old world-model weights do not load) is in `docs/migration.md`.
- **Agent setup.** `CLAUDE.md`, and two skills in `.claude/skills/`: `jepa-dev-setup` and
  `jepa-collapse-check`, whose script measures the collapse metric on random, identical and
  blind-trunk inputs.

- **The equivalence test is removed.** `tests/test_equivalence.py` compared this package with
  gym-CDA's built-in `jepa`, which gym-CDA 2.3 and later no longer ship, so it always skipped.
  Its result is recorded in `docs/migration.md`.
- **Docs brought up to date.** `docs/jepa.md` and `docs/pretraining.md` name this package's files
  and `PRETRAIN_DEFAULTS` instead of gym-CDA's old paths and `cli_defaults.cda_pretrain`, and link
  cda-cbp's note rather than gym-CDA's pointer page. `docs/development.md` lists the CI
  workflow. The README draws how the forks,
  this package and gym-CDA fit.

## 0.1.0

- **JEPA moves here from gym-CDA.** The encoder (`cda_jepa/encoder.py`), its module and Learner
  (`cda_jepa/learner.py`), the offline pretrainer (`cda_jepa/pretrain/`), their tests and the
  research note (`docs/jepa.md`) come from gym-CDA at 67d436a, with their git history.
- **Found through entry points.** `gym_cda.encoders` lists `cda_jepa.encoder`, which registers
  `jepa` when gym-CDA first looks the name up; `gym_cda.extensions` lists `EXTENSION`, so a config
  selects `jepa` only with `"jepa"` in `extensions_enabled`. Needs gym-CDA 2.2.
- **The forks are submodules, installed unchanged.** `third_party/ijepa` (installed as `src`) and
  `third_party/lejepa` (installed as `lejepa`). `tests/test_packaging.py` checks every installed
  file against the pinned commit.
- **I-JEPA's `apply_masks` gathers the tokens.** It replaces the encoder's own `torch.gather` and
  gives identical results, so training is unchanged.
- **The pretrainer is `python -m cda_jepa.pretrain`**, with its defaults in `cda_jepa/config.py`
  (they were gym-CDA's `cli_defaults.cda_pretrain`). It takes this package's encoder under any
  registered name.
- **Equivalence with gym-CDA's built-in `jepa`.** `tests/test_equivalence.py` updates both learners
  from the same state on the same episodes for three rounds and requires identical weights, EMA
  target, optimiser state and `jepa_*` metrics in five cases (defaults, shared trunk, world model,
  both, time masks). It runs while gym-CDA still ships `jepa`.
