# Changelog

## Unreleased

- **Fixes from the agent-labelled review issues.**
  `jepa_latent_std` is now the spread across the batch at each masked position, so identical
  observations and a blind trunk read near zero instead of the predictor's positional ~0.87 (#4);
  `jepa_offdiag_cov` follows the same axis. The variance hinge's target is now 0.2: the corrected
  statistic's healthy scale is about 0.2, and the old 1.0 target made the hinge dominate the
  objective (the target remains the human decision in #22). `level` masking under `level`
  tokenisation hides a
  contiguous band and keeps the global token visible, instead of silently drawing a random mask
  (#9). `CDAJEPALearner` unwraps RLlib's DDP wrapper before reading the encoder config, so
  `num_learners > 1` no longer raises (#6). The mask generator's state travels in the encoder's
  `state_dict` as `_mask_state`, so a resumed run continues the stream, and its seed now comes from
  a digest of the encoder's structural settings rather than a process-wide build counter, so build
  order no longer shifts it (#13). The EMA step is one fused `_foreach_lerp_`, non-CPU mask draws
  happen on the device, and pretraining has an objective-only forward path that skips the discarded
  policy pass (#14, #12).
- **The pretrainer validates its arguments** (`steps`, `batch_size`, `lr`, `log_every`), averages
  `train_loss` over the logging interval so it is comparable with validation, gives validation its
  own seeded mask generator, and runs it with dropout off (#12). `--log-every 0` is an argparse
  error, not a `ZeroDivisionError`.
- **The encoder no longer imports I-JEPA's top-level `src` package.** `_gather` is the one-line
  `torch.gather` it used, written here, so a user's own `src/` package cannot break
  `import cda_jepa.encoder` (#5). The fork still ships unchanged and `test_packaging` still checks
  its bytes.
- **CI and packaging hygiene.** The workflow has `permissions: contents: read`, actions pinned to
  commit SHAs, and a Dependabot config for `github-actions` (#20, #21). LeJEPA and `scipy` are no
  longer installed by default: nothing imports LeJEPA yet, so the default wheel carries only
  I-JEPA, and `cda-jepa[lejepa]` installs the pinned LeJEPA fork plus `scipy` (#19).
- **The LeJEPA fork no longer commits `__pycache__`.** The fork at `8f1a7bc` removes the tracked
  `.pyc` files and ignores bytecode, and this package's submodule pin moved there (#27). An
  editable install no longer dirties the submodule on the first import.
- **The gym-CDA-format tests moved upstream.** The fingerprint-merge and checkpoint-format tests
  that exercised gym-CDA code were removed from `tests/test_pretrain.py`; the equivalent coverage
  now lives in gym-CDA's `test_pretrained_format.py`, which uses a test encoder with a
  `pretrained_path` field. This package keeps only its own round-trip and JEPA-spec-mismatch checks
  (#28).
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
- **Docs were brought in line with the fixes above.** The old caveats that `jepa_latent_std`
  cannot detect a collapse, the variance hinge is never off, only one Learner works, and `level`
  masks are random are removed. Stale references to gym-CDA paths and old names in the code are
  fixed (#15). `docs/pretraining.md` lost gym-CDA's section numbers, has a working table, and says
  to load only trusted checkpoints. `docs/jepa.md` lost its review-fix log; the one note a user
  needs (old world-model weights do not load) is in `docs/migration.md`.
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
