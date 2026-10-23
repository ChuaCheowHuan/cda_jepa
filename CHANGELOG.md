# Changelog

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
