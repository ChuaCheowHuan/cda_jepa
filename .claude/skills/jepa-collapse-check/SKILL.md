---
name: jepa-collapse-check
description: Measure whether jepa_latent_std and the variance hinge can see a collapse, on inputs whose answer is known (random, identical, blind trunk). Use before and after any change to _jepa_loss, VARIANCE_TARGET, the predictor or the pretrainer's collapse gate, and when someone asks whether a run collapsed.
---

# Check the collapse metric against known answers

A collapsed JEPA drives its prediction loss to zero, which reads as success. The only guard is
`latent_std` (the Learner's `jepa_latent_std`, the pretrainer's `PretrainReport.usable`), so the
metric itself has to be checked: a number that stays near 0.9 for a dead encoder protects nothing
(issue #4).

## Run it

Needs the environment from `jepa-dev-setup`.

```bash
python .claude/skills/jepa-collapse-check/scripts/collapse_check.py
python .claude/skills/jepa-collapse-check/scripts/collapse_check.py --check   # exit 1 if missed
```

It builds a fresh default module and runs three batches through it with one fixed mask:

| Input | What it models | A working metric reads |
|---|---|---|
| `random` | distinct observations | the healthy reference |
| `identical` | one observation repeated | far below `random` |
| `blind` | the online trunk ignores its input | far below `random` |

`--check` fails when `identical` or `blind` is above half of `random`. The `hinge` column is
`(aux_loss - predict_loss) / variance_coeff`: on `random` it shows whether the hinge is active on
a healthy encoder (issue #7).

## Read the result

- **On main before #4 is fixed** all three read about 0.87 to 0.88: the statistic measures spread
  across masked *positions*, which the predictor's position embeddings provide on their own.
- **After a fix** (spread across the batch per position), `identical` and `blind` should read
  near 0, and `--check` should pass. Add the same two cases as regression tests in
  `tests/test_encoder.py`; the script is the quick check, the tests are the guard.
- A fix that changes the statistic also changes the hinge, which reads it: rerun and report both
  columns.
