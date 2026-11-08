---
name: jepa-dev-setup
description: Set up a working cda_jepa test environment (submodules, CPU torch, gym-CDA, this package) and run the suite. Use at the start of any session that will run tests or reproduce a finding in this repository.
---

# Set up cda_jepa for tests

The package imports gym-CDA, Ray and torch at module level, and installs two forks from git
submodules, so nothing runs until all four are in place.

## Steps

1. **Submodules.** `git submodule update --init`. Without them the build stops at the missing
   `third_party/*` directories, and `test_packaging` skips.
2. **A virtual environment** outside the repository (the session scratchpad), Python 3.12 or later.
3. **Torch, CPU only.** `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
   If that host is blocked, the PyPI wheel works too (`pip install torch`), but it pulls in
   several GB of CUDA libraries and takes minutes: start it in the background.
4. **gym-CDA.** A sibling checkout is fastest and lets you pin a commit:
   `pip install -e ../gym-continuousDoubleAuction`. Otherwise
   `pip install "gym_continuousDoubleAuction @ git+https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction@<ref>"`.
   Record the gym-CDA commit next to any result: CI tests against `master`, which moves.
5. **This package.** `pip install -e ".[test]"`. Editable is fine for every test.
6. **Run.** `python -m pytest -q` (about half a minute; `test_wiring` builds and trains a real
   `Algorithm`, so it needs no GPU but does start Ray). Then `ruff check . && ruff format --check .`.

## Reproducing a claim

- Write the repro in the scratchpad, not in the repository, and keep its raw output.
- Run the inverse case too (a case that should pass); a repro that cannot tell the two apart
  proves nothing.
- For a fix, show the new test failing on the old source first:
  `git stash push -- cda_jepa`, run the test, `git stash pop`.
- For anything about collapse or the variance hinge, use the `jepa-collapse-check` skill.

## Gotchas

- Mask sampling has its own generator (`TorchJEPAEncoder._mask_rng`). `torch.manual_seed` does
  not reset it once an encoder has drawn a mask: to pin a mask, set
  `encoder._mask_generator = torch.Generator().manual_seed(n)` before the forward.
- On gym-CDA 2.2, which still ships its own `jepa`, the tests build this package's encoder under
  the name `conftest.JEPA` (`_jepa_ext`). Build with `JEPA`, never the literal `"jepa"`.
