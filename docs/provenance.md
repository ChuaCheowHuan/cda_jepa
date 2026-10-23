# What comes from the I-JEPA and LeJEPA code

Both forks are git submodules pinned to one commit and installed unchanged:
`third_party/ijepa/src` as the top-level package `src`, `third_party/lejepa/lejepa` as `lejepa`.
Nothing of theirs is copied into this repository, and nothing of theirs is edited.
`tests/test_packaging.py` checks every installed file against the submodule, byte for byte.

## I-JEPA (`src`)

The authors' repository has no packaging and no `__init__.py` files, and its modules import each
other as `src.masks...`, `src.utils...`. It can only be installed as a package named `src`, so it
is (a namespace package, as in the repository). A project that has its own top-level `src`
package would clash with it; gym-CDA does not.

| Piece | Used | Why |
|---|---|---|
| `src.masks.utils.apply_masks` | **yes**, in `encoder.py`'s `_gather` | Gathers the context tokens and the target slice. Bit-identical to the `torch.gather` it replaced |
| `src.models.vision_transformer.VisionTransformerPredictor` | no | It places tokens with fixed 2-D sin-cos positions on a square grid. The book grid is snapshots x levels, which is not square: run on it, the predictor fails on a shape mismatch (48 positions against 36). And every checkpoint holds this encoder's learned time and level embeddings instead |
| `src.models.vision_transformer.VisionTransformer` | no | An image patch embedding; the book has its own tokenisation, shared with gym-CDA's `transformer` |
| `src.masks.multiblock.MaskCollator` | no, planned opt-in | It draws from the global torch RNG, which moved a seeded run's random stream (gym-CDA #185, fixed by giving the encoder its own generator), and it is a `DataLoader` collator, not a per-batch call |
| `src.utils.schedulers` | no, planned opt-in | The EMA decay and loss weights are constant today; a schedule would change training |
| EMA step, layer-normed target, smooth-L1 loss | written here | They are inline in the authors' `train.py`, not functions that can be imported |

## LeJEPA (`lejepa`)

| Piece | Used | Why |
|---|---|---|
| `lejepa.multivariate.SlicingUnivariateTest` with `lejepa.univariate.EppsPulley` (SIGReg) | no, planned opt-in | It is a different anti-collapse objective from today's EMA target plus variance hinge; making it the default would change every run. Planned as `"objective": "lejepa"`. It takes its own `torch.Generator`, so it can keep seeded runs reproducible |

`lejepa` imports `scipy` without declaring it; this package declares `scipy` so the import works.
The fork's `pyproject.toml` declares `pytest` and `loguru`, which its code does not import, and it
is not installed from that file: only the `lejepa` directory is mapped in.

## Licence

Both forks are CC BY-NC 4.0. See [NOTICE](../NOTICE).
