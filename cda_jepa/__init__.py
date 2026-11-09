"""JEPA for gym-continuousDoubleAuction, as an installable extension.

Installing this package registers two things with gym-CDA, neither imported
until a run asks for it:

* the `jepa` extension (`gym_cda.extensions`), which a run lists in
  `extensions_enabled`; and
* the `jepa` encoder (`gym_cda.encoders`), which that run selects with
  `encoder_type: "jepa"` and configures under `encoder_specs.jepa`.

The encoder, its RLModule and its Learner are `encoder.py` and `learner.py`; the
offline pretrainer is `python -m cda_jepa.pretrain`. I-JEPA's code ships unchanged
as the top-level package `src`, from a pinned submodule of the fork. LeJEPA is
optional: `cda-jepa[lejepa]` installs its pinned fork, and the default wheel does
not carry it (docs/provenance.md). See docs/jepa.md for the method.
"""

from gym_continuousDoubleAuction.train.extensions import Extension

__version__ = "0.1.0"

#: The extension has no settings of its own: the encoder's live in
#: `encoder_specs.jepa`, where gym-CDA's encoder fingerprint already guards them,
#: and its Learner and RLModule come with the encoder (`register`'s class paths).
EXTENSION = Extension(name="jepa", version=__version__)
