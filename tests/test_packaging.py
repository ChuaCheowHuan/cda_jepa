"""What this package installs, and how gym-CDA finds it.

Two claims:

  1. **The forks ship unchanged.** `src` (I-JEPA) and `lejepa` are installed straight
     from the pinned submodules by `package-dir`, with no copy in this repository to
     drift. Every installed file must equal its submodule file byte for byte.
  2. **gym-CDA finds the encoder through entry points.** It imports nothing from this
     package by name: the `gym_cda.extensions` and `gym_cda.encoders` entries are the
     whole contract.
"""

import importlib
import os
from importlib.metadata import entry_points

import pytest

import cda_jepa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Installed package -> the submodule directory it is built from.
FORKS = {
    "src": os.path.join(ROOT, "third_party", "ijepa", "src"),
    "lejepa": os.path.join(ROOT, "third_party", "lejepa", "lejepa"),
}


def _installed_files(package):
    (path,) = importlib.import_module(package).__path__
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith(".py"):
                full = os.path.join(dirpath, name)
                yield os.path.relpath(full, path), full


@pytest.mark.parametrize("package", sorted(FORKS))
def test_every_installed_fork_file_equals_the_submodule(package):
    source = FORKS[package]
    if not os.path.isdir(source) or not os.listdir(source):
        pytest.skip("submodules not checked out (git submodule update --init)")

    installed = dict(_installed_files(package))
    assert installed, f"nothing installed for {package}"
    for relpath, full in installed.items():
        with open(full, "rb") as got, open(os.path.join(source, relpath), "rb") as want:
            assert got.read() == want.read(), f"{package}/{relpath} differs from the submodule"


def test_the_ijepa_function_the_encoder_uses_is_the_forks():
    from src.masks.utils import apply_masks

    from cda_jepa import encoder

    assert encoder.apply_masks is apply_masks


def test_the_extension_entry_point_is_this_packages():
    (entry,) = [ep for ep in entry_points(group="gym_cda.extensions") if ep.name == "jepa"]
    extension = entry.load()

    assert extension is cda_jepa.EXTENSION
    assert extension.name == "jepa"
    assert extension.version == cda_jepa.__version__


def test_the_encoder_entry_point_names_the_encoder_module():
    (entry,) = [ep for ep in entry_points(group="gym_cda.encoders") if ep.name == "jepa"]

    assert entry.value == "cda_jepa.encoder"


def test_the_encoder_belongs_to_the_extension():
    from gym_continuousDoubleAuction.train.model.encoders import encoder_extension

    from conftest import GYM_CDA_HAS_JEPA

    if GYM_CDA_HAS_JEPA:
        pytest.skip("this gym-CDA still ships its own jepa")
    assert encoder_extension("jepa") == "jepa"
