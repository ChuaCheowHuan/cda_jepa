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
import subprocess
import sys
import tomllib
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
    # A namespace package's `__path__` can also hold an editable install's finder hook,
    # which is not a directory (#8); only real directories hold installed files.
    (path,) = [p for p in importlib.import_module(package).__path__ if os.path.isdir(p)]
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
    if package == "lejepa":
        # The LeJEPA code still ships, but its only dependency (scipy) is an
        # optional extra now (#19); importing it without that extra skips here.
        pytest.importorskip("lejepa", reason="install the lejepa extra")

    installed = dict(_installed_files(package))
    assert installed, f"nothing installed for {package}"
    for relpath, full in installed.items():
        with open(full, "rb") as got, open(os.path.join(source, relpath), "rb") as want:
            assert got.read() == want.read(), f"{package}/{relpath} differs from the submodule"


def test_the_encoder_imports_from_a_directory_that_shadows_src(tmp_path):
    """A user's own top-level `src/` must not break `import cda_jepa.encoder` (#5).

    I-JEPA is installed as a namespace package named `src`, and a regular
    package anywhere earlier on `sys.path` wins. The encoder used to import
    `src.masks.utils` at module level, so a normal `src/` layout made it fail
    far from the cause. The gather is written here now, so the import must work
    with a shadowing `src/__init__.py` present.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [ROOT, env["PYTHONPATH"]] if env.get("PYTHONPATH") else [ROOT]
    )
    output = tmp_path / "result.txt"
    with open(output, "w") as handle:
        completed = subprocess.run(
            [sys.executable, "-c", "import cda_jepa.encoder"],
            cwd=tmp_path,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )

    assert completed.returncode == 0, output.read_text()


def test_lejepa_and_scipy_are_optional():
    """The fork is unused, so a default install must not pull scipy in (#19)."""
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
        project = tomllib.load(handle)["project"]

    assert "scipy" not in project["dependencies"]
    assert project["optional-dependencies"]["lejepa"] == ["scipy"]


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
