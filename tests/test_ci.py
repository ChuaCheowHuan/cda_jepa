"""The CI workflow is part of the supply chain this package ships (#16, #20, #21)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "cda_jepa.yml"


def test_the_token_is_read_only():
    text = WORKFLOW.read_text()
    assert re.search(r"^permissions:\s*\n\s+contents:\s*read\s*$", text, flags=re.M)


def test_every_action_is_pinned_to_a_commit():
    """A movable tag lets the action's owner change what CI runs."""
    uses = re.findall(r"^\s*-\s*uses:\s*(\S+)(?:\s+#.*)?$", WORKFLOW.read_text(), flags=re.M)
    assert uses
    for action in uses:
        ref = action.rsplit("@", 1)[-1]
        assert re.fullmatch(r"[0-9a-f]{40}", ref), action


def test_dependabot_keeps_the_pins_current():
    config = (ROOT / ".github" / "dependabot.yml").read_text()
    assert 'package-ecosystem: "github-actions"' in config
