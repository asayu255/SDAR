"""RUN_TAG moves the run's own directories and nothing else.

`resume_mode` is `auto`, so a second run of an arm pointed at a directory that
already holds `global_step_*` resumes rather than starting over -- and a
directory holding a COMPLETED run resumes to the final step and exits with
nothing done, silently. That is the failure this knob exists to prevent, so what
it must and must not touch is worth pinning: the two $HOME-derived paths move,
and the pinned run identity (experiment_name / project_name) does not, because a
re-run of the same arm is the same experiment and the intent lock says so.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted(
    p for p in ROOT.glob("examples/opd*_trainer/run_*.sh")
    if "trainer.default_local_dir=" in p.read_text()
)
TAGGED = ("trainer.default_local_dir", "trainer.val_instance_log_dir")
UNTAGGED = ("trainer.experiment_name", "trainer.project_name")


def _assign(text, key):
    m = re.search(rf"{re.escape(key)}=([^ \\\n]*)", text)
    assert m, f"{key} not found"
    return m.group(1)


def _expand(value, home, tag):
    """What the shell would produce for this assignment."""
    out = subprocess.run(
        ["bash", "-c", f'HOME={home} RUN_TAG={tag!r}; echo "{value}"'],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_an_unset_tag_leaves_every_path_exactly_as_it_was(script):
    """Byte-for-byte, or an existing run stops resuming the moment this lands."""
    text = script.read_text()
    for key in TAGGED:
        raw = _assign(text, key)
        assert "${RUN_TAG:+_$RUN_TAG}" in raw, key
        plain = raw.replace("${RUN_TAG:+_$RUN_TAG}", "")
        assert _expand(raw, "/h", "") == _expand(plain, "/h", "")
        assert not _expand(raw, "/h", "").endswith("_")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_a_set_tag_moves_both_directories(script):
    text = script.read_text()
    for key in TAGGED:
        raw = _assign(text, key)
        assert _expand(raw, "/h", "v2") == _expand(raw, "/h", "") + "_v2"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_the_tag_does_not_touch_the_pinned_run_identity(script):
    """experiment_name and project_name are pinned in the intent lock as the
    identity of the CONFIG. Suffixing them would make every tagged run fail the
    lock, which is the check that catches a mislabelled multi-hour run."""
    text = script.read_text()
    for key in UNTAGGED:
        assert "RUN_TAG" not in _assign(text, key), key


def test_every_arm_still_has_its_own_untagged_directory():
    """The tag separates re-runs of ONE arm. Two arms sharing a directory would
    be a different bug, and this is where it would show up."""
    dirs = {}
    for script in SCRIPTS:
        d = _assign(script.read_text(), "trainer.default_local_dir")
        dirs.setdefault(d, []).append(script.name)
    clashes = {d: names for d, names in dirs.items() if len(names) > 1}
    assert not clashes, clashes
