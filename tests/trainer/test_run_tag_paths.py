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
TAGGED = (
    "trainer.default_local_dir", "trainer.val_instance_log_dir",
    "trainer.project_name", "trainer.experiment_name",
)
# Only the arms that dump them have it, so it is checked where it appears rather
# than required everywhere. It writes one file per STEP with open(..., "w"), so
# an untagged re-run does not append beside the finished run's dumps -- it
# overwrites them, step by step, as it reaches each one.
TAGGED_IF_PRESENT = ("trainer.sign_token_dump_dir",)


def _assign(text, key):
    m = re.search(rf"{re.escape(key)}=([^ \\\n]*)", text)
    assert m, f"{key} not found"
    return m.group(1)


def _expand(value, home, tag, arm=""):
    """What the shell would produce for this assignment.

    RUN_TAG_SUFFIX is derived exactly as the scripts derive it, so the test is
    reading the same definition rather than a second copy of it. ARM rides along
    for the scripts that carry several arms in one file -- see _arms.
    """
    out = subprocess.run(
        ["bash", "-c",
         f'HOME={home} RUN_TAG={tag!r} ARM={arm!r}; RUN_TAG_SUFFIX="${{RUN_TAG:+_$RUN_TAG}}"; echo "{value}"'],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _arms(text):
    """The ARM values a script offers, or [""] when it is a single-arm script.

    A script whose arms differ only in a coefficient keeps them in one file --
    that is how "the arms differ in nothing else" stays true -- so the checks
    below have to run once per arm rather than once per file, or two of three
    arms would go unchecked.
    """
    # Detected by the switch, not by a default assignment: the coef script has
    # no ARM default any more (a forgotten ARM= used to run the retired
    # redistribute arm silently), and keying off "ARM=${ARM:-" made this helper
    # report a multi-arm script as single-arm and then resolve the lock path
    # with an empty arm.
    if 'case "$ARM" in' not in text:
        return [""]
    block = text.split('case "$ARM" in', 1)[1].split("esac", 1)[0]
    arms = re.findall(r"^\s+([a-z][a-z0-9_]*)\)\s*$", block, re.M)
    assert arms, "the script has an ARM switch with no branches"
    return arms


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_an_unset_tag_leaves_every_path_exactly_as_it_was(script):
    """Byte-for-byte, or an existing run stops resuming the moment this lands."""
    text = script.read_text()
    for arm in _arms(text):
        for key in TAGGED + tuple(k for k in TAGGED_IF_PRESENT if f"{k}=" in text):
            raw = _assign(text, key)
            assert "$RUN_TAG_SUFFIX" in raw, key
            plain = raw.replace("$RUN_TAG_SUFFIX", "")
            assert _expand(raw, "/h", "", arm) == _expand(plain, "/h", "", arm)
            assert not _expand(raw, "/h", "", arm).endswith("_")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_a_set_tag_moves_every_tagged_value(script):
    text = script.read_text()
    for arm in _arms(text):
        for key in TAGGED + tuple(k for k in TAGGED_IF_PRESENT if f"{k}=" in text):
            raw = _assign(text, key)
            assert _expand(raw, "/h", "v2", arm) == _expand(raw, "/h", "", arm) + "_v2"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_the_lock_expects_the_same_suffixed_name_the_script_passes(script):
    """Suffixing the wandb names without teaching the lock would make every
    tagged run fail the check that catches a mislabelled multi-hour run. The
    lock has to carry the suffix too, and against the SAME base."""
    import re as _re

    text = script.read_text()
    lock = _re.search(r"\+trainer\.expected_config=(\S+)", text)
    assert lock, "the arm does not pin an expectations file"
    lock_path = lock.group(1).strip('"')
    for arm in _arms(text):
        # A multi-arm script names its lock by $ARM, so the path is resolved the
        # same way the shell would resolve it rather than read literally.
        resolved = _expand(lock_path, "/h", "", arm)
        expected = (ROOT / resolved).read_text()
        for key in ("trainer.project_name", "trainer.experiment_name"):
            passed = _expand(_assign(text, key).strip('"'), "/h", "", arm)
            pinned = _re.search(rf'^"{_re.escape(key)}":\s*(\S+)\s*$', expected, _re.M)
            assert pinned, f"{key} is not pinned in {resolved}"
            want = _expand(pinned.group(1), "/h", "", arm)
            assert want == passed, (resolved, key, want, passed)
            assert pinned.group(1).endswith("$RUN_TAG_SUFFIX"), key


def test_every_arm_still_has_its_own_untagged_directory():
    """The tag separates re-runs of ONE arm. Two arms sharing a directory would
    be a different bug, and this is where it would show up."""
    dirs = {}
    for script in SCRIPTS:
        text = script.read_text()
        raw = _assign(text, "trainer.default_local_dir")
        for arm in _arms(text):
            d = _expand(raw, "/h", "", arm)
            dirs.setdefault(d, []).append(f"{script.name}{':' + arm if arm else ''}")
    clashes = {d: names for d, names in dirs.items() if len(names) > 1}
    assert not clashes, clashes
