# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""What the privileged notice claims by construction, checked as arithmetic.

The text is the mechanism, so its hashes are pinned here against the design
document's table; the prefix/strip pair must be exact inverses; and the
fingerprint adjustment must equal what prepending actually does to
``row_fingerprint`` -- otherwise the teacher-mode exchange rejects every entry.
"""
import os

import pytest
import torch

from verl.trainer.ppo import privileged_notice as pn
from verl.workers.teacher_cache import row_fingerprint


def _home_dir():
    """HOME-independent: another test in this suite points HOME elsewhere and
    does not restore it, which would turn this test into a hub download."""
    try:
        import pwd
        return pwd.getpwuid(os.getuid()).pw_dir
    except (ImportError, KeyError):
        return os.path.expanduser("~")


_REAL_QWEN3_DIR = os.path.join(_home_dir(), "offline_ladder", "klw_control_step300_hf")

# docs/privileged_multitask_notice_design.md section 2.3
DESIGN_TABLE = {
    ("named", "alfworld"): "46f6f162d9fb",
    ("named", "search"): "35500e7a3bec",
    ("named", "webshop"): "c58107a11ff3",
    ("placebo", "alfworld"): "53b879e2adb6",
    ("placebo", "search"): "a382c97fdebf",
    ("placebo", "webshop"): "a4774a75f179",
}


# ------------------------------------------------------------------ the texts
def test_the_texts_hash_to_the_design_table():
    for (variant, task), want in DESIGN_TABLE.items():
        assert pn.notice_sha256(variant, task).startswith(want), (variant, task)


def test_named_names_the_other_two_and_placebo_names_none():
    others = {"alfworld": ("WebShop", "Search"), "webshop": ("ALFWorld", "Search"),
              "search": ("ALFWorld", "WebShop")}
    for task, (a, b) in others.items():
        named = pn.notice_text("named", task)
        placebo = pn.notice_text("placebo", task)
        assert a in named and b in named and "three environments at once" in named
        assert a not in placebo and b not in placebo and "three environments" not in placebo


def test_placebo_keeps_the_same_avoid_list():
    """The contrast isolates the multi-task framing ONLY, so the third paragraph
    -- what to avoid -- must be identical between the two variants."""
    for task in pn.TASKS:
        n3 = pn.notice_text("named", task).split("\n\n")[2]
        p3 = pn.notice_text("placebo", task).split("\n\n")[2]
        assert n3 == p3, task


def test_normalize_task_matches_the_trainers_rule():
    assert pn.normalize_task("AlfWorld/put") == "alfworld"
    assert pn.normalize_task("webshop_v1") == "webshop"
    assert pn.normalize_task("search-nq") == "search"
    assert pn.normalize_task(None) is None


# ----------------------------------------------------------------- the config
def _cfg(**kw):
    from omegaconf import OmegaConf
    base = {"enable": True, "variant": "named", "apply_to": ["student"],
            "doc_sha256": {t: DESIGN_TABLE[("named", t)] for t in pn.TASKS}}
    base.update(kw)
    return OmegaConf.create(base)


def test_disabled_or_absent_is_the_control():
    assert pn.parse_notice_config(None) is None
    assert pn.parse_notice_config(_cfg(enable=False)) is None


def test_an_enabled_notice_shown_to_nobody_is_refused():
    with pytest.raises(AssertionError, match="shown to nobody"):
        pn.parse_notice_config(_cfg(apply_to=[]))


def test_unknown_targets_are_refused():
    with pytest.raises(AssertionError, match="unknown targets"):
        pn.parse_notice_config(_cfg(apply_to=["critic"]))


def test_hash_pins_are_verified_and_a_drift_is_refused():
    nc = pn.parse_notice_config(_cfg())
    pn.verify_doc_hashes(nc)  # the real hashes pass
    bad = pn.parse_notice_config(_cfg(doc_sha256={**nc.doc_sha256, "search": "000000000000"}))
    with pytest.raises(AssertionError, match="drifted"):
        pn.verify_doc_hashes(bad)
    short = pn.parse_notice_config(_cfg(doc_sha256={**nc.doc_sha256, "search": "35500e"}))
    with pytest.raises(AssertionError, match="too short"):
        pn.verify_doc_hashes(short)


def test_the_placebo_pins_are_different_hashes():
    """A lock that pinned the named hashes on the placebo arm would pass the
    identity check while running the wrong text; the two sets must not overlap."""
    named = {pn.notice_sha256("named", t) for t in pn.TASKS}
    placebo = {pn.notice_sha256("placebo", t) for t in pn.TASKS}
    assert not (named & placebo)


# ------------------------------------------------------------- the token prefix
class _QwenLikeTokenizer:
    """The Qwen3 template's system/user rendering, with a toy vocabulary."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kw):
        out = ""
        for m in messages:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out

    def encode(self, text, add_special_tokens=False):
        # whitespace-and-punctuation split, deterministic ids
        toks = text.replace("<|im_start|>", " <|im_start|> ").replace("<|im_end|>", " <|im_end|> ").split()
        return [(hash(t) % 50000) + 1 for t in toks]


def test_system_block_is_the_exact_render_difference():
    tok = _QwenLikeTokenizer()
    block = pn.system_block(tok, "NOTICE")
    assert block == "<|im_start|>system\nNOTICE<|im_end|>\n"


def test_prefix_ids_are_the_head_of_the_full_prompt():
    tok = _QwenLikeTokenizer()
    pre = pn.notice_prefix_ids(tok, "named", "alfworld")
    full = tok.encode(tok.apply_chat_template(
        [{"role": "system", "content": pn.notice_text("named", "alfworld")},
         {"role": "user", "content": "hello"}]))
    assert full[: len(pre)] == pre


@pytest.mark.skipif(
    not os.path.isdir(_REAL_QWEN3_DIR),
    reason="needs the Qwen3 tokenizer on disk",
)
def test_prefix_property_holds_under_the_real_qwen3_template():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_REAL_QWEN3_DIR)
    kw = {"enable_thinking": False}
    for variant in pn.VARIANTS:
        for task in pn.TASKS:
            pre = pn.notice_prefix_ids(tok, variant, task, kw)
            usr = {"role": "user", "content": "go to drawer 1"}
            full = tok.encode(tok.apply_chat_template(
                [{"role": "system", "content": pn.notice_text(variant, task)}, usr],
                add_generation_prompt=True, tokenize=False, **kw), add_special_tokens=False)
            only = tok.encode(tok.apply_chat_template([usr], add_generation_prompt=True, tokenize=False, **kw),
                              add_special_tokens=False)
            assert full == pre + only, (variant, task)
            assert 140 <= len(pre) <= 205, (variant, task, len(pre))   # the design table's range


# ------------------------------------------------------------ prepend / strip
def _rows():
    pad = 0
    ids = torch.tensor([[pad, pad, 11, 12, 13, 14],
                        [pad, 21, 22, 23, 24, 25]])
    mask = (ids != pad).long()
    return ids, mask, pad


def test_prepend_keeps_rows_end_aligned_and_rebuilds_positions():
    ids, mask, pad = _rows()
    out, m, pos = pn.prepend_prefix(ids, mask, [[7, 8, 9], [7]], pad)
    assert out.shape == (2, 9)
    assert out[0].tolist()[-7:] == [7, 8, 9, 11, 12, 13, 14]
    assert out[1].tolist()[-6:] == [7, 21, 22, 23, 24, 25]
    assert m.sum(-1).tolist() == [7, 6]
    assert pos[0].tolist()[-7:] == list(range(7))
    assert pos[1].tolist()[-6:] == list(range(6))
    # the response tail is still the row's last tokens, untouched
    assert out[:, -3:].tolist() == ids[:, -3:].tolist()


def test_strip_is_the_inverse_of_prepend_up_to_width():
    ids, mask, pad = _rows()
    pre = [[7, 8, 9], [7]]
    out, m, _ = pn.prepend_prefix(ids, mask, pre, pad)
    back, bm, _ = pn.strip_prefix(out, m, torch.tensor([3, 1]), pad)
    for i in range(2):
        assert back[i][bm[i].bool()].tolist() == ids[i][mask[i].bool()].tolist()


def test_fingerprint_adjust_is_exactly_what_prepending_adds():
    ids, mask, pad = _rows()
    pre = [[7, 8, 9], [7]]
    out, m, _ = pn.prepend_prefix(ids, mask, pre, pad)
    before = row_fingerprint(ids, mask)
    after = row_fingerprint(out, m)
    adj = torch.tensor([pn.fingerprint_adjust(p) for p in pre], dtype=after.dtype)
    assert torch.equal(after - adj, before)


def test_strip_with_zero_notice_is_the_identity():
    ids, mask, pad = _rows()
    back, bm, _ = pn.strip_prefix(ids, mask, torch.tensor([0, 0]), pad)
    assert torch.equal(back, ids) and torch.equal(bm, mask)


# --------------------------------------------------------------- the floor
def test_leak_patterns_name_the_other_tasks_syntax_only():
    assert pn.leak_flags(["<action>go to drawer 1</action>"], "alfworld") == [False]
    assert pn.leak_flags(["<search>who is</search>"], "alfworld") == [True]
    assert pn.leak_flags(["search[red shoes]"], "alfworld") == [True]
    assert pn.leak_flags(["<action>search[red shoes]</action>"], "webshop") == [False]
    assert pn.leak_flags(["<answer>Paris</answer>"], "webshop") == [True]
    assert pn.leak_flags(["<search>q</search><answer>a</answer>"], "search") == [False]
    assert pn.leak_flags(["<action>go to</action>"], "search") == [True]
    # ordinary English "go to" is NOT leakage -- the word list was dropped for this
    assert pn.leak_flags(["I will go to the next step of reasoning."], "search") == [False]
