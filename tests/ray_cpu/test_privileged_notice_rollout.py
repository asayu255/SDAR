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
"""The student-mode notice reaches the prompt the rollout tokenises, per task,
and every row carries notice_len / notice_truncated so collate sees one schema.

A toy tokenizer standing in for Qwen3's: the same chat rendering, whitespace
tokens. What is checked is the plumbing, not the text.
"""
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
from omegaconf import OmegaConf  # noqa: E402

import agent_system.multi_turn_rollout.rollout_loop as rollout_loop  # noqa: E402
from verl import DataProto  # noqa: E402
from verl.trainer.ppo import privileged_notice as pn  # noqa: E402


class _Tok:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kw):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def _ids(self, text):
        toks = text.replace("<|im_start|>", " <|im_start|> ").replace("<|im_end|>", " <|im_end|> ").split()
        return [(hash(t) % 50000) + 1 for t in toks]

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def __call__(self, text, return_tensors="pt", add_special_tokens=False, **kw):
        ids = torch.tensor([self._ids(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _collector(variant, apply_to, max_prompt_length=4096):
    cfg = OmegaConf.create({
        "data": {"max_prompt_length": max_prompt_length, "truncation": "left",
                 "return_raw_chat": True, "apply_chat_template_kwargs": {}},
        "algorithm": {"opd": {"privileged_notice": {
            "enable": variant is not None, "variant": variant or "named", "apply_to": apply_to,
            "doc_sha256": {t: pn.notice_sha256(variant or "named", t)[:12] for t in pn.TASKS}}}},
    })
    return rollout_loop.TrajectoryCollector(config=cfg, tokenizer=_Tok(), processor=None)


def _gen_batch(tasks):
    return DataProto.from_dict(
        tensors={"dummy": torch.zeros(len(tasks))},
        non_tensors={"raw_prompt": np.array([[{"role": "user", "content": "x"}]] * len(tasks), dtype=object),
                     "data_source": np.array(tasks, dtype=object),
                     "task_name": np.array(tasks, dtype=object)},
    )


def _row(collector, task, obs="You are in a room."):
    gb = _gen_batch([task])
    return collector.preprocess_single_sample(item=0, gen_batch=gb, obs={"text": [obs]})


def test_off_means_no_system_message_and_zero_columns():
    c = _collector(None, [])
    row = _row(c, "alfworld")
    assert row["raw_prompt"][0]["role"] == "user"
    assert int(row["notice_len"]) == 0 and int(row["notice_truncated"]) == 0


def test_teacher_only_mode_leaves_the_student_prompt_alone():
    c = _collector("named", ["teacher"])
    row = _row(c, "alfworld")
    assert row["raw_prompt"][0]["role"] == "user" and int(row["notice_len"]) == 0


@pytest.mark.parametrize("variant", ["named", "placebo"])
def test_student_mode_prepends_the_tasks_own_text(variant):
    c = _collector(variant, ["student"])
    for task in pn.TASKS:
        row = _row(c, task)
        assert row["raw_prompt"][0] == {"role": "system", "content": pn.notice_text(variant, task)}
        assert row["raw_prompt"][1]["role"] == "user"
        # notice_len is the block's token count, and the prompt really is longer by it
        plain = _row(_collector(None, []), task)
        assert int(row["notice_len"]) > 0
        assert int(row["attention_mask"].sum()) - int(plain["attention_mask"].sum()) == int(row["notice_len"])
        assert int(row["notice_truncated"]) == 0


def test_the_task_is_normalised_the_way_the_trainer_does_it():
    c = _collector("named", ["student"])
    row = _row(c, "WebShop/electronics")
    assert row["raw_prompt"][0]["content"] == pn.notice_text("named", "webshop")


def test_truncation_is_flagged_when_the_prompt_hits_the_cap():
    c = _collector("named", ["student"], max_prompt_length=60)   # far below notice + obs
    row = _row(c, "alfworld", obs=" ".join(["word"] * 50))
    assert int(row["attention_mask"].sum()) == 60                 # left-truncated to the cap
    assert int(row["notice_truncated"]) == 1


def test_placeholder_rows_carry_the_same_columns():
    c = _collector("named", ["student"])
    live = _row(c, "search")
    template = {k: live[k] for k in ("input_ids", "attention_mask", "position_ids")}
    ph = c._placeholder_single_sample(item=0, gen_batch=_gen_batch(["search"]), obs={"text": ["o"]},
                                      template=template)
    assert int(ph["notice_len"]) == 0 and int(ph["notice_truncated"]) == 0
    assert ph["raw_prompt"][0]["role"] == "system"   # cosmetic, but the same shape as a live row
    # and both rows collate into one schema
    from verl.utils.dataset.rl_dataset import collate_fn
    batch = collate_fn([live, ph])
    assert batch["notice_len"].tolist() == [int(live["notice_len"]), 0]
