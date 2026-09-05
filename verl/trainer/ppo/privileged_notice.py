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
"""The privileged multitask notice: one system message, shown during training only.

Design and pre-registration: ``docs/privileged_multitask_notice_design.md``.

WHAT IT IS. A short system-role document telling the model that one set of
weights is being trained on ALFWorld, WebShop and Search at once, and that it
must use only the vocabulary this environment established. Two variants:

    named    names the other two environments and says why they matter
    placebo  the same avoid-list with the multi-task framing and the naming
             removed -- the control, so the contrast isolates *knowing that other
             tasks are co-trained* from *being told what to avoid*

and two places it can be shown, selectable per run (``apply_to``):

    student  the notice is in the STUDENT's prompt. It is then in the rollout,
             so the trajectories change, and the on-task teacher -- which scores
             the student's own (prompt, response) -- sees it too. Shared context.
    teacher  the notice is prepended to the ON-TASK TEACHER's input only. The
             student's ``input_ids`` are untouched, the rollout is the control's,
             and only the KL target moves. The base policy and the off-task
             teachers never see it in either mode.

WHY ONE TEXT SERVES BOTH MODES. Under Qwen3's chat template the system block is
an exact prefix, as a string and as tokens: ``template([system, user]) ==
"<|im_start|>system\\n{text}<|im_end|>\\n" + template([user])``, checked at
startup by :func:`system_block`. So the student-mode message and the teacher-mode
token prefix are the same ids, and the fingerprint adjustment below is exact.

THE TEXT IS THE MECHANISM, so it is pinned by hash in the intent lock
(``doc_sha256``) and :func:`verify_doc_hashes` refuses a run whose text drifted.
It is also why the six texts live in code rather than a data file: a data file is
a thing one edits.

THE PREMISE IS MEASURED NULL AND THE RUN PROCEEDS ANYWAY (design section 0).
Across 138,109 saved validation responses the rate at which any task emits
another task's action syntax is exactly zero, and the off-task ladder says the
student is already further from the other teachers than its own teacher is. The
operator's reading (2026-09-05) is that the notice is not about suppressing
those tokens, and the arm runs regardless; :data:`LEAK_PATTERNS` is kept as the
floor check the design's section 4 asks for.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

VARIANTS = ("named", "placebo")
TASKS = ("alfworld", "search", "webshop")
APPLY_TARGETS = ("teacher", "student")

# --------------------------------------------------------------------------- #
# the six texts (design section 2). Verbatim; the hashes in the lock are of these.
# --------------------------------------------------------------------------- #
_NAMED_ALFWORLD = """One set of weights is being trained to act in three environments at once: ALFWorld (this one: locating and manipulating household objects), WebShop (finding and buying a specified product), and Search (answering a question with a retriever). You are acting in ALFWorld now.

Because the same parameters serve all three, words and assumptions that belong to WebShop or Search can surface here, where they are simply wrong.

Use only what this environment has established: the receptacles, objects and movements ALFWorld names, and the action forms it accepts. Nothing here has a price, a rating, a listing, a search result or a citation, and no question here is answered from a retrieved passage.

Before you act, check that every term you used came from this room's observations or from ALFWorld's own action set. If a phrasing would fit WebShop or Search but not here, drop it."""

_NAMED_WEBSHOP = """One set of weights is being trained to act in three environments at once: ALFWorld (locating and manipulating household objects), WebShop (this one: finding and buying a specified product), and Search (answering a question with a retriever). You are acting in WebShop now.

Because the same parameters serve all three, words and assumptions that belong to ALFWorld or Search can surface here, where they are simply wrong.

Use only what this environment has established: the product attributes, search terms, buttons and page elements WebShop names, and the action forms it accepts. Nothing here is a room, a receptacle or an object you can pick up, heat or cool, and no question here is answered from a retrieved passage.

Before you act, check that every term you used came from this page's observations or from WebShop's own action set. If a phrasing would fit ALFWorld or Search but not here, drop it."""

_NAMED_SEARCH = """One set of weights is being trained to act in three environments at once: ALFWorld (locating and manipulating household objects), WebShop (finding and buying a specified product), and Search (this one: answering a question with a retriever). You are acting in Search now.

Because the same parameters serve all three, words and assumptions that belong to ALFWorld or WebShop can surface here, where they are simply wrong.

Use only what this environment has established: the question, the passages the retriever returns, and the query and answer forms it accepts. Nothing here is a room, a receptacle or a movable object, and nothing here has a price, a rating, a listing or a cart.

Before you act, check that every term you used came from the question, from a retrieved passage, or from Search's own action set. If a phrasing would fit ALFWorld or WebShop but not here, drop it."""

_PLACEBO_ALFWORLD = """One set of weights is being trained to act in ALFWorld: locating and manipulating household objects. You are acting in ALFWorld now.

Words and assumptions that do not belong to this environment can surface here, where they are simply wrong.

Use only what this environment has established: the receptacles, objects and movements ALFWorld names, and the action forms it accepts. Nothing here has a price, a rating, a listing, a search result or a citation, and no question here is answered from a retrieved passage.

Before you act, check that every term you used came from this room's observations or from ALFWorld's own action set. If a phrasing would not fit here, drop it."""

_PLACEBO_WEBSHOP = """One set of weights is being trained to act in WebShop: finding and buying a specified product. You are acting in WebShop now.

Words and assumptions that do not belong to this environment can surface here, where they are simply wrong.

Use only what this environment has established: the product attributes, search terms, buttons and page elements WebShop names, and the action forms it accepts. Nothing here is a room, a receptacle or an object you can pick up, heat or cool, and no question here is answered from a retrieved passage.

Before you act, check that every term you used came from this page's observations or from WebShop's own action set. If a phrasing would not fit here, drop it."""

_PLACEBO_SEARCH = """One set of weights is being trained to act in Search: answering a question with a retriever. You are acting in Search now.

Words and assumptions that do not belong to this environment can surface here, where they are simply wrong.

Use only what this environment has established: the question, the passages the retriever returns, and the query and answer forms it accepts. Nothing here is a room, a receptacle or a movable object, and nothing here has a price, a rating, a listing or a cart.

Before you act, check that every term you used came from the question, from a retrieved passage, or from Search's own action set. If a phrasing would not fit here, drop it."""

NOTICE_TEXTS: Dict[str, Dict[str, str]] = {
    "named": {"alfworld": _NAMED_ALFWORLD, "webshop": _NAMED_WEBSHOP, "search": _NAMED_SEARCH},
    "placebo": {"alfworld": _PLACEBO_ALFWORLD, "webshop": _PLACEBO_WEBSHOP, "search": _PLACEBO_SEARCH},
}


def normalize_task(task_name) -> Optional[str]:
    """The trainer's rule (``ray_trainer.normalize_task_name``), restated here so
    the rollout loop, the driver and the actor pick the same text for a row."""
    if task_name is None:
        return None
    name = str(task_name).lower()
    for task in TASKS:
        if task in name:
            return task
    return name


def notice_text(variant: str, task: str) -> str:
    assert variant in VARIANTS, f"variant {variant!r} not in {VARIANTS}"
    task = normalize_task(task)
    assert task in TASKS, f"no notice for task {task!r}; have {TASKS}"
    return NOTICE_TEXTS[variant][task]


def notice_sha256(variant: str, task: str) -> str:
    return hashlib.sha256(notice_text(variant, task).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NoticeConfig:
    variant: str
    apply_to: frozenset
    doc_sha256: Dict[str, str]
    effect_probe_every: int = 5

    @property
    def to_student(self) -> bool:
        return "student" in self.apply_to

    @property
    def to_teacher(self) -> bool:
        return "teacher" in self.apply_to


def parse_notice_config(cfg) -> Optional[NoticeConfig]:
    """``None`` when the block is absent or ``enable`` is false -- the control.

    Refuses an ``apply_to`` naming neither target: an enabled notice shown to
    nobody would train the control while reporting the arm's identity.
    """
    if cfg is None or not bool(cfg.get("enable", False)):
        return None
    variant = str(cfg.get("variant", "named"))
    assert variant in VARIANTS, f"privileged_notice.variant={variant!r}; expected one of {VARIANTS}"
    raw = cfg.get("apply_to", None)
    targets = frozenset(str(x) for x in (raw or []))
    bad = targets - set(APPLY_TARGETS)
    assert not bad, f"privileged_notice.apply_to has unknown targets {sorted(bad)}; allowed {APPLY_TARGETS}"
    assert targets, (
        "privileged_notice.enable=true but apply_to is empty: the notice would be shown to "
        "nobody and the run would be the control under the arm's name. Set apply_to to "
        "[student], [teacher] or both, or set enable=false"
    )
    sha = cfg.get("doc_sha256", None)
    sha = {str(k): str(v) for k, v in (dict(sha) if sha is not None else {}).items()}
    return NoticeConfig(
        variant=variant, apply_to=targets, doc_sha256=sha,
        effect_probe_every=int(cfg.get("effect_probe_every", 5)),
    )


def verify_doc_hashes(nc: NoticeConfig) -> None:
    """The lock pins a hash per task; the run refuses to start on a mismatch.

    A prefix of the hex digest is accepted (the design table prints 12 hex chars)
    so long as it is at least 12 characters -- shorter would not be a pin.
    """
    assert set(nc.doc_sha256) == set(TASKS), (
        f"privileged_notice.doc_sha256 must pin exactly {TASKS}; got {sorted(nc.doc_sha256)}"
    )
    for task, pinned in nc.doc_sha256.items():
        assert len(pinned) >= 12, f"doc_sha256.{task}={pinned!r} is too short to be a pin (need >= 12 hex chars)"
        actual = notice_sha256(nc.variant, task)
        assert actual.startswith(pinned), (
            f"privileged_notice text for ({nc.variant}, {task}) has drifted: the lock pins "
            f"{pinned}, the code's text hashes to {actual[:12]}. The text IS the mechanism; "
            "either the lock or the text was edited without the other"
        )


# --------------------------------------------------------------------------- #
# the token prefix
# --------------------------------------------------------------------------- #
def system_block(tokenizer, text: str, chat_kwargs: Optional[dict] = None) -> str:
    """The exact string the chat template adds for a system message.

    Computed as a difference of two renders rather than assumed, and checked:
    the with-system render must END with the user-only render, or the notice is
    not a prefix under this template and neither mode below is sound.
    """
    kw = dict(chat_kwargs or {})
    kw.setdefault("add_generation_prompt", True)
    kw["tokenize"] = False
    probe = {"role": "user", "content": "probe"}
    with_sys = tokenizer.apply_chat_template([{"role": "system", "content": text}, probe], **kw)
    without = tokenizer.apply_chat_template([probe], **kw)
    assert with_sys.endswith(without), (
        "the chat template does not render a system message as a pure prefix of the "
        "user-only prompt; the privileged notice cannot be prepended for the teacher "
        "consistently with how the student sees it"
    )
    return with_sys[: len(with_sys) - len(without)]


def notice_prefix_ids(tokenizer, variant: str, task: str, chat_kwargs: Optional[dict] = None) -> List[int]:
    """Token ids of the system block, as they will appear at the head of a prompt."""
    block = system_block(tokenizer, notice_text(variant, task), chat_kwargs)
    return list(tokenizer.encode(block, add_special_tokens=False))


def fingerprint_adjust(prefix_ids: Sequence[int]) -> int:
    """What prepending ``prefix_ids`` adds to :func:`teacher_cache.row_fingerprint`.

    The fingerprint is ``sum(ids) * 1000003 + count`` over the unpadded row, so a
    prefix adds exactly this and the worker can subtract it to hash the row the
    STUDENT holds -- which is what the actor will compare against.
    """
    return int(sum(int(t) for t in prefix_ids)) * 1000003 + len(prefix_ids)


def prepend_prefix(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                   prefix_ids: Sequence[Sequence[int]], pad_token_id: int
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Put a per-row prefix between the left padding and the row's first real token.

    EVERY OTHER COLUMN KEEPS ITS ROLE. A row here is
    ``[left pad | prompt | response | right pad]`` and the readers downstream are
    anchored to the END of it: the worker takes the response window as
    ``attention_mask[:, -response_length-1:-1]`` and the teacher cache REQUIRES
    that window to be ``[live..., pad...]`` -- a prefix. So the prefix is written
    into left padding the row already has, in place, and the tail is not touched.

    Repacking the live tokens right-aligned instead (what the first version did)
    deletes each row's right padding, which slides the response out of its window:
    the cache then either refuses the batch ("this batch has holes") or, worse,
    stores hidden states taken at prompt positions.

    Only when some row has less left padding than its prefix is the tensor
    widened, by that deficit, on the LEFT -- which leaves the tail where it was.
    With a 4096-wide prompt region and prompts under 2800 tokens this never fires.

    Returns (input_ids, attention_mask, position_ids); position_ids are rebuilt
    from the new mask.
    """
    bs, width = input_ids.shape
    lens = [len(p) for p in prefix_ids]
    assert len(lens) == bs, f"{len(lens)} prefixes for {bs} rows"
    if bs == 0:
        return input_ids.clone(), attention_mask.clone(), torch.zeros_like(input_ids)

    # first live column per row; a fully padded row is treated as having room
    live = attention_mask.bool()
    any_live = live.any(dim=1)
    first = torch.where(any_live, live.float().argmax(dim=1), torch.full((bs,), width))
    deficit = max(0, int(max(int(lens[i]) - int(first[i]) for i in range(bs))))

    out_ids = torch.full((bs, width + deficit), int(pad_token_id), dtype=input_ids.dtype)
    out_mask = torch.zeros((bs, width + deficit), dtype=attention_mask.dtype)
    out_ids[:, deficit:] = input_ids
    out_mask[:, deficit:] = attention_mask
    for i in range(bs):
        n = int(lens[i])
        if n == 0:
            continue
        stop = int(first[i]) + deficit          # the row's first live column, shifted
        out_ids[i, stop - n:stop] = torch.tensor(list(prefix_ids[i]), dtype=input_ids.dtype)
        out_mask[i, stop - n:stop] = 1
    pos = (out_mask.cumsum(-1) - 1).clamp(min=0).to(torch.long)
    return out_ids, out_mask, pos


def strip_prefix(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 notice_len: torch.Tensor, pad_token_id: int
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn each row's first ``notice_len[i]`` live tokens back into left padding.

    The inverse of the student-mode injection, used by the actor's effect probe to
    ask what the same student would have said WITHOUT the notice, on the same
    response tokens. Done in place for the reason prepend_prefix gives: the probe
    compares its output against a forward on the original row position by
    position, so every column after the notice has to stay where it is. Repacking
    the row right-aligned deletes the right padding and slides the response, which
    makes the probe compare two different positions and report a KL of tens of
    nats where the true answer is a fraction of one.
    """
    bs, width = input_ids.shape
    out_ids = input_ids.clone()
    out_mask = attention_mask.clone()
    live = attention_mask.bool()
    any_live = live.any(dim=1)
    first = torch.where(any_live, live.float().argmax(dim=1), torch.full((bs,), width))
    for i in range(bs):
        n = int(notice_len[i])
        if n <= 0:
            continue
        f = int(first[i])
        out_ids[i, f:f + n] = int(pad_token_id)
        out_mask[i, f:f + n] = 0
    pos = (out_mask.cumsum(-1) - 1).clamp(min=0).to(torch.long)
    return out_ids, out_mask, pos



# --------------------------------------------------------------------------- #
# the floor check (design section 4, diagnostic 2)
# --------------------------------------------------------------------------- #
# Unambiguous action syntax of EACH task, so "foreign" is defined per destination
# as the union of the other two. The word-level list the design also tried is
# not used: its one class of hits ("go to" in ordinary English) was not leakage.
_SYNTAX = {
    "alfworld": (r"<action>",),
    "webshop": (r"<action>", r"search\[", r"click\["),
    "search": (r"<search>", r"<answer>"),
}
LEAK_PATTERNS: Dict[str, "re.Pattern"] = {}
for _dst in TASKS:
    _foreign = sorted({p for src, ps in _SYNTAX.items() if src != _dst for p in ps} - set(_SYNTAX[_dst]))
    LEAK_PATTERNS[_dst] = re.compile("|".join(_foreign))


def leak_flags(texts: Iterable[str], task: str) -> List[bool]:
    """Per response: does it contain another task's action syntax?"""
    pat = LEAK_PATTERNS[normalize_task(task)]
    return [bool(pat.search(t or "")) for t in texts]
