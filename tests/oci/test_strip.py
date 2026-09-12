"""CPU test: the plan's token span, and that removing it is the exact inverse.

If it is not, rho is comparing two prompts that differ in more than the
conditioning, and the whole reachability number is meaningless.

THE BUG THIS FILE EXISTS TO CATCH. The first version measured only a LENGTH and
stripped that many leading tokens, copying the student-mode notice, which is a
system message at position 0 and really is a prefix of the render. The corrupted
plan goes at the head of the USER turn's content, behind the chat header, so the
front-strip removed the header and left the plan in place. Every check below
runs the real tokenizer and the real chat template, because a synthetic
token-list round trip passes either way.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from verl.trainer.ppo.oci_reachability import strip_span, wrong_plan_strip_fn, strippable_rows

CKPT = "/opt1/ohara/offline_ladder/probe_hf/klwctl_step300"
PAD = 0
ok = True

# --- the arithmetic, on synthetic rows of different real lengths -------------
rows = [[11, 12, 13, 14, 15], [21, 22, 23, 24, 25, 26], [31, 32, 33]]
W = max(len(r) for r in rows) + 2
ids = torch.full((len(rows), W), PAD, dtype=torch.long)
mask = torch.zeros((len(rows), W), dtype=torch.long)
for i, r in enumerate(rows):
    ids[i, W - len(r):] = torch.tensor(r)
    mask[i, W - len(r):] = 1

# remove [1,3) from row 0, [2,4) from row 1, nothing from row 2
off = torch.tensor([1, 2, 0])
ln = torch.tensor([2, 2, 0])
out_ids, out_mask, out_pos = strip_span(ids, mask, off, ln, PAD)
got = [out_ids[i][out_mask[i].bool()].tolist() for i in range(len(rows))]
want = [[11, 14, 15], [21, 22, 25, 26], [31, 32, 33]]
good = got == want
ok &= good
print(("  OK  " if good else "  FAIL") + f" span removal: {got}")

good = all(int(out_mask[i].sum()) == len(want[i]) for i in range(len(rows)))
ok &= good
print(("  OK  " if good else "  FAIL") + " width kept, live span shortened by the span")

good = all(out_pos[i][out_mask[i].bool()].tolist() == list(range(len(want[i])))
           for i in range(len(rows)))
ok &= good
print(("  OK  " if good else "  FAIL") + " position ids restart at 0 on the live span")

# an offset that does not fit leaves the row alone rather than corrupting it
bad_ids, bad_mask, _ = strip_span(ids, mask, torch.tensor([4, 0, 0]),
                                  torch.tensor([9, 0, 0]), PAD)
good = bad_ids[0][bad_mask[0].bool()].tolist() == rows[0]
ok &= good
print(("  OK  " if good else "  FAIL") + " an out-of-range span is a no-op, not a corruption")

# --- the real thing: chat template, real plan, exact inverse ----------------
if not os.path.isdir(CKPT):
    print(f"  SKIP  no tokenizer at {CKPT}; span-vs-render check not run")
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT)

    PLAN = ("### SOLUTION PLAN (training only) ###\n"
            "A correct high-level plan for this task is:\n"
            "1. GotoLocation(cabinet)\n2. PickupObject(mug)\n"
            "3. PutObject(mug, countertop)\n"
            "Ground each step into one admissible action at a time.\n\n")
    OBS = ("You are in the middle of a room. Looking quickly around you, you see "
           "a cabinet 1, a countertop 1, and a fridge 1.\nYour task is to: put a "
           "clean mug in the countertop.\nAdmissible actions: 'go to cabinet 1'\n")

    def render(content):
        return tok.apply_chat_template([{"role": "user", "content": content}],
                                       add_generation_prompt=True, tokenize=False)

    ids_w = tok.encode(render(PLAN + OBS), add_special_tokens=False)
    ids_o = tok.encode(render(OBS), add_special_tokens=False)

    # exactly the computation the rollout loop does
    head = 0
    while head < len(ids_o) and ids_w[head] == ids_o[head]:
        head += 1
    tail = 0
    while (tail < len(ids_o) - head
           and ids_w[len(ids_w) - 1 - tail] == ids_o[len(ids_o) - 1 - tail]):
        tail += 1
    clean = len(ids_w) > len(ids_o) and head + tail == len(ids_o)
    plan_off, plan_len = (head, len(ids_w) - len(ids_o)) if clean else (0, 0)

    good = clean
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" the two renders differ in ONE contiguous span (off {plan_off}, "
          f"len {plan_len} tokens)")

    # THE POINT: the span does not start at 0, so a front-strip is wrong.
    good = plan_off > 0
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" the span starts at token {plan_off}, not 0 -- the chat header comes "
          f"first, which is why a length alone cannot locate it")

    # removing the span reproduces the no-plan render exactly
    W2 = len(ids_w) + 3
    r_ids = torch.full((1, W2), PAD, dtype=torch.long)
    r_mask = torch.zeros((1, W2), dtype=torch.long)
    r_ids[0, W2 - len(ids_w):] = torch.tensor(ids_w)
    r_mask[0, W2 - len(ids_w):] = 1
    s_ids, s_mask, _ = strip_span(r_ids, r_mask, torch.tensor([plan_off]),
                                  torch.tensor([plan_len]), PAD)
    stripped = s_ids[0][s_mask[0].bool()].tolist()
    good = stripped == ids_o
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          " stripping the span reproduces the no-plan render token for token")

    # what a front-strip would have produced instead
    front = ids_w[plan_len:]
    good = front != ids_o
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" a front-strip of {plan_len} tokens does NOT (first token "
          f"{tok.decode(front[:1])!r} vs {tok.decode(ids_o[:1])!r})")

# --- the strip_fn's own gating ----------------------------------------------
import types, numpy as np

def fake(off, ln, trunc):
    b = {"input_ids": ids, "attention_mask": mask,
         "oci_plan_off": torch.tensor(off), "oci_plan_len": torch.tensor(ln),
         "oci_plan_truncated": torch.tensor(trunc)}
    return types.SimpleNamespace(batch=b, non_tensor_batch={}, meta_info={},
                                 __len__=lambda: len(rows))

b = fake([1, 2, 0], [2, 2, 0], [0, 1, 0])
good = strippable_rows(b).tolist() == [True, False, False]
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" strippable_rows: length 0 and truncated rows are excluded "
      f"({strippable_rows(b).tolist()})")


def test_strip():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
