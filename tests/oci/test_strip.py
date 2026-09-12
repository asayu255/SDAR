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
from verl.trainer.ppo.oci_reachability import splice_span, wrong_plan_strip_fn, strippable_rows

CKPT = "/opt1/ohara/offline_ladder/probe_hf/klwctl_step300"
PAD = 0
ok = True

# --- the arithmetic, on the REAL post-generation layout ---------------------
# vllm_rollout writes input_ids = cat([prompt, response]) with the PROMPT
# LEFT-PADDED and the RESPONSE RIGHT-PADDED, and states it:
#   attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
# so the live tokens are NOT one block at the right-hand end. A version of
# strip_span that gathered every live token and right-aligned the result slid
# the response rightwards by however much trailing padding it had, and the
# forward reads logits at [-response_length-1:-1] -- so every log-prob came from
# the wrong position while the shape check passed and rho looked plausible.
PL, RL = 8, 8


def row(prompt_toks, response_toks):
    ids = torch.zeros(PL + RL, dtype=torch.long)
    am = torch.zeros(PL + RL, dtype=torch.long)
    pos = torch.zeros(PL + RL, dtype=torch.long)
    lp = len(prompt_toks)
    ids[PL - lp:PL] = torch.tensor(prompt_toks)
    am[PL - lp:PL] = 1
    pos[PL - lp:PL] = torch.arange(lp)
    ids[PL:PL + len(response_toks)] = torch.tensor(response_toks)
    am[PL:PL + len(response_toks)] = 1
    pos[PL:] = torch.arange(lp, lp + RL)
    return ids, am, pos


rows_spec = [
    ([101, 102, 103, 104], [201, 202, 203]),      # short response -> trailing pad
    ([111, 112, 113, 114, 115], [211, 212]),
    ([121, 122, 123], [221, 222, 223, 224, 225, 226, 227, 228]),  # full response
]
ids = torch.stack([row(p, r)[0] for p, r in rows_spec])
mask = torch.stack([row(p, r)[1] for p, r in rows_spec])
posi = torch.stack([row(p, r)[2] for p, r in rows_spec])
responses = ids[:, PL:].clone()

# remove 2 prompt tokens from offset 1 on row 0, 2 from offset 2 on row 1, none row 2
off = torch.tensor([1, 2, 0])
ln = torch.tensor([2, 2, 0])
out_ids, out_mask, out_pos = splice_span(ids, mask, off, ln, torch.zeros(len(rows_spec), 4, dtype=torch.long),
                                          torch.zeros(len(rows_spec), dtype=torch.long), PAD,
                                          response_length=RL, position_ids=posi)

# THE INVARIANT THE WHOLE MEASUREMENT RESTS ON.
good = torch.equal(out_ids[:, PL:], responses) and torch.equal(out_ids[:, -RL:], responses)
ok &= good
print(("  OK  " if good else "  FAIL") +
      " the response region is byte-identical, so input_ids[:, -response_length:] "
      "still equals responses")

good = torch.equal(out_mask[:, PL:], mask[:, PL:])
ok &= good
print(("  OK  " if good else "  FAIL") + " the response's attention mask is untouched")

live = [out_ids[i, :PL][out_mask[i, :PL].bool()].tolist() for i in range(len(rows_spec))]
want = [[101, 104], [111, 112, 115], [121, 122, 123]]
good = live == want
ok &= good
print(("  OK  " if good else "  FAIL") + f" prompt spans after removal: {live}")

good = all(int(out_mask[i, :PL].sum()) == len(want[i]) for i in range(len(rows_spec)))
ok &= good
print(("  OK  " if good else "  FAIL") + " the prompt stays left-padded inside its own region")

# position ids: prompt renumbered from 0, response shifted down by exactly what
# was removed from the prompt in front of it
good = all(
    out_pos[i, :PL][out_mask[i, :PL].bool()].tolist() == list(range(len(want[i])))
    for i in range(len(rows_spec)))
ok &= good
print(("  OK  " if good else "  FAIL") + " prompt position ids restart at 0 on the live span")

good = all(torch.equal(out_pos[i, PL:], (posi[i, PL:] - int(ln[i])).clamp(min=0))
           for i in range(len(rows_spec)))
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" response position ids drop by exactly the tokens removed {ln.tolist()}")

# and what the old right-aligning version produced on row 0, so the test is
# known to be able to fail
live0 = ids[0][mask[0].bool()]
old_keep = torch.cat([live0[:1], live0[3:]])
old_out = torch.full((PL + RL,), PAD, dtype=torch.long)
old_out[PL + RL - old_keep.numel():] = old_keep
good = not torch.equal(old_out[-RL:], responses[0])
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" the old right-align gives {old_out[-RL:].tolist()} against a responses "
      f"of {responses[0].tolist()}")

# an offset that does not fit leaves the row alone rather than corrupting it
bad_ids, bad_mask, _ = splice_span(ids, mask, torch.tensor([4, 0, 0]),
                                   torch.tensor([9, 0, 0]),
                                   torch.zeros(3, 4, dtype=torch.long),
                                   torch.zeros(3, dtype=torch.long), PAD,
                                   response_length=RL, position_ids=posi)
good = torch.equal(bad_ids[0], ids[0]) and torch.equal(bad_mask[0], mask[0])
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
    # THE REAL TEMPLATE, not a hand-written stand-in. The first version of this
    # test wrote its own observation text and it began with a letter; the real
    # one begins with a NEWLINE, which merges with the chat header's newline into
    # a single "\n\n" token when no plan is in front of it. That merge is the
    # whole reason the edit is a replacement rather than a deletion, so a test
    # whose observation lacks the leading newline passes while production files
    # every candidate row unstrippable.
    from agent_system.environments.env_manager import ALFWORLD_TEMPLATE_NO_HIS
    OBS = ALFWORLD_TEMPLATE_NO_HIS.format(
        current_observation="You are in the middle of a room. You see a cabinet 1.",
        admissible_actions="'go to cabinet 1'")
    assert OBS.startswith("\n"), "the template's leading newline is what this checks"

    def render(content):
        return tok.apply_chat_template([{"role": "user", "content": content}],
                                       add_generation_prompt=True, tokenize=False)

    ids_w = tok.encode(render(PLAN + OBS), add_special_tokens=False)
    ids_o = tok.encode(render(OBS), add_special_tokens=False)

    # exactly the computation the rollout loop does
    lo = min(len(ids_w), len(ids_o))
    head = 0
    while head < lo and ids_w[head] == ids_o[head]:
        head += 1
    tail = 0
    while (tail < lo - head
           and ids_w[len(ids_w) - 1 - tail] == ids_o[len(ids_o) - 1 - tail]):
        tail += 1
    plan_take = len(ids_w) - head - tail
    plan_repl = ids_o[head:len(ids_o) - tail]
    plan_off = head
    clean = (plan_take > 0 and len(plan_repl) <= 16
             and ids_w[:head] + plan_repl + ids_w[len(ids_w) - tail:] == ids_o)
    plan_len = plan_take

    good = clean
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" the edit reconstructs exactly: take {plan_take} tokens at {plan_off}, "
          f"put back {len(plan_repl)} ({[tok.decode([t]) for t in plan_repl]})")

    # AND IT IS NOT A PURE DELETION, which is what the first version required.
    deletion_only = len(ids_w) > len(ids_o) and head + tail == len(ids_o)
    good = not deletion_only and len(plan_repl) > 0
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" a deletion-only rule would reject this row (head+tail={head+tail}, "
          f"no-plan length {len(ids_o)}) -- the boundary token differs")

    # THE POINT: the span does not start at 0, so a front-strip is wrong.
    good = plan_off > 0
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" the span starts at token {plan_off}, not 0 -- the chat header comes "
          f"first, which is why a length alone cannot locate it")

    # removing the span reproduces the no-plan render exactly
    # A real row: the prompt left-padded in its own region, a response after it.
    P2 = len(ids_w) + 3
    R2 = 4
    r_ids = torch.full((1, P2 + R2), PAD, dtype=torch.long)
    r_mask = torch.zeros((1, P2 + R2), dtype=torch.long)
    r_ids[0, P2 - len(ids_w):P2] = torch.tensor(ids_w)
    r_mask[0, P2 - len(ids_w):P2] = 1
    r_ids[0, P2:P2 + 2] = torch.tensor([9001, 9002])
    r_mask[0, P2:P2 + 2] = 1
    s_ids, s_mask, _ = splice_span(r_ids, r_mask, torch.tensor([plan_off]),
                                   torch.tensor([plan_take]),
                                   torch.tensor([plan_repl + [0] * (16 - len(plan_repl))]),
                                   torch.tensor([len(plan_repl)]), PAD,
                                   response_length=R2)
    stripped = s_ids[0, :P2][s_mask[0, :P2].bool()].tolist()
    good = stripped == ids_o and torch.equal(s_ids[0, P2:], r_ids[0, P2:])
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          " stripping the span reproduces the no-plan render token for token, "
      "and leaves the response region alone")

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
