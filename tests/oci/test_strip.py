"""CPU test: prepending a plan and stripping it again must be a round trip.

If it is not, rho is comparing two prompts that differ in more than the
conditioning, and the whole reachability number is meaningless.
"""
import os, sys, types
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from verl.trainer.ppo.privileged_notice import prepend_prefix, strip_prefix
from verl.trainer.ppo.oci_reachability import plan_length_column

PAD = 0
ok = True

# rows of different real lengths, left-padded to one width, as the loop has them
rows = [[11, 12, 13], [21, 22, 23, 24, 25], [31, 32]]
W = max(len(r) for r in rows) + 2
ids = torch.full((len(rows), W), PAD, dtype=torch.long)
mask = torch.zeros((len(rows), W), dtype=torch.long)
for i, r in enumerate(rows):
    ids[i, W - len(r):] = torch.tensor(r)
    mask[i, W - len(r):] = 1

# only the last slot carries a plan, as the candidate gate arranges
prefixes = [[], [], [91, 92, 93, 94]]
p_ids, p_mask, p_pos = prepend_prefix(ids, mask, prefixes, pad_token_id=PAD)
lens = torch.tensor([len(p) for p in prefixes], dtype=torch.long)
s_ids, s_mask, s_pos = strip_prefix(p_ids, p_mask, lens, pad_token_id=PAD)

for i, r in enumerate(rows):
    live = s_ids[i][s_mask[i].bool()].tolist()
    good = live == r
    ok &= good
    print(("  OK  " if good else "  FAIL") + f" row {i}: stripped back to {live} (want {r})")

# the untouched rows must be bit-identical through the round trip
same = all(p_ids[i][p_mask[i].bool()].tolist() == rows[i] for i in (0, 1))
ok &= same
print(("  OK  " if same else "  FAIL") + " rows without a plan pass through prepend unchanged")

# position ids are rebuilt from the mask, so they must be contiguous from 0
pos_ok = all(int(s_pos[i][s_mask[i].bool()][0]) == 0 for i in range(len(rows)))
ok &= pos_ok
print(("  OK  " if pos_ok else "  FAIL") + " position ids restart at 0 on the live span")

# the length column must count the plan and nothing else
class Tok:
    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[0] * len(text.split()))

col = plan_length_column(["", "", "a b c d"], Tok())
good = col.tolist() == [0, 0, 4]
ok &= good
print(("  OK  " if good else "  FAIL") + f" plan_length_column -> {col.tolist()}")

# a zero-length strip is the identity
z_ids, z_mask, _ = strip_prefix(ids, mask, torch.zeros(len(rows), dtype=torch.long), PAD)
good = torch.equal(z_ids, ids) and torch.equal(z_mask, mask)
ok &= good
print(("  OK  " if good else "  FAIL") + " stripping nothing is the identity")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
