"""CPU test: the trainer side of algorithm.tied_opsd (verl/trainer/ppo/tied_opsd.py).

WHAT MUST HOLD.
  1. Groups: stuck / saturated / live / other from the real rows, a trajectory won when its best episode
     reward is > 0, padding rows ignored.
  2. The weights: shares and M are discounted means; w_f = M (1 - q_f); w_s = M min(q_s, live / q_s) with
     the guard (q_s in the guard's terms when it is off); nothing before the first live group; the state
     round-trips through state_dict.
  3. The loss: on the teacher's top-k plus a tail bucket, stuck rows 0.5 forward + 0.5 reverse, saturated
     rows reverse, other rows 0; the gradient w.r.t. the student's logits matches finite differences;
     an unscored row (teacher log-prob UNSCORED_LP) contributes 0 and no NaN.
  4. The token marks with the real tokenizer: tags and special tokens never; Search only the query, and
     the answer only after the evidence.
  5. apply_edit: the edit applied to the chosen rows only, the window widened, the response untouched.
"""
import math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from verl.trainer.ppo import tied_opsd as T  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


print("1. groups")
uids = ["g1"] * 4 + ["g2"] * 4 + ["g3"] * 4 + ["g4"] * 2
tuids = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f", "h", "h"]
tasks = ["alfworld"] * 12 + ["search"] * 2
rew = [0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 0, 0, 1, 1]
real = [True] * 13 + [False]
status, task_of, won = T.classify_groups(uids, tuids, tasks, rew, real)
check(status == {"g1": "stuck", "g2": "saturated", "g3": "live", "g4": "other"}
      and won["c"] and not won["a"] and task_of["g4"] == "search",
      f"stuck / saturated / live, and a group with one real trajectory is other ({status})")
z = T.live_abs_z([True, True, False, False], ["e", "e", "f", "f"])
r = np.array([1, 1, 0, 0.0])
want = abs((1 - r.mean()) / (r.std(ddof=1) + 1e-6))
check(len(z) == 2 and abs(z[0] - want) < 1e-9 and abs(z[1] - want) < 1e-9, f"|z| per trajectory, ddof 1 ({z})")

print("2. weights")
ctl = T.TiedController(["alfworld", "search"], retention=0.8, guard=True)
w0 = ctl.weights("alfworld")
check(w0["w_f"] == 0 and w0["w_s"] == 0 and w0["m_defined"] == 0, "nothing before the first live group")
ctl.update({"s1": "stuck", "s2": "stuck", "l1": "live", "t1": "saturated"},
           {"s1": "alfworld", "s2": "alfworld", "l1": "alfworld", "t1": "alfworld"}, {"alfworld": [0.8, 0.8]})
w1 = ctl.weights("alfworld")
check(abs(w1["q_f"] - 0.5) < 1e-12 and abs(w1["q_s"] - 0.25) < 1e-12 and abs(w1["m"] - 0.8) < 1e-12
      and abs(w1["w_f"] - 0.8 * 0.5) < 1e-12 and abs(w1["w_s"] - 0.8 * 0.25) < 1e-12 and w1["guard_bound"] == 0,
      f"one step: q_f 0.5, q_s 0.25, M 0.8 -> w_f 0.4, w_s 0.2 ({w1})")
ctl.update({f"t{i}": "saturated" for i in range(8)} | {"l9": "live"},
           {**{f"t{i}": "alfworld" for i in range(8)}, "l9": "alfworld"}, {"alfworld": [1.0, 1.0]})
w2 = ctl.weights("alfworld")
n_all = 0.8 * 4 + 9
q_s, live = (0.8 * 1 + 8) / n_all, (0.8 * 1 + 1) / n_all
m = (0.8 * 1.6 + 2.0) / (0.8 * 2 + 2)
check(abs(w2["q_s"] - q_s) < 1e-12 and abs(w2["m"] - m) < 1e-12
      and abs(w2["w_s"] - m * min(q_s, live / q_s)) < 1e-12 and w2["guard_bound"] == 1,
      f"discounted counts; the guard binds when q_s^2 > live ({w2['q_s']:.3f}^2 vs {w2['live']:.3f})")
off = T.TiedController(["alfworld"], guard=False)
off.load_state_dict({**ctl.state_dict(), "guard": False})
check(abs(off.weights("alfworld")["w_s"] - m * q_s) < 1e-12, "guard off: w_s = M q_s")
again = T.TiedController(["alfworld", "search"])
again.load_state_dict(ctl.state_dict())
check(again.weights("alfworld") == w2, "state_dict round-trips")
side, w = T.row_sides_and_weights(["s1", "t1", "l1", "s1"], ["alfworld"] * 4,
                                  {"s1": "stuck", "t1": "saturated", "l1": "live"}, {"alfworld": w2},
                                  eligible=[True, True, True, False])
check(side.tolist() == [1, 2, 0, 1] and w[0] == np.float32(w2["w_f"]) and w[1] == np.float32(w2["w_s"])
      and w[2] == 0 and w[3] == 0, "per row: side, and the side's weight where eligible")

print("3. the loss")
torch.manual_seed(0)
V, K = 12, 4
logits_s = torch.randn(3, 2, V, dtype=torch.float64, requires_grad=True)
logits_t = torch.randn(3, 2, V, dtype=torch.float64)
lt = torch.log_softmax(logits_t, -1)
ids = lt.topk(K, -1).indices
side = torch.tensor([1, 2, 0])


def total(ls):
    lp_s = torch.log_softmax(ls, -1).gather(-1, ids)
    lp_t = lt.gather(-1, ids)
    loss, rkl, fkl = T.tied_token_loss(lp_s, lp_t, side)
    return loss.sum(), loss, rkl, fkl


tot, loss, rkl, fkl = total(logits_s)
ps, pt = torch.softmax(logits_s, -1).detach(), torch.softmax(logits_t, -1)


def kl_partition(p, q):
    a, b = p.gather(-1, ids), q.gather(-1, ids)
    ta, tb = 1 - a.sum(-1), 1 - b.sum(-1)
    return (a * (a.log() - b.log())).sum(-1) + ta * (ta.log() - tb.log())


check(torch.allclose(rkl.detach(), kl_partition(ps, pt)) and torch.allclose(fkl.detach(), kl_partition(pt, ps)),
      "reverse and forward KL on the teacher's top-k + tail")
check(torch.allclose(loss[0].detach(), 0.5 * fkl[0].detach() + 0.5 * rkl[0].detach())
      and torch.allclose(loss[1].detach(), rkl[1].detach()) and float(loss[2].abs().sum()) == 0.0,
      "stuck rows mix, saturated rows reverse, other rows 0")
tot.backward()
g = logits_s.grad.clone()
eps, num = 1e-6, torch.zeros_like(g)
for idx in [(0, 0, 3), (0, 1, 7), (1, 0, 0), (1, 1, 11), (2, 0, 5)]:
    e = torch.zeros_like(logits_s)
    e[idx] = eps
    num[idx] = (total(logits_s.detach() + e)[0] - total(logits_s.detach() - e)[0]) / (2 * eps)
    check(abs(float(num[idx] - g[idx])) < 1e-6, f"gradient at {idx}: analytic {float(g[idx]):+.6f} = numeric {float(num[idx]):+.6f}")
lp_s = torch.log_softmax(torch.randn(1, 2, V), -1).gather(-1, torch.zeros(1, 2, K, dtype=torch.long))
lp_t = torch.full((1, 2, K), T.UNSCORED_LP)
l_un, _, _ = T.tied_token_loss(lp_s, lp_t, torch.tensor([1]))
check(torch.isfinite(l_un).all() and float((l_un * 0).sum()) == 0.0, "an unscored row is finite, and x0 is 0")

print("4. token marks, real tokenizer")
CKPT = "/opt1/ohara/offline_ladder/probe_hf/klwctl_step300"
if not os.path.isdir(CKPT):
    print("SKIP: no tokenizer")
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT)
    masker = T.TokenMasker(tok)

    def marked(text, task, ev):
        ids_ = tok.encode(text, add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
        m_ = masker.row(ids_, task, ev)
        kept = "".join(masker.pieces[i] for i, k_ in zip(ids_, m_) if k_)
        return kept, m_

    alf = "<think>\nThe apple is on the sink.\n</think>\n<action>go to sinkbasin 1</action>"
    kept, m_ = marked(alf, "alfworld", False)
    check("<" not in kept and ">" not in kept and "apple is on the sink" in kept and "go to sinkbasin 1" in kept
          and m_[-1] == 0, f"alfworld: every tag and <|im_end|> out, the rest in ({kept!r})")
    sr = "<think>\nI should look it up.\n</think>\n<search> capital of the Comoros </search>"
    kept, _ = marked(sr, "search", False)
    check(kept.strip() == "capital of the Comoros", f"search: only the query ({kept!r})")
    sa = "<think>\nDoc 2 says Moroni.\n</think>\n<answer> Moroni </answer>"
    kept_y, _ = marked(sa, "search", True)
    kept_n, _ = marked(sa, "search", False)
    check(kept_y.strip() == "Moroni" and kept_n.strip() == "", f"search: the answer only after the evidence ({kept_y!r}, {kept_n!r})")

print("5. apply_edit")
W, R = 10, 3
ids = torch.zeros((2, W + R), dtype=torch.long)
am = torch.zeros((2, W + R), dtype=torch.long)
for i, toks in enumerate([[5, 6, 7, 8], [5, 6, 7, 8]]):
    ids[i, W - len(toks):W] = torch.tensor(toks)
    am[i, W - len(toks):W] = 1
ids[:, W:] = 9
am[:, W:] = 1
repl = torch.zeros((2, 4), dtype=torch.long)
repl[:, :3] = torch.tensor([1, 2, 3])
nid, nam, _, okv = T.apply_edit(ids, am, None, R, torch.tensor([1, 1]), torch.tensor([1, 1]), repl,
                                torch.tensor([3, 3]), 0, rows=[True, False])
live0 = nid[0, :-R][nam[0, :-R].bool()].tolist()
live1 = nid[1, :-R][nam[1, :-R].bool()].tolist()
check(live0 == [5, 1, 2, 3, 7, 8] and live1 == [5, 6, 7, 8] and nid[:, -R:].eq(9).all()
      and okv.tolist() == [True, False] and nid.shape[1] == W + R + 2,
      "row 0 edited (window +2), row 1 untouched, responses intact")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
