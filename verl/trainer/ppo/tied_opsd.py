"""Tied-group self-distillation (algorithm.tied_opsd): the driver's bookkeeping and the actor's loss.

THE DESIGN (2026-10-01, artifact "停滞・飽和群の自己蒸留"). GRPO's outcome signal is zero in a group whose
G rollouts all failed (stuck) or all succeeded (saturated). In those groups -- and only there -- the student
is distilled toward a teacher that is ITSELF with the instance's correct document in its prompt:

    stuck rows       w_f * [0.5 KL(p_T || p_S) + 0.5 KL(p_S || p_T)]   (document + the progress sentence)
    saturated rows   w_s * KL(p_S || p_T)                               (document + the efficiency sentence)
    mixed rows       nothing (GRPO alone)

per distilled token, summed and divided by the policy gradient's own per-task reference (the same row weights,
normalize_loss_by_task). The per-token weights come from the task's group shares and the mixed groups' signal:

    M    discounted mean (retention 0.8) of |z| over the live groups' trajectories, z the outcome-only GRPO
         advantage (turn rows, mean and std with ddof 1, +1e-6) -- the push an average mixed-group token gets
    q_f, q_s, live   the discounted shares of stuck / saturated / live groups (same retention)
    w_f = M (1 - q_f)
    w_s = M min(q_s, n_live / n_sat)   the second term is the amount guard, on THIS batch's group counts of
                                       the task (the user, 2026-10-01: "そのバッチの群数で上限を計算する"):
                                       the step's saturated groups never carry more weight in total
                                       (n_sat w_s) than its mixed groups' (n_live M). With no mixed group in
                                       the batch, w_s is 0.

Nothing is distilled in a task before its first live group (M undefined). A row is distilled only when its
group is stuck or saturated, the state pointer placed it (tied_match == 1), and its teacher prompt was
recorded (the document edit, and for a saturated row the sentence edit too). Tokens: never a tag or a
special token; on Search only the query and -- once a result had carried the answer -- the answer.

The KL is computed on the TEACHER's top-k plus one tail bucket (the existing teacher-indexed top-k path):
exact for the forward half, a partition lower bound for the reverse half.
"""
import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TIED_TASKS = ("alfworld", "webshop", "search")
SIDE_NONE, SIDE_STUCK, SIDE_SAT = 0, 1, 2
# Teacher log-prob written at ids a row was not scored at: finite, so 0 * it stays 0 (never NaN).
UNSCORED_LP = -20.0
STATE_VERSION = 1

_TAG = re.compile(r"</?(?:think|action|search|answer)>")
_QUERY = re.compile(r"<search>(.*?)</search>", flags=re.S)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", flags=re.S)


def classify_groups(uids, tuids, tasks, episode_rewards, real) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, bool]]:
    """``(status by uid, task by uid, won by traj_uid)`` from the REAL rows only.

    A trajectory's outcome is its best row's episode reward > 0 (progress_rank's definition, the one the
    traj/<task>/groups_*_share metrics use). A group with fewer than two real trajectories is 'other'.
    """
    best: Dict[str, float] = {}
    traj_uid_group: Dict[str, str] = {}
    task_of: Dict[str, str] = {}
    for u, t, k, r, ok in zip(uids, tuids, tasks, episode_rewards, real):
        if not ok:
            continue
        u, t = str(u), str(t)
        try:
            r = float(r)
        except (TypeError, ValueError):
            r = float("nan")
        if math.isnan(r):
            r = float("-inf")
        best[t] = max(best.get(t, float("-inf")), r)
        traj_uid_group[t] = u
        task_of.setdefault(u, str(k))
    won = {t: r > 0.0 for t, r in best.items()}
    by: Dict[str, List[bool]] = defaultdict(list)
    for t, u in traj_uid_group.items():
        by[u].append(won[t])
    status = {}
    for u, ws in by.items():
        status[u] = ("other" if len(ws) < 2 else "saturated" if all(ws) else "stuck" if not any(ws) else "live")
    return status, task_of, won


def live_abs_z(rows_won: Sequence[bool], rows_traj: Sequence[str]) -> List[float]:
    """|z| per trajectory of one live group: the outcome-only GRPO advantage on turn rows (ddof 1, +1e-6)."""
    r = np.asarray([1.0 if w else 0.0 for w in rows_won], dtype=np.float64)
    if r.size < 2:
        return []
    mean, std = float(r.mean()), float(r.std(ddof=1))
    out, seen = [], set()
    for w, t in zip(rows_won, rows_traj):
        if t in seen:
            continue
        seen.add(t)
        out.append(abs(((1.0 if w else 0.0) - mean) / (std + 1e-6)))
    return out


class TiedController:
    """Per-task discounted shares and M, and the weights they give. state_dict travels with the checkpoint."""

    def __init__(self, tasks: Iterable[str], retention: float = 0.8, guard: bool = True):
        self.tasks = [str(t) for t in tasks]
        self.retention = float(retention)
        self.guard = bool(guard)
        z = {t: 0.0 for t in self.tasks}
        self.n_all, self.n_stuck, self.n_sat, self.n_live = dict(z), dict(z), dict(z), dict(z)
        self.m_sum, self.m_cnt = dict(z), dict(z)

    def update(self, status: Dict[str, str], task_of: Dict[str, str], live_z: Dict[str, List[float]]) -> None:
        """Fold in one step: every group's status, and the live groups' |z| by task."""
        cnt = {t: defaultdict(int) for t in self.tasks}
        for u, s in status.items():
            t = task_of.get(u)
            if t in cnt and s in ("stuck", "saturated", "live"):
                cnt[t][s] += 1
        a = self.retention
        for t in self.tasks:
            c = cnt[t]
            self.n_stuck[t] = a * self.n_stuck[t] + c["stuck"]
            self.n_sat[t] = a * self.n_sat[t] + c["saturated"]
            self.n_live[t] = a * self.n_live[t] + c["live"]
            self.n_all[t] = a * self.n_all[t] + c["stuck"] + c["saturated"] + c["live"]
            zs = live_z.get(t, [])
            self.m_sum[t] = a * self.m_sum[t] + float(sum(zs))
            self.m_cnt[t] = a * self.m_cnt[t] + float(len(zs))

    def shares(self, task) -> Tuple[float, float, float]:
        n = self.n_all.get(task, 0.0)
        if n <= 0:
            return 0.0, 0.0, 0.0
        return self.n_stuck[task] / n, self.n_sat[task] / n, self.n_live[task] / n

    def m(self, task) -> Optional[float]:
        c = self.m_cnt.get(task, 0.0)
        return None if c <= 0 else self.m_sum[task] / c

    def weights(self, task, n_live: Optional[int] = None, n_sat: Optional[int] = None) -> Dict[str, float]:
        """w_f, w_s and what they came from; both 0 before the task's first live group.

        ``n_live`` / ``n_sat``: this batch's live and saturated groups of the task, for the amount guard
        (n_sat w_s <= n_live M). Required when the guard is on.
        """
        q_f, q_s, live = self.shares(task)
        m = self.m(task)
        if m is None:
            return {"w_f": 0.0, "w_s": 0.0, "m": 0.0, "q_f": q_f, "q_s": q_s, "live": live,
                    "guard_bound": 0.0, "m_defined": 0.0}
        w_f = m * (1.0 - q_f)
        free = q_s
        if self.guard:
            if n_live is None or n_sat is None:
                raise ValueError("tied_opsd: the amount guard needs this batch's n_live and n_sat")
            bound = (float(n_live) / float(n_sat)) if n_sat > 0 else float("inf")
        else:
            bound = float("inf")
        w_s = m * min(free, bound)
        return {"w_f": w_f, "w_s": w_s, "m": m, "q_f": q_f, "q_s": q_s, "live": live,
                "guard_bound": float(bound < free), "m_defined": 1.0}

    def state_dict(self) -> dict:
        return {"version": STATE_VERSION, "retention": self.retention, "guard": self.guard,
                "n_all": self.n_all, "n_stuck": self.n_stuck, "n_sat": self.n_sat, "n_live": self.n_live,
                "m_sum": self.m_sum, "m_cnt": self.m_cnt}

    def load_state_dict(self, sd: dict) -> None:
        if int(sd.get("version", -1)) != STATE_VERSION:
            raise ValueError(f"tied_opsd state version {sd.get('version')} != {STATE_VERSION}")
        if abs(float(sd.get("retention", -1)) - self.retention) > 1e-12 or bool(sd.get("guard")) != self.guard:
            raise ValueError("tied_opsd state was saved under another retention/guard")
        for k in ("n_all", "n_stuck", "n_sat", "n_live", "m_sum", "m_cnt"):
            getattr(self, k).update({t: float(v) for t, v in dict(sd[k]).items() if t in self.tasks})


def row_sides_and_weights(uids, tasks, status: Dict[str, str], weights_by_task: Dict[str, Dict[str, float]],
                          eligible) -> Tuple[np.ndarray, np.ndarray]:
    """Per row: side (0 none, 1 stuck, 2 saturated) and weight (w_f / w_s, 0 where not eligible)."""
    n = len(uids)
    side = np.zeros(n, dtype=np.int64)
    w = np.zeros(n, dtype=np.float32)
    for i, (u, t) in enumerate(zip(uids, tasks)):
        s = status.get(str(u))
        if s == "stuck":
            side[i] = SIDE_STUCK
        elif s == "saturated":
            side[i] = SIDE_SAT
        else:
            continue
        if not eligible[i]:
            continue
        ws = weights_by_task.get(str(t))
        if ws is None:
            continue
        w[i] = ws["w_f"] if side[i] == SIDE_STUCK else ws["w_s"]
    return side, w


class TokenMasker:
    """Which response tokens are distilled: never a tag or a special token; on Search only the query
    and, when a result had carried the answer before this turn, the answer."""

    def __init__(self, tokenizer, special_from: Optional[int] = None):
        n = len(tokenizer)
        self.pieces = tokenizer.batch_decode([[i] for i in range(n)], skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)
        base = int(special_from if special_from is not None else getattr(tokenizer, "vocab_size", n))
        special = set(int(x) for x in (getattr(tokenizer, "all_special_ids", None) or []))
        self.special = np.zeros(n + 1, dtype=bool)
        self.special[base:] = True
        for x in special:
            if 0 <= x < n:
                self.special[x] = True

    def row(self, ids: Sequence[int], task: str, evidence: bool) -> np.ndarray:
        ids = [int(x) for x in ids]
        m = np.ones(len(ids), dtype=np.float32)
        if not ids:
            return m
        starts, pos = [], 0
        for x in ids:
            starts.append(pos)
            pos += len(self.pieces[x]) if 0 <= x < len(self.pieces) else 0
        ends = starts[1:] + [pos]
        text = "".join(self.pieces[x] if 0 <= x < len(self.pieces) else "" for x in ids)
        for j, x in enumerate(ids):
            if x >= len(self.special) - 1 or self.special[x]:
                m[j] = 0.0
        tags = [(a.start(), a.end()) for a in _TAG.finditer(text)]

        def overlaps(j, spans):
            return any(starts[j] < e and ends[j] > s for s, e in spans)

        for j in range(len(ids)):
            if m[j] and overlaps(j, tags):
                m[j] = 0.0
        if task == "search":
            keep = [(a.start(1), a.end(1)) for a in _QUERY.finditer(text)]
            if evidence:
                keep += [(a.start(1), a.end(1)) for a in _ANSWER.finditer(text)]
            for j in range(len(ids)):
                if m[j] and not overlaps(j, keep):
                    m[j] = 0.0
        return m


def tied_token_loss(student_lp, teacher_lp, side, eps: float = 1e-8):
    """``(loss, rkl, fkl)`` per token, (bs, L), on the teacher's top-k support plus one tail bucket.

    ``student_lp`` / ``teacher_lp``: (bs, L, k) full-vocabulary log-probs at the teacher's top-k ids; only the
    student carries gradient. ``side``: (bs,) 1 stuck -> 0.5 forward + 0.5 reverse; 2 saturated -> reverse;
    anything else -> 0.
    """
    import torch

    teacher_lp = teacher_lp.detach().to(student_lp.dtype)
    p_s, p_t = student_lp.exp(), teacher_lp.exp()
    tail_s = (1.0 - p_s.sum(-1)).clamp(min=eps, max=1.0)
    tail_t = (1.0 - p_t.sum(-1)).clamp(min=eps, max=1.0)
    rkl = (p_s * (student_lp - teacher_lp)).sum(-1) + tail_s * (tail_s.log() - tail_t.log())
    fkl = (p_t * (teacher_lp - student_lp)).sum(-1) + tail_t * (tail_t.log() - tail_s.log())
    side = side.reshape(-1, 1).to(student_lp.device)
    zero = torch.zeros_like(rkl)
    loss = torch.where(side == SIDE_STUCK, 0.5 * fkl + 0.5 * rkl, torch.where(side == SIDE_SAT, rkl, zero))
    return loss, rkl, fkl


def apply_edit(input_ids, attention_mask, position_ids, response_length: int, off, take, repl, rlen,
               pad_token_id: int, rows=None):
    """One recorded prompt edit applied to ``rows`` (all rows when None): ``(ids, mask, pos, ok)``.

    The same widening and the same verified splice as oci_rank.with_document -- the prompt window first
    grows by the largest growth among the edited rows (left padding costs nothing at the forward), then
    oci_reachability.splice_span replaces live prompt tokens [off, off+take) with repl[:rlen]. A row not in
    ``rows`` is passed through (its take is zeroed). ``ok`` says which edited rows took the edit exactly.
    """
    import torch

    from verl.trainer.ppo.oci_reachability import splice_span

    off = off.reshape(-1).long()
    take = take.reshape(-1).long()
    rlen = rlen.reshape(-1).long()
    if rows is not None:
        rows_t = torch.as_tensor(np.asarray(rows, dtype=bool), device=take.device)
        take = torch.where(rows_t, take, torch.zeros_like(take))
    ids, am, pos = input_ids, attention_mask, position_ids
    plen = int(ids.shape[1]) - int(response_length)
    grow = int(torch.clamp(rlen - take, min=0)[take > 0].max().item()) if bool((take > 0).any()) else 0
    if grow:
        pad = torch.full((ids.shape[0], grow), int(pad_token_id), dtype=ids.dtype, device=ids.device)
        zero = torch.zeros((ids.shape[0], grow), dtype=am.dtype, device=am.device)
        ids = torch.cat([pad, ids[:, :plen], ids[:, plen:]], dim=1)
        am = torch.cat([zero, am[:, :plen], am[:, plen:]], dim=1)
        if pos is not None:
            pos = torch.cat([torch.zeros((pos.shape[0], grow), dtype=pos.dtype, device=pos.device),
                             pos[:, :plen], pos[:, plen:]], dim=1)
        plen += grow
    live_before = am[:, :plen].sum(-1)
    new_ids, new_am, new_pos = splice_span(ids, am, off, take, repl, rlen, int(pad_token_id),
                                           response_length=int(response_length), position_ids=pos)
    want = live_before - take + rlen
    ok = ((new_am[:, :plen].sum(-1) == want) & (take > 0)).detach().cpu().numpy()
    return new_ids, new_am, new_pos, ok
