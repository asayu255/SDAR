"""The measured-only self-distillation term (algorithm.opsd.measure_only).

The term is built from the privileged self's log-probs and its first-order effect
on the reward objective is reported per task; nothing reaches the loss. What has
to hold:

* the reported geometry is the one autograd gives for the two real losses --
  checked exactly on a batch where every position shares a distribution, which is
  the case the dropped per-position weight cannot change (see opsd_geometry);
* a gate that says nothing (constant) against zero-sum advantages reports ~0, and
  a gate that opens on the tokens the reward pushes up reports positive: that is
  the whole decision the metric exists for;
* measure_only leaves use_sdar_loss off, and source=document turns the document
  render on by itself rather than borrowing the rank arm's switch;
* a row whose document did not fit its prompt window is excluded rather than
  scored against the student's own prompt.
"""

import ast
import os

import pytest
import torch
from omegaconf import OmegaConf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _first_order(terms):
    """The pooled ratio the accumulator's sums would produce."""
    return float(terms["g_dot"].sum() / terms["g_grpo_sq"].sum())


def test_the_reported_geometry_is_what_autograd_gives():
    from verl.trainer.ppo.core_algos import policy_loss_gradient_coef
    from verl.trainer.ppo.opsd_geometry import gate_and_gap, geometry_terms

    torch.manual_seed(0)
    B, T, V = 3, 4, 6
    coef, pg_coef, beta = 0.01, 1.0, 5.0
    # The same distribution and the same emitted token at every position, so the
    # per-position |1[v=y] - p|^2 the columns drop is a constant and cancels.
    row = torch.randn(V)
    logits = row.repeat(B, T, 1).clone().requires_grad_(True)
    actions = torch.zeros(B, T, dtype=torch.long)
    mask = torch.ones(B, T)
    advantages = torch.randn(B, T)

    log_prob = torch.log_softmax(logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    teacher = log_prob.detach() + torch.randn(B, T)

    # The two terms as the actor builds them, aggregated by a token SUM -- the
    # convention the per-position columns are summed in.
    gate, gap = gate_and_gap(log_prob.detach(), teacher, beta)
    pg_loss = -(advantages * log_prob * mask).sum()
    sdar_loss = (gate * (teacher - log_prob) * mask).sum()
    g_pg = torch.autograd.grad(pg_loss, logits, retain_graph=True)[0] * pg_coef
    g_sdar = torch.autograd.grad(sdar_loss, logits)[0] * coef
    want = float((g_sdar * g_pg).sum() / (g_pg * g_pg).sum())

    pgc = policy_loss_gradient_coef(
        old_log_prob=log_prob.detach(), log_prob=log_prob.detach(), advantages=advantages,
        cliprange=0.2, cliprange_low=0.2, cliprange_high=0.2, clip_ratio_c=3.0,
    )
    got = _first_order(geometry_terms(gate=gate, gap=gap, pg_grad_coef=pgc,
                                      coef=coef, pg_coef=pg_coef))
    assert got == pytest.approx(want, rel=1e-5), f"{got} vs autograd {want}"


def test_a_gate_that_says_nothing_reports_no_effect():
    from verl.trainer.ppo.opsd_geometry import gate_and_gap, geometry_terms

    B, T = 4, 6
    # Zero gap -> gate exactly 1/2 everywhere: the term pushes up whatever was
    # sampled, which is what the skill-conditioned self did at step 240.
    student = torch.randn(B, T)
    gate, gap = gate_and_gap(student, student.clone(), 5.0)
    assert torch.allclose(gate, torch.full_like(gate, 0.5))
    # GRPO's advantages sum to zero over a group, so a constant weight has
    # nothing to add along them.
    advantages = torch.randn(B, T)
    advantages = advantages - advantages.mean()
    terms = geometry_terms(gate=gate, gap=gap, pg_grad_coef=-advantages, coef=0.01, pg_coef=1.0)
    assert abs(_first_order(terms)) < 1e-6


def test_a_gate_that_opens_where_the_reward_pushes_up_reports_positive():
    from verl.trainer.ppo.opsd_geometry import gate_and_gap, geometry_terms

    B, T = 4, 6
    advantages = torch.randn(B, T)
    advantages = advantages - advantages.mean()
    # The privileged self is more confident exactly on the tokens the reward
    # wants pushed up (positive advantage) -- the alignment the metric is for.
    gate, gap = gate_and_gap(torch.zeros(B, T), advantages.clone(), 5.0)
    agree = _first_order(geometry_terms(gate=gate, gap=gap, pg_grad_coef=-advantages,
                                        coef=0.01, pg_coef=1.0))
    gate_r, gap_r = gate_and_gap(torch.zeros(B, T), -advantages, 5.0)
    disagree = _first_order(geometry_terms(gate=gate_r, gap=gap_r, pg_grad_coef=-advantages,
                                           coef=0.01, pg_coef=1.0))
    assert agree > 0 > disagree, f"{agree} / {disagree}"
    assert agree == pytest.approx(-disagree, rel=1e-6)


def test_the_metrics_carry_the_gate_and_the_same_keys_the_teacher_publishes():
    from verl.trainer.ppo.cross_teacher_kl_weight import gradient_metrics
    from verl.trainer.ppo.opsd_geometry import opsd_metrics

    sums = {None: {"g_opd_sq": 4.0, "g_grpo_sq": 9.0, "g_dot": 3.0, "gate": 5.0, "gap": -2.0, "n": 10.0},
            "alfworld": {"g_opd_sq": 1.0, "g_grpo_sq": 4.0, "g_dot": -1.0, "gate": 2.0, "gap": 1.0, "n": 4.0}}
    out = opsd_metrics(sums)
    assert out["opsd/grpo/first_order"] == pytest.approx(3.0 / 9.0)
    assert out["opsd/alfworld/grpo/first_order"] == pytest.approx(-0.25)
    assert out["opsd/gate_mean"] == pytest.approx(0.5)
    assert out["opsd/alfworld/gap_mean"] == pytest.approx(0.25)
    # The same key shape the external teacher publishes, so the two are one chart.
    teacher_keys = {k.replace("opd/", "", 1) for k in gradient_metrics(sums, prefix="opd")}
    assert {k.replace("opsd/", "", 1) for k in out} >= teacher_keys


def test_rows_without_a_document_are_excluded_from_the_sums():
    from verl.trainer.ppo.opsd_geometry import OPSD_TERMS, gate_and_gap, geometry_terms
    from verl.trainer.ppo.sign_weights import ScopeTermStats

    B, T = 3, 5
    gate, gap = gate_and_gap(torch.zeros(B, T), torch.ones(B, T), 5.0)
    terms = geometry_terms(gate=gate, gap=gap, pg_grad_coef=torch.ones(B, T), coef=0.01, pg_coef=1.0)
    valid = torch.tensor([1.0, 0.0, 1.0])
    stats = ScopeTermStats(names=OPSD_TERMS, n_tasks=0, device=torch.device("cpu"))
    stats.update(terms, response_mask=torch.ones(B, T) * valid.reshape(-1, 1))
    assert stats.sums()[None]["n"] == pytest.approx(2 * T)


def test_measure_only_leaves_the_loss_alone_and_document_turns_the_render_on():
    from verl.trainer.main_opd_grpo import inject_opd_grpo_config

    def _cfg(**opsd):
        return OmegaConf.create({
            "algorithm": {"opd": {"kl_loss_type": "topk_kl"}, "opsd": {"enable": True, **opsd}},
            "actor_rollout_ref": {"actor": {"pg_loss_coef": 1.0}},
        })

    cfg = _cfg(measure_only=True)
    inject_opd_grpo_config(cfg)
    assert cfg.actor_rollout_ref.actor.use_sdar_loss is False
    assert cfg.actor_rollout_ref.actor.opsd_measure_only is True

    applied = _cfg()
    inject_opd_grpo_config(applied)
    assert applied.actor_rollout_ref.actor.use_sdar_loss is True
    assert applied.actor_rollout_ref.actor.opsd_measure_only is False

    # source=document does NOT borrow the rank arm's switch: (a) refuses to run
    # beside that arm, and the render it needs is turned on by the layout gate.
    from agent_system.environments import oci_layout as ol

    doc = _cfg(source="document", measure_only=True)
    inject_opd_grpo_config(doc)
    assert doc.actor_rollout_ref.actor.opsd_measure_only is True
    assert ol.document_render_on(doc) and "webshop" in ol.document_render_tasks(doc)

    with pytest.raises(AssertionError, match="source"):
        from verl.trainer.ppo.opd_grpo_ray_trainer import OPDGRPORayTrainer

        class _Self:
            pass

        me = _Self()
        me.config = OmegaConf.create({"algorithm": {"opd": {"kl_loss_type": "topk_kl"}}})
        OPDGRPORayTrainer._compute_self_teacher_log_probs(
            me, None, OmegaConf.create({"source": "skills"}), {})


def test_the_actor_reads_the_column_under_its_own_name_only():
    src = open(os.path.join(REPO, "verl/workers/actor/dp_actor.py")).read()
    tree = ast.parse(src)
    names = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "opsd_teacher_log_probs" in names and "opsd_valid" in names
    # Nothing may add the measured term to the loss: the only assignments to
    # policy_loss in the file are the ones that existed before it.
    assert "policy_loss = policy_loss + opsd" not in src


def _doc_self(batch_tensors, spliced, monkeypatch):
    """``_self_teacher_on_document`` over a stubbed document splice."""
    import numpy as np

    from verl.protocol import DataProto
    from verl.trainer.ppo import oci_rank
    from verl.trainer.ppo.opd_grpo_ray_trainer import OPDGRPORayTrainer

    batch = DataProto.from_dict(tensors=batch_tensors)
    n, r = batch_tensors["responses"].shape

    def _with_document(b, pad_token_id, min_grow=0):
        # The document adds LIVE prompt tokens (that is what the metric counts),
        # so the widened window comes with an attention mask of ones.
        grown = dict(b.batch.items())
        live = torch.ones((n, 5), dtype=grown["attention_mask"].dtype)
        grown["attention_mask"] = torch.cat([live, grown["attention_mask"]], dim=1)
        grown["input_ids"] = torch.cat([torch.ones_like(live), grown["input_ids"]], dim=1)
        return DataProto.from_dict(tensors=grown), np.array(spliced)

    monkeypatch.setattr(oci_rank, "with_document", _with_document)
    monkeypatch.setattr(oci_rank, "document_rows", lambda b: np.array(spliced))

    class _WG:
        def compute_log_prob(self, data):
            return DataProto.from_dict(tensors={"old_log_probs": torch.full((n, r), -1.5)})

    class _Self:
        pass

    me = _Self()
    me.tokenizer = type("T", (), {"pad_token_id": 0})()
    me.actor_rollout_wg = _WG()
    metrics = {}
    return OPDGRPORayTrainer._self_teacher_on_document(me, batch, metrics), metrics


def test_a_row_whose_document_did_not_fit_is_marked_invalid(monkeypatch):
    n, p, r = 3, 6, 4
    tensors = {
        "input_ids": torch.ones(n, p + r, dtype=torch.long),
        "attention_mask": torch.ones(n, p + r, dtype=torch.long),
        "responses": torch.ones(n, r, dtype=torch.long),
    }
    (lp, valid), metrics = _doc_self(tensors, [True, False, True], monkeypatch)
    assert lp.shape == (n, r)
    assert valid.tolist() == [1.0, 0.0, 1.0]
    assert metrics["opsd/doc_rows_share"] == pytest.approx(2 / 3)
    # Only the rows that took the edit count towards how much it added.
    assert metrics["opsd/prompt_tokens_added/mean"] == pytest.approx(5.0)


def test_the_document_teacher_refuses_a_batch_with_no_document_edit(monkeypatch):
    n, p, r = 2, 6, 4
    tensors = {
        "input_ids": torch.ones(n, p + r, dtype=torch.long),
        "attention_mask": torch.ones(n, p + r, dtype=torch.long),
        "responses": torch.ones(n, r, dtype=torch.long),
    }
    with pytest.raises(AssertionError, match="oci_rank.enable"):
        _doc_self(tensors, [False, False], monkeypatch)


def _layout_cfg(**algorithm):
    from types import SimpleNamespace
    return SimpleNamespace(env=SimpleNamespace(history_length=2),
                           data=SimpleNamespace(max_prompt_length=4096),
                           algorithm=algorithm)


def test_the_document_render_follows_either_reader_not_just_the_rank_arm():
    from agent_system.environments import oci_layout as ol

    rank = _layout_cfg(oci_rank={"enable": True, "tasks": ["alfworld"]})
    assert ol.document_render_on(rank) and ol.document_render_tasks(rank) == ("alfworld",)

    # The self-distillation teacher asks for all three: its term is per task.
    opsd = _layout_cfg(opsd={"enable": True, "source": "document"})
    assert ol.document_render_on(opsd)
    assert ol.document_render_tasks(opsd) == ("alfworld", "webshop", "search")

    # ...and only when it is the document it is conditioned on.
    skill = _layout_cfg(opsd={"enable": True, "source": "skill"})
    assert not ol.document_render_on(skill) and ol.document_render_tasks(skill) == ()
    assert not ol.document_render_on(_layout_cfg())

    both = _layout_cfg(oci_rank={"enable": True, "tasks": ["search"]},
                       opsd={"enable": True, "source": "document"})
    assert ol.document_render_tasks(both) == ("alfworld", "webshop", "search")


def test_the_webshop_manager_renders_the_goal_record_beside_the_plain_prompt():
    import agent_system.environments.env_manager as em
    from agent_system.environments import oci_layout as ol

    def render(cfg):
        m = em.WebshopEnvironmentManager.__new__(em.WebshopEnvironmentManager)
        m.config = cfg
        m.tasks = ["Find me a red dress"]
        m.document_block = lambda i: "[Privileged Solution Path]\nsearch[red dress]\n"
        infos = [{"available_actions": {"has_search_bar": True, "clickables": ["Search"]}}]
        text = em.WebshopEnvironmentManager.build_text_obs(m, ["'Search'"], infos, init=True)
        return text, m._oci_docs

    text, docs = render(_layout_cfg(opsd={"enable": True, "source": "document"}))
    assert docs[0] == "[Privileged Solution Path]\nsearch[red dress]\n" + text[0]
    # the prompt the policy is given is the plain one either way
    plain_text, plain_docs = render(_layout_cfg())
    assert plain_text == text and plain_docs == [""]
    assert ol.OCI_DOC_KEY == "oci_doc"
