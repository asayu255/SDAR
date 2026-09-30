"""progress_value_gae: a per-turn advantage from a value table of the state, by TD errors and GAE.

THE OBJECTIVE. J_H = E[gamma^T R]: succeed within the task's turn cap H. R in {0, 1}
is whether the trajectory won -- episode_rewards > 0, the test progress_rank's
failed_groups uses -- and a failure and a timeout both end at 0; there is no other
reward. The value of the state z_t the agent is in BEFORE action t is

    V(z_t) = E[gamma^(T-t) R | z_t],        V(z_T) := R        (the boundary)

and every turn is credited with how far its action moved that value:

    delta_t = gamma V(z_{t+1}) - V(z_t)
    A_t     = sum_l (gamma lam)^l delta_{t+l}                       (GAE)

Two identities hold for ANY table, and tests/trainer/test_progress_value.py holds
the code to both: sum_t gamma^t delta_t = gamma^T R - V(z_0) (the deltas
telescope onto the boundary), and at lam = 1, A_t = gamma^(T-t) R - V(z_t) (the
Monte Carlo return minus the value: the table is then only a baseline, and a
wrong table costs variance, not bias).

WHY A TABLE OF COUNTED STATE AND NOT A CRITIC NETWORK. The offline check
(~/scratch_keep/value_calib, val replays of the RL arms) found that a small table
over (ALFWorld type, progress k, turns since k last rose, turns remaining),
shrunk towards coarser tables, is calibrated and ranks states well (ALFWorld AUC
0.86, WebShop 0.81, 0.84 with the score a purchase would get now), and holds
from steps 100-150 to 200-300. The current-state features (ALFWorld hold / inside
/ at, WebShop ongoal / optnow) added nothing there and are off by default. The first-hit
value of a progress level alone overestimates stagnant and near-timeout states
by +0.5-0.8. The buckets below are that check's (evaluate_features.py rb/sb).

THE TABLE (ProgressValueTable), one per task:

    cell    (task, the task's features)            e.g. (alfworld, type, k, stag, rem)
    parent  (task, type, rem)
    root    (task)

    V_root   = s_r / n_r
    V_parent = (s_p + n0 V_root)   / (n_p + n0)
    V_cell   = (s_c + n0 V_parent) / (n_c + n0)        clipped to [0, 1]

s and n are the sum of targets y and the number of states that reached the node.
This is the Beta's only role here: a sparse cell's value is its parent's, moved
by as much data as the cell has, at strength n0.

    targets  y_{i,t} = gamma^(T_i - t) R_i for every real row of every trajectory
    counts   at each update, every (s, n) is multiplied by ``retention`` BEFORE
             the new batch is added (discounted counts: the table follows the
             policy, at an effective memory of 1 / (1 - retention) steps); a node
             whose n falls below 1e-6 is dropped

FROZEN WITHIN A STEP. The batch's advantages are computed from the table as the
PREVIOUS steps left it, and the batch is added only after (``stage`` inside the
advantage computation, ``commit`` once the step's update is done). Were this
step's outcomes in the table, a cell visited only by one trajectory would
predict that trajectory's own R and cancel its advantage: the baseline would
depend on the action it is supposed to judge.

WHEN THE TABLE IS EMPTY for a task (step 1, a fresh start), V_t is the mean R of
the OTHER rollouts of the trajectory's group, the same at every turn -- a
baseline that does not depend on the trajectory's own outcome, which the group
mean with itself would. A group of one gets 0. progress_value/<task>/
fallback_share says which steps ran on it.

WEBSHOP GOALS THE ENVIRONMENT CANNOT PAY (goal_capped == 1): their true value is
0 under any policy, so V_t = 0, and they are kept out of the table update --
counted in, they would drag down every winnable state they share a cell with.
They are kept out of the value-side metrics too (see _metrics); capped_share
says how many there were. The flag scores ONE purchase -- the goal's own
product with exactly the goal's options (webshop envs.goal_capped) -- not every
purchase a policy could make, so it is not a proof that no policy can win;
capped_won counts the capped trajectories that DID win -- 0 if the flag is
what it claims (the trainer prints a warning when it is not).

OPTIONS (algorithm.progress_value):

    prefix_discount  A_t *= gamma^t: the discounted objective's own policy
                     gradient weight. Applied to the GAE term only: the episode
                     term below already carries gamma^T in U_i, which is
                     gamma^t gamma^(T-t) R, the same weight. REQUIRED when
                     gamma < 1 and eta > 0 (ProgressValueConfig refuses the
                     rest): the gradient of J_H from the start state weights
                     turn t by gamma^t, and a GAE term without it credits every
                     turn as if the episode began there -- a GiGPO-style update,
                     which is another method, not this objective's gradient.
    eta              A = (1 - eta) (U_i - b_{g,-i}) + eta A_t with U_i = gamma^(T_i) R_i
                     and b_{g,-i} the mean of U over the OTHER trajectories of the
                     group, each trajectory counted once (NOT once per turn: a long
                     failure must not weigh eight times a short win in the baseline
                     the win is judged against). eta = 1, the default, is the single GAE.
    adv_scale        A_rl = adv_scale * A. 2.0 = 1 / sigma of a Bernoulli at
                     p = 1/2, so a p = 1/2 group gets the magnitude GRPO's z gives it.
                     A FIXED unit conversion -- the same for every batch, group and
                     task -- not a normalisation: dividing by a group's std would
                     bring back the tied-group blow-up and make the unit depend on
                     the batch.

THE FORMAT TERM, KEPT OUT OF delta. The -0.1 invalid-action penalty is not part
of J_H (R is 0/1), and inside the TD error it would be an uncentred push-down of
every turn of every trajectory before the format is acquired. What the control
gets from it is something else: in a group whose rollouts all have the same R,
the outcome cancels in GRPO's z-score and what is left is exactly the z-score of
the invalid indicator over the group's turn rows -- the regularizer that stops
verbosity. That is reproduced as a separate term:

    x_row  = 0 if is_action_valid else -1, over the uid group's real turn rows
    A_fmt  = (x - mean x) / std x       float64, ddof=1 (torch.std, as GRPO)
    A_fmt  = 0 when std < 1e-6          (a uniform group: all valid or all invalid)

In float64 and with the std cut, a uniform group gets exactly 0 -- the float32
phantom advantage GRPO gives such groups (a uniform ~0.48 from round-off) is
gone. ``format_scope`` tied (default) applies it only to groups whose
trajectories all have the same R, where the control applies it; all applies it
to every group. GRPO divides by std + 1e-6 on the 0.1-scaled score, i.e. by
std x + 1e-5; the difference is below 1e-4 of the term.

    advantage = A_rl + format_coef * A_fmt       one scalar per row

WHAT A SAVED TABLE IS A TABLE OF (ProgressValueConfig.fingerprint, checked by
load_state_dict). A table resumed under another reading of its cells would be
read as other states, and one whose targets counted another success is of
another objective, so the fingerprint names everything a cell and a target mean:

    features, rem_buckets, stag_buckets    the cell a row is keyed on
    n0, retention                          how the counts were shrunk and discounted
    gamma                                  the targets' discount
    horizons       {task: H}               the turn caps the trajectories ran under:
                                           the targets are success WITHIN H, and rem
                                           is a fraction of H (a trajectory whose
                                           pv_cap is not its task's H is refused)
    k_definitions  {task: name}            which count k (and so stag) is: ALFWorld's
                                           alfworld_k, Search's search_k, WebShop's
                                           session count (WEBSHOP_K_DEFINITION)
    feature_schema_version                 FEATURE_SCHEMA_VERSION: how a row becomes a cell,
                                           and what the environment writes into the columns
    reward_schema_version                  REWARD_SCHEMA_VERSION: what counts as a win

Those names are what the fingerprint compares, so each is listed beside the code it names, in
this module and in the environment, and that code points back to it (the constants below).

API. Nothing here imports the trainer; the arrays are numpy (or anything
np.asarray accepts).

    cfg   = ProgressValueConfig.from_config(config.algorithm.progress_value,
                                            gamma=config.algorithm.gamma, lam=config.algorithm.lam,
                                            horizons={task: H}, k_definitions={task: name})
            # (the trainer's one builder: opd_ray_trainer.progress_value_config)
    table = ProgressValueTable(cfg)                  # once; table.load_state_dict(saved) on resume
    res   = compute_progress_value_advantage(columns, table)   # reads the table, changes nothing
    table.stage(res.records)                         # the update this batch owes
    ...  actor update  ...
    table.commit()                                   # decay, then add the staged batch
                                                     # (table.discard() for a batch that takes no
                                                     # optimizer step: a grad_probe batch)
    json.dump(table.state_dict(), f)                 # beside the checkpoint

``columns`` maps the batch's column names to 1-D per-row arrays, rows in ANY
order (after _balance_batch, adjust_batch's padding copies included):

    traj_uid         trajectory id; its real rows must carry pv_t = 0..T-1 exactly
    uid              group id (the rollouts of one prompt)
    pv_t             0-based turn index of the row in its trajectory
    pv_cap           the task's turn cap H (rem = pv_cap - pv_t); the configured horizon
                     of the task, when the configuration names one
    task_name        the row's task, matched to the configured tasks by substring
    episode_rewards  the trajectory's environment reward; R = 1 iff > 0
    is_action_valid  bool / 0-1: the format term's indicator
    is_padding_row   optional (absent = no padding): True for adjust_batch's copies
    pv_k_before      when a task's features include k   (k_hist before the action)
    pv_stag_before   ...                        stag     (turns since k last rose)
    pv_hold_b, pv_inside_b, pv_at_b, pv_ongoal_b, pv_optnow_b, pv_buynow_b, pv_evid_b
                     ... hold, inside, at, ongoal, optnow, buynow, evid (the state before the
                     action; buynow is keyed in quarters, FEATURE_BINS)
    gamefile         required when an ALFWorld row is present: its task type
    goal_capped      required when a WebShop row is present: 1 = the goal cannot pay
    pv_k_after       optional: only for the delta-on-progress-turns metrics

(``required_columns(cfg, tasks)`` lists them for a set of tasks.) It returns a
ProgressValueResult: per-row float64 arrays aligned with the input -- a padding
copy carries its original's values -- ``advantage`` (the scalar to broadcast over
the row's response tokens), ``a_rl``, ``a_fmt``, ``value`` (V(z_t)), ``returns``
(a_rl / adv_scale + value), ``delta``, ``fallback`` (bool); ``records`` (the table
update, for ``stage``) and ``metrics`` (keys ``progress_value/<task>/<name>``).
It raises ValueError on a trajectory whose real rows do not carry pv_t = 0..T-1,
a padding copy with no original, a trajectory longer than its cap, a trajectory
whose cap is not its task's configured horizon, and a non-finite pv_t, pv_cap,
episode_rewards or is_action_valid on a real row.
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# ---- column names (the rollout loop writes the pv_* ones; see the API above) ---- #
TRAJ_UID = "traj_uid"
UID = "uid"
PV_T = "pv_t"
PV_CAP = "pv_cap"
PV_K_BEFORE = "pv_k_before"
PV_K_AFTER = "pv_k_after"
PV_STAG_BEFORE = "pv_stag_before"
TASK_NAME = "task_name"
GAMEFILE = "gamefile"
EPISODE_REWARDS = "episode_rewards"
GOAL_CAPPED = "goal_capped"
IS_ACTION_VALID = "is_action_valid"
IS_PADDING_ROW = "is_padding_row"

ALFWORLD = "alfworld"
WEBSHOP = "webshop"

# Feature name -> the column it reads. type and rem are derived: type from the
# gamefile path (ALFWorld) or the task name (every other task, which has one
# type); rem from pv_cap - pv_t, bucketed by the fraction of H left.
FEATURE_COLUMNS: Dict[str, Optional[str]] = {
    "type": None,
    "k": PV_K_BEFORE,
    "stag": PV_STAG_BEFORE,
    "rem": None,
    "hold": "pv_hold_b",
    "inside": "pv_inside_b",
    "at": "pv_at_b",
    "ongoal": "pv_ongoal_b",
    "optnow": "pv_optnow_b",
    "buynow": "pv_buynow_b",
    "evid": "pv_evid_b",
}
# A feature whose recorded value is continuous, and the bin it is keyed on. buynow is WebShop's
# purchase score of the open product in [0, 1], keyed in quarters (0, .25, .5, .75, 1 -> 0..4),
# as in the offline check (ws_replay_features / evaluate_features "bn").
FEATURE_BINS = {
    "buynow": lambda v: int(round(4.0 * v)),
}
DEFAULT_FEATURES: Dict[str, Tuple[str, ...]] = {
    "alfworld": ("type", "k", "stag", "rem"),
    "webshop": ("k", "stag", "rem", "buynow"),
    "search": ("k", "rem"),
}
# rem: bucket i is the first i with rem > f_i * H (6 bins: > 0.8H, > 0.6H, ... , the rest).
DEFAULT_REM_BUCKETS: Tuple[float, ...] = (0.8, 0.6, 0.4, 0.2, 0.1)
# stag: bucket = number of edges <= stag (0 | 1-2 | 3-5 | 6-9 | 10-19 | 20+).
DEFAULT_STAG_BUCKETS: Tuple[float, ...] = (1.0, 3.0, 6.0, 10.0, 20.0)
FORMAT_SCOPES = ("tied", "all")
# A node below this count holds nothing a prediction could tell from zero.
DROP_BELOW = 1e-6
# The format term's cut: an invalid indicator's std is 0 or >= 1/sqrt(m) for m rows.
FORMAT_STD_MIN = 1e-6
STATE_VERSION = 1
# What a saved table's cells and targets MEAN beyond the configuration; all three are in the
# fingerprint. The fingerprint holds these names, not the code: a change to what the code computes
# moves nothing in it unless the name moves too, and a resumed table is then read under the new
# meaning without an error. So each one lists the code it names, the environment's as well as this
# module's, and every piece of that code points back here.
#
# FEATURE_SCHEMA_VERSION: how a row becomes a cell and a parent. Here: the rem buckets as fractions of
# H left and the stag buckets as lower edges (rem_bucket / stag_bucket), buynow's quarters
# (FEATURE_BINS), the ALFWorld type parse (alfworld_type), a missing value as a cell of its own
# (_key_value), the parent (task, type, rem) -- test_the_feature_schema_is_pinned. In the environment
# (agent_system/environments), what fills the columns those read: pv_stag_before's rule
# (progress.PvTracker.step: 0 once k rose, else one more); what the ALFWorld and Search counts behind
# pv_k count (progress.AlfworldMilestones.k / k_arrive, search_progress / search_progress_answered --
# k_definitions names WHICH count, this versions WHAT it counts); pv_buynow (env_package/webshop/envs.py
# WebshopWorker._buy_now_score, through env_manager._ws_buy_now); and the current-state features a
# configuration may key on (AlfworldMilestones.current_state's hold / inside / at,
# WebshopProgress.on_goal / opts_now, the Search manager's evidence flag) --
# tests/oci/test_progress_value_records.py, section 10. BUMP IT with any change on either side: a
# table keyed the old way would be read, without an error, as other states.
FEATURE_SCHEMA_VERSION = 1
# REWARD_SCHEMA_VERSION: what the targets count as a win. R = 1 iff episode_rewards > 0, which is
# ALFWorld's 10 on won (0 otherwise), WebShop's 10 iff the environment's task_score is 1.0, and Search's
# exact match (1, else 0). A change to any of them -- a partial WebShop score, a substring match for
# Search -- is another objective, and a table of it another table. Change the string with it. The
# code: env_package/alfworld/envs.py compute_reward, env_package/webshop/envs.py WebshopWorker.step
# (its reward redefinition), and the Search environment's compute_score (skyrl_gym search/utils.py,
# exact match by em_check) as env_package/search/envs.py takes it --
# tests/oci/test_progress_value_records.py, section 10.
REWARD_SCHEMA_VERSION = ("win_v1: R = episode_rewards > 0; alfworld 10/0 on won; "
                         "webshop 10 iff task_score == 1.0; search exact match 1/0")
# The WebShop k of the pv_* columns: ALWAYS the session count (WebshopProgress.k_session: goal product
# found + opened + best required options held + bought, read off the environment's session),
# whatever algorithm.progress_rank.webshop_k puts into progress_k. Versioned like the schemas: a
# change to what the session count counts is a change of k. The code: progress.WebshopProgress
# (k_session and _note_session, and found, which it shares with the legacy count) and the session
# WebshopWorker._session_state ships -- tests/oci/test_progress_value_records.py, sections 2 and 10.
WEBSHOP_K_DEFINITION = "session_v1"

ALFWORLD_TYPES = ("pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
                  "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place")


def alfworld_type(gamefile) -> str:
    """The task type in an ALFWorld gamefile path, "unknown" if none.

    .../json_2.1.1/train/pick_two_obj_and_place-Book-None-Desk-310/trial_.../game.tw-pddl: the type
    is the part before the first "-" of the game's directory. Read from the path, not traj_data.json,
    so the table needs no file access.
    """
    for part in reversed(str(gamefile or "").replace("\\", "/").split("/")):
        head = part.split("-", 1)[0]
        if head in ALFWORLD_TYPES:
            return head
    return "unknown"


def _finite(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _key_value(x):
    """A feature value as a table key: an int when integral (1.0 and 1 are one cell), None when missing."""
    v = _finite(x)
    if v is None:
        return None
    return int(v) if v.is_integer() else v


def gae(values: Sequence[float], reward: float, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(delta, A)`` of one trajectory: V(z_0..z_{T-1}) and the boundary V(z_T) = reward.

    delta_t = gamma V_{t+1} - V_t; A_t = delta_t + gamma lam A_{t+1}, A_{T-1} = delta_{T-1}.
    """
    v = np.append(np.asarray(values, dtype=np.float64), float(reward))
    delta = gamma * v[1:] - v[:-1]
    adv = np.empty_like(delta)
    acc = 0.0
    for t in range(len(delta) - 1, -1, -1):
        acc = delta[t] + gamma * lam * acc
        adv[t] = acc
    return delta, adv


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class ProgressValueConfig:
    """algorithm.progress_value, plus algorithm.gamma and algorithm.lam (see the module docstring)."""

    features: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_FEATURES))
    rem_buckets: Tuple[float, ...] = DEFAULT_REM_BUCKETS
    stag_buckets: Tuple[float, ...] = DEFAULT_STAG_BUCKETS
    n0: float = 8.0
    retention: float = 0.9
    gamma: float = 1.0
    lam: float = 1.0
    prefix_discount: bool = False
    eta: float = 1.0
    adv_scale: float = 2.0
    format_scope: str = "tied"
    format_coef: float = 1.0
    # {task: H}, the turn caps the trajectories ran under, and {task: name}, which count k is: not
    # keys of the block but read off the run's config by the trainer's builder
    # (opd_ray_trainer.progress_value_config). Empty = unknown (no check, and so fingerprinted).
    horizons: Dict[str, int] = field(default_factory=dict)
    k_definitions: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.features = {str(t).lower(): tuple(str(f) for f in fs) for t, fs in dict(self.features).items()}
        self.rem_buckets = tuple(float(x) for x in self.rem_buckets)
        self.stag_buckets = tuple(float(x) for x in self.stag_buckets)
        for name in ("n0", "retention", "gamma", "lam", "eta", "adv_scale", "format_coef"):
            setattr(self, name, float(getattr(self, name)))
        self.prefix_discount = bool(self.prefix_discount)
        self.format_scope = str(self.format_scope)
        horizons = {}
        for task, h in dict(self.horizons or {}).items():
            v = _finite(h)
            assert v is not None and v.is_integer() and v >= 1.0, (
                f"progress_value horizons.{task}={h!r}: a turn cap is a positive integer")
            horizons[str(task).lower()] = int(v)
        self.horizons = horizons
        self.k_definitions = {str(t).lower(): str(k) for t, k in dict(self.k_definitions or {}).items()}
        assert all(self.k_definitions.values()), f"progress_value k_definitions={self.k_definitions}: empty name"
        for task, fs in self.features.items():
            assert fs, f"progress_value.features.{task} is empty; every task needs at least one feature"
            bad = [f for f in fs if f not in FEATURE_COLUMNS]
            assert not bad, f"progress_value.features.{task}: unknown {bad}; allowed {sorted(FEATURE_COLUMNS)}"
            assert len(set(fs)) == len(fs), f"progress_value.features.{task} repeats a feature: {fs}"
        rb = self.rem_buckets
        assert all(0.0 < f < 1.0 for f in rb) and all(a > b for a, b in zip(rb, rb[1:])), (
            f"progress_value.rem_buckets={rb}: fractions of H in (0, 1), strictly decreasing")
        sb = self.stag_buckets
        assert all(e > 0.0 for e in sb) and all(a < b for a, b in zip(sb, sb[1:])), (
            f"progress_value.stag_buckets={sb}: positive lower edges, strictly increasing")
        assert self.n0 > 0.0, f"progress_value.n0={self.n0}: must be > 0"
        assert 0.0 < self.retention <= 1.0, f"progress_value.retention={self.retention}: must be in (0, 1]"
        assert 0.0 < self.gamma <= 1.0, f"algorithm.gamma={self.gamma}: must be in (0, 1]"
        assert 0.0 <= self.lam <= 1.0, f"algorithm.lam={self.lam}: must be in [0, 1]"
        assert 0.0 <= self.eta <= 1.0, f"progress_value.eta={self.eta}: must be in [0, 1]"
        # The GAE term is the gradient of J_H = E[gamma^T R] FROM THE START STATE only with turn t
        # weighted by gamma^t: the objective discounts the whole episode from turn 0, and the turn-t
        # term of its gradient carries gamma^t. Without the prefix every turn is credited as if the
        # episode began at it (GiGPO's step returns do exactly that) -- a legitimate update, but a
        # different method, and not the one this estimator claims to be. The episode term (eta < 1)
        # already carries gamma^T; at gamma = 1 the prefix is 1.
        assert not (self.gamma < 1.0 and self.eta > 0.0 and not self.prefix_discount), (
            f"algorithm.gamma={self.gamma} < 1 with progress_value.eta={self.eta} > 0 needs "
            "progress_value.prefix_discount=True: the GAE term is the start-state discounted objective's "
            "gradient only with the gamma^t weight on turn t. A no-prefix (GiGPO-style) update, every turn "
            "credited as if the episode began there, would be a different method, not this objective's.")
        assert self.adv_scale > 0.0, f"progress_value.adv_scale={self.adv_scale}: must be > 0"
        assert self.format_scope in FORMAT_SCOPES, (
            f"progress_value.format_scope={self.format_scope!r}; expected one of {FORMAT_SCOPES}")
        assert self.format_coef >= 0.0, f"progress_value.format_coef={self.format_coef}: must be >= 0"

    @classmethod
    def from_config(cls, cfg, *, gamma: float, lam: float, horizons: Optional[Mapping[str, Any]] = None,
                    k_definitions: Optional[Mapping[str, str]] = None) -> "ProgressValueConfig":
        """From the algorithm.progress_value node (a dict or DictConfig); a missing or null key keeps
        its default, and ``features`` overrides the default feature list task by task. ``horizons``
        and ``k_definitions`` come from outside the block (the turn caps, progress_rank's counter
        keys): the trainer reads them in one place, opd_ray_trainer.progress_value_config."""
        cfg = cfg or {}

        def get(key, default):
            v = cfg.get(key, None)
            return default if v is None else v

        features = dict(DEFAULT_FEATURES)
        for task, fs in dict(get("features", {})).items():
            features[str(task).lower()] = tuple(str(f) for f in fs)
        return cls(features=features,
                   rem_buckets=tuple(get("rem_buckets", DEFAULT_REM_BUCKETS)),
                   stag_buckets=tuple(get("stag_buckets", DEFAULT_STAG_BUCKETS)),
                   n0=get("n0", 8.0), retention=get("retention", 0.9), gamma=gamma, lam=lam,
                   prefix_discount=get("prefix_discount", False), eta=get("eta", 1.0),
                   adv_scale=get("adv_scale", 2.0), format_scope=get("format_scope", "tied"),
                   format_coef=get("format_coef", 1.0),
                   horizons=dict(horizons or {}), k_definitions=dict(k_definitions or {}))

    def fingerprint(self) -> dict:
        """What the table's contents depend on: a saved table under another of these is another table.

        The cells' keys (features, buckets) and how they were counted (n0, retention), the targets'
        objective (gamma, the horizons H, the reward schema) and what the keyed values mean (the k
        definitions, the feature schema). Not lam, eta, adv_scale, prefix_discount or the format
        term: they act on the advantage computed FROM the table, never on what is in it.
        """
        return {"features": {t: list(fs) for t, fs in sorted(self.features.items())},
                "rem_buckets": list(self.rem_buckets), "stag_buckets": list(self.stag_buckets),
                "n0": self.n0, "retention": self.retention, "gamma": self.gamma,
                "horizons": dict(sorted(self.horizons.items())),
                "k_definitions": dict(sorted(self.k_definitions.items())),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "reward_schema_version": REWARD_SCHEMA_VERSION}

    def task_of(self, name) -> str:
        """The configured task a raw task name belongs to (exact, else by substring, as normalize_task_name)."""
        s = str(name).lower()
        if s in self.features:
            return s
        hits = [t for t in sorted(self.features) if t in s]
        if len(hits) != 1:
            raise ValueError(f"progress_value: task_name {name!r} matches {hits or 'none'} of the configured "
                             f"tasks {sorted(self.features)}")
        return hits[0]

    def rem_bucket(self, rem: float, cap: float) -> int:
        # The offline check's rb(), comparison for comparison, so its cells are these cells.
        for i, f in enumerate(self.rem_buckets):
            if rem > f * cap:
                return i
        return len(self.rem_buckets)

    def stag_bucket(self, stag) -> Optional[int]:
        s = _finite(stag)
        if s is None:
            return None
        return sum(1 for e in self.stag_buckets if s >= e)


def required_columns(cfg: ProgressValueConfig, tasks: Iterable[str]) -> List[str]:
    """The columns compute_progress_value_advantage reads for rows of these (configured) tasks."""
    cols = [TRAJ_UID, UID, PV_T, PV_CAP, TASK_NAME, EPISODE_REWARDS, IS_ACTION_VALID]
    for task in sorted(set(tasks)):
        for f in cfg.features[task]:
            c = FEATURE_COLUMNS[f]
            if c is not None and c not in cols:
                cols.append(c)
        if task == ALFWORLD and GAMEFILE not in cols:
            cols.append(GAMEFILE)
        if task == WEBSHOP and GOAL_CAPPED not in cols:
            cols.append(GOAL_CAPPED)
    return cols


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

class ProgressValueTable:
    """Per-task tabular V with discounted counts and cell -> (task, type, rem) -> task shrinkage.

    Keys are tuples whose first element is the task: cells (task, *features), parents
    (task, type, rem_bucket). Each node holds [s, n]. ``records`` are (cell, parent, y) triples.
    """

    def __init__(self, cfg: ProgressValueConfig):
        self.cfg = cfg
        self.cells: Dict[tuple, List[float]] = {}
        self.parents: Dict[tuple, List[float]] = {}
        self.roots: Dict[str, List[float]] = {}
        self.updates = 0
        self.pending: Optional[List[tuple]] = None

    # --- prediction -------------------------------------------------------- #

    def has_mass(self, task: str) -> bool:
        sn = self.roots.get(task)
        return sn is not None and sn[1] > 0.0

    def value(self, cell: tuple, parent: tuple) -> float:
        """V_cell shrunk through V_parent to V_root; the task must have mass (has_mass)."""
        n0 = self.cfg.n0
        s_r, n_r = self.roots[cell[0]]
        v = s_r / n_r
        s_p, n_p = self.parents.get(parent, (0.0, 0.0))
        v = (s_p + n0 * v) / (n_p + n0)
        s_c, n_c = self.cells.get(cell, (0.0, 0.0))
        v = (s_c + n0 * v) / (n_c + n0)
        return min(max(v, 0.0), 1.0)

    def task_stats(self, task: str) -> Tuple[int, float]:
        """(cells, mass): the task's number of cells and its root count n."""
        cells = sum(1 for k in self.cells if k[0] == task)
        return cells, float(self.roots.get(task, (0.0, 0.0))[1])

    # --- update ------------------------------------------------------------ #

    def update(self, records: Iterable[tuple]) -> None:
        """Decay every node by ``retention``, drop the ones below 1e-6, then add the records."""
        r = self.cfg.retention
        for store in (self.cells, self.parents, self.roots):
            for key in list(store):
                sn = store[key]
                sn[0] *= r
                sn[1] *= r
                if sn[1] < DROP_BELOW:
                    del store[key]
        for cell, parent, y in records:
            y = float(y)
            for store, key in ((self.cells, tuple(cell)), (self.parents, tuple(parent)), (self.roots, cell[0])):
                sn = store.setdefault(key, [0.0, 0.0])
                sn[0] += y
                sn[1] += 1.0
        self.updates += 1

    def stage(self, records: Iterable[tuple]) -> None:
        """Hold a batch's update until ``commit``. A second stage before the commit replaces the first:
        the table is still the one both batches were scored against."""
        self.pending = list(records)

    def commit(self) -> bool:
        """Apply the staged update (decay, then add); False when nothing was staged (the table is kept
        as it is -- no decay without a batch)."""
        if self.pending is None:
            return False
        records, self.pending = self.pending, None
        self.update(records)
        return True

    def discard(self) -> bool:
        """Drop the staged update without applying it (no decay either); False when nothing was
        staged. For a batch that takes no optimizer step -- a grad_probe batch, which must leave
        the table as the checkpoint had it, so every probe batch is scored against the same one."""
        staged, self.pending = self.pending is not None, None
        return staged

    # --- persistence ------------------------------------------------------- #

    def state_dict(self) -> dict:
        """JSON-serialisable. A staged, uncommitted batch is not part of the table and is not saved."""
        def dump(store):
            return [[list(k), float(sn[0]), float(sn[1])]
                    for k, sn in sorted(store.items(), key=lambda kv: json.dumps(list(kv[0])))]

        return {"version": STATE_VERSION, "fingerprint": self.cfg.fingerprint(), "updates": int(self.updates),
                "cells": dump(self.cells), "parents": dump(self.parents),
                "roots": [[t, float(sn[0]), float(sn[1])] for t, sn in sorted(self.roots.items())]}

    def load_state_dict(self, state: Optional[dict]) -> None:
        """Restore a saved table; None or {} leaves it empty. Raises ValueError when the saved table was
        built under another configuration (features, buckets, n0, retention, gamma, the horizons, the k
        definitions) or another feature / reward schema: its cells would be read as other states, or
        its targets are of another objective. A table saved before the fingerprint named the horizons,
        the k definitions and the schemas is refused the same way: nothing says what it is."""
        if not state:
            return
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"progress_value state version {state.get('version')!r}; expected {STATE_VERSION}")
        saved = _canonical(state.get("fingerprint"))
        mine = _canonical(self.cfg.fingerprint())
        if saved != mine:
            diff = sorted(k for k in set(saved or {}) | set(mine) if (saved or {}).get(k) != mine.get(k))
            raise ValueError(f"progress_value state was saved under another configuration (differs in {diff}): "
                             f"saved {saved}, configured {mine}")
        self.cells = {tuple(k): [float(s), float(n)] for k, s, n in state.get("cells", [])}
        self.parents = {tuple(k): [float(s), float(n)] for k, s, n in state.get("parents", [])}
        self.roots = {str(t): [float(s), float(n)] for t, s, n in state.get("roots", [])}
        self.updates = int(state.get("updates", 0))
        self.pending = None


def _canonical(x):
    return None if x is None else json.loads(json.dumps(x, sort_keys=True))


# --------------------------------------------------------------------------- #
# The advantage
# --------------------------------------------------------------------------- #

@dataclass
class ProgressValueResult:
    advantage: np.ndarray          # a_rl + format_coef * a_fmt, per row
    a_rl: np.ndarray               # adv_scale * (the GAE term, mixed with the episode term by eta)
    a_fmt: np.ndarray              # the format term (0 outside its scope)
    value: np.ndarray              # V(z_t)
    returns: np.ndarray            # a_rl / adv_scale + value
    delta: np.ndarray              # gamma V(z_{t+1}) - V(z_t)
    fallback: np.ndarray           # bool: V was the leave-one-out group mean (empty table)
    records: List[tuple]           # (cell, parent, y) for ProgressValueTable.stage
    metrics: Dict[str, float]


@dataclass
class _Traj:
    tuid: str
    uid: str
    task: str
    typ: str
    rows: List[int]                # real rows, turn order
    cap: float
    R: float
    capped: bool

    @property
    def T(self) -> int:
        return len(self.rows)


def compute_progress_value_advantage(columns: Mapping[str, Any], table: ProgressValueTable) -> ProgressValueResult:
    """Per-row advantages of one batch from the frozen table (see the module docstring for the columns).

    Every sum runs in a canonical order -- trajectories by traj_uid, rows by pv_t -- so the result of
    every row is bit-identical whatever order the rows arrive in.
    """
    cfg = table.cfg
    if TRAJ_UID not in columns:
        raise KeyError(f"progress_value: missing column {TRAJ_UID!r}")
    n = len(np.asarray(columns[TRAJ_UID]).reshape(-1))

    def col(name):
        a = np.asarray(columns[name]).reshape(-1)
        if len(a) != n:
            raise ValueError(f"progress_value: column {name!r} has {len(a)} rows, {TRAJ_UID!r} has {n}")
        return a

    missing = [c for c in (UID, PV_T, PV_CAP, TASK_NAME, EPISODE_REWARDS, IS_ACTION_VALID) if c not in columns]
    if missing:
        raise KeyError(f"progress_value: missing columns {missing}")
    tuids = [str(x) for x in col(TRAJ_UID)]
    uids = [str(x) for x in col(UID)]
    task_cache: Dict[Any, str] = {}
    tasks = []
    for x in col(TASK_NAME):
        key = str(x)
        if key not in task_cache:
            task_cache[key] = cfg.task_of(x)
        tasks.append(task_cache[key])
    pv_t = col(PV_T)
    pv_cap = col(PV_CAP)
    rewards = col(EPISODE_REWARDS)
    valid_col = col(IS_ACTION_VALID)
    pad = col(IS_PADDING_ROW).astype(bool) if IS_PADDING_ROW in columns else np.zeros(n, dtype=bool)

    present = sorted({tasks[i] for i in range(n) if not pad[i]})
    missing = [c for c in required_columns(cfg, present) if c not in columns]
    if missing:
        raise KeyError(f"progress_value: missing columns {missing} for tasks {present}")
    feat_cols = {c: col(c) for t in present for f in cfg.features[t]
                 for c in [FEATURE_COLUMNS[f]] if c is not None}
    gamefiles = col(GAMEFILE) if GAMEFILE in columns else None
    capped_col = col(GOAL_CAPPED) if GOAL_CAPPED in columns else None

    # ---- trajectories from the real rows, put back in turn order ----
    def turn_of(i) -> int:
        t = _finite(pv_t[i])
        if t is None or not t.is_integer() or t < 0:
            raise ValueError(f"progress_value: pv_t={pv_t[i]!r} on row {i} (trajectory {tuids[i]}) "
                             f"is not a turn index")
        return int(t)

    by_traj: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    # The format term's indicator, x = 0 valid / -1 invalid, checked on every real row whatever the scope.
    invalid_x = np.zeros(n)
    for i in range(n):
        if not pad[i]:
            by_traj[tuids[i]].append((turn_of(i), i))
            ok = _finite(valid_col[i])
            if ok is None:
                raise ValueError(f"progress_value: is_action_valid={valid_col[i]!r} on row {i} "
                                 f"(trajectory {tuids[i]})")
            invalid_x[i] = 0.0 if ok != 0.0 else -1.0
    trajs: List[_Traj] = []
    for tu in sorted(by_traj):
        pairs = sorted(by_traj[tu])
        ts = [t for t, _ in pairs]
        if ts != list(range(len(ts))):
            raise ValueError(f"progress_value: trajectory {tu} has turns {ts}, not 0..{len(ts) - 1}: a turn is "
                             f"missing or repeated, so its rows cannot be put back in order")
        rows = [i for _, i in pairs]
        for name, vals in (("uid", uids), ("task_name", tasks)):
            if len({vals[i] for i in rows}) != 1:
                raise ValueError(f"progress_value: trajectory {tu} spans {name}s {sorted({vals[i] for i in rows})}")
        caps = {_finite(pv_cap[i]) for i in rows}
        if len(caps) != 1 or None in caps or min(caps) <= 0:
            raise ValueError(f"progress_value: trajectory {tu} has pv_cap {sorted(map(str, caps))}; "
                             f"expected one positive cap")
        cap = caps.pop()
        if len(rows) > cap:
            raise ValueError(f"progress_value: trajectory {tu} has {len(rows)} turns, above its cap {cap}")
        rs = [_finite(rewards[i]) for i in rows]
        if any(r is None for r in rs):
            raise ValueError(f"progress_value: trajectory {tu} has a non-finite episode_rewards")
        task = tasks[rows[0]]
        # The horizon the table is fingerprinted with is a statement about its data: a trajectory run
        # under another cap would add targets of another objective (success within another H) to cells
        # keyed on fractions of another H, and be scored by them. (The trainer checks every row's
        # pv_cap against the same horizons before this; this holds for any caller.)
        want = cfg.horizons.get(task)
        if want is not None and cap != want:
            raise ValueError(f"progress_value: trajectory {tu} ({task}) ran under the cap {cap:g}; the table's "
                             f"horizon for {task} is {want}")
        if task == ALFWORLD:
            typ = alfworld_type(gamefiles[rows[0]])
        else:
            typ = task
        capped = capped_col is not None and any(_finite(capped_col[i]) == 1.0 for i in rows)
        trajs.append(_Traj(tuid=tu, uid=uids[rows[0]], task=task, typ=typ, rows=rows, cap=cap,
                           R=1.0 if max(rs) > 0.0 else 0.0, capped=capped))
    groups: Dict[str, List[_Traj]] = defaultdict(list)
    for tr in trajs:
        groups[tr.uid].append(tr)
    for u, trs in groups.items():
        if len({tr.task for tr in trs}) != 1:
            raise ValueError(f"progress_value: group {u} spans tasks {sorted({tr.task for tr in trs})}")

    # ---- the state of every real row: its cell and parent ----
    gamma, lam = cfg.gamma, cfg.lam
    keys: Dict[int, Tuple[tuple, tuple]] = {}
    feature_missing: Dict[int, bool] = {}
    for tr in trajs:
        fs = cfg.features[tr.task]
        for t, i in enumerate(tr.rows):
            rem = cfg.rem_bucket(tr.cap - t, tr.cap)
            vals = []
            for f in fs:
                if f == "type":
                    vals.append(tr.typ)
                elif f == "rem":
                    vals.append(rem)
                elif f == "stag":
                    vals.append(cfg.stag_bucket(feat_cols[PV_STAG_BEFORE][i]))
                elif f in FEATURE_BINS:
                    x = _finite(feat_cols[FEATURE_COLUMNS[f]][i])
                    vals.append(None if x is None else FEATURE_BINS[f](x))
                else:
                    vals.append(_key_value(feat_cols[FEATURE_COLUMNS[f]][i]))
            # A missing feature value is a cell of its own (None), still shrunk to its parent.
            feature_missing[i] = any(v is None for v in vals)
            keys[i] = ((tr.task, *vals), (tr.task, tr.typ, rem))

    # ---- V, delta, GAE per trajectory ----
    mass = {task: table.has_mass(task) for task in present}
    value = np.full(n, np.nan)
    delta = np.full(n, np.nan)
    a_rl = np.full(n, np.nan)
    a_fmt = np.zeros(n)
    fallback = np.zeros(n, dtype=bool)
    y_row = np.full(n, np.nan)
    records: List[tuple] = []
    for u in sorted(groups):
        trs = groups[u]
        m = len(trs)
        sum_r = sum(tr.R for tr in trs)
        u_all = [gamma ** tr.T * tr.R for tr in trs]
        sum_u = sum(u_all)
        for tr, u_i in zip(trs, u_all):
            T = tr.T
            if tr.capped:
                v = np.zeros(T)
            elif mass[tr.task]:
                v = np.array([table.value(*keys[i]) for i in tr.rows])
            else:
                # Empty table: the other rollouts' mean R, the same at every turn.
                v = np.full(T, (sum_r - tr.R) / (m - 1) if m > 1 else 0.0)
                fallback[tr.rows] = True
            d, a = gae(v, tr.R, gamma, lam)
            if cfg.prefix_discount:
                a = a * gamma ** np.arange(T, dtype=np.float64)
            # The episode term's baseline: the other trajectories' mean U, one weight per trajectory.
            b = (sum_u - u_i) / (m - 1) if m > 1 else 0.0
            a = (1.0 - cfg.eta) * (u_i - b) + cfg.eta * a
            value[tr.rows] = v
            delta[tr.rows] = d
            a_rl[tr.rows] = cfg.adv_scale * a
            y = gamma ** (T - np.arange(T, dtype=np.float64)) * tr.R
            if not tr.capped:
                y_row[tr.rows] = y
                records.extend((keys[i][0], keys[i][1], float(y[t])) for t, i in enumerate(tr.rows))

        # ---- the format term over the group's real turn rows ----
        tied = len({tr.R for tr in trs}) == 1
        if cfg.format_scope == "all" or tied:
            rows = [i for tr in trs for i in tr.rows]
            x = invalid_x[rows]
            if len(rows) > 1:
                sd = float(x.std(ddof=1))
                if sd >= FORMAT_STD_MIN:
                    a_fmt[rows] = (x - x.mean()) / sd

    advantage = a_rl + cfg.format_coef * a_fmt
    returns = a_rl / cfg.adv_scale + value

    # ---- padding copies take their original's values ----
    row_of = {(tr.tuid, t): i for tr in trajs for t, i in enumerate(tr.rows)}
    for i in np.flatnonzero(pad):
        src = row_of.get((tuids[i], turn_of(i)))
        if src is None:
            raise ValueError(f"progress_value: padding row {i} (trajectory {tuids[i]}, pv_t {pv_t[i]!r}) "
                             f"has no real row to copy")
        for arr in (advantage, a_rl, a_fmt, value, returns, delta, fallback):
            arr[i] = arr[src]
    # Every row is a real row or a copy of one, and every real row got a value above.
    assert np.isfinite(advantage).all() and np.isfinite(returns).all(), "progress_value: a row got no advantage"

    metrics = _metrics(trajs, present, table, columns, col, value, y_row, a_rl, a_fmt, delta, fallback,
                       feature_missing)
    return ProgressValueResult(advantage=advantage, a_rl=a_rl, a_fmt=a_fmt, value=value, returns=returns,
                               delta=delta, fallback=fallback, records=records, metrics=metrics)


def _metrics(trajs, present, table, columns, col, value, y_row, a_rl, a_fmt, delta, fallback,
             feature_missing) -> Dict[str, float]:
    """progress_value/<task>/*, over real rows; a metric whose population is empty is left out.

    Every metric of the value side (fallback_share, missing_feature_share, mean_v, calibration,
    the A_rl ones, delta_progress / delta_stagnant) runs over the rows the table scores: a WebShop
    goal_capped trajectory's rows are left out, as they are of the update -- their V, delta and
    A_rl are 0 by rule, not by the table, and counted in they would pull every mean towards 0 by
    the capped share, make mean_v - calibration differ from the batch's mean target, and put
    fallback_share below 1 on a step the table had no mass. capped_share reports how many
    trajectories that leaves out, capped_won how many of them won (should be 0). rows,
    trajectories and mean_abs_a_fmt count every real row: the format term does act on capped rows.
    """
    out: Dict[str, float] = {}
    k_before = col(PV_K_BEFORE) if PV_K_BEFORE in columns else None
    k_after = col(PV_K_AFTER) if PV_K_AFTER in columns else None

    def put(key, xs):
        if len(xs):
            out[key] = float(np.mean(xs))

    for task in present:
        p = f"progress_value/{task}/"
        trs = [tr for tr in trajs if tr.task == task]
        rows = np.array([i for tr in trs for i in tr.rows], dtype=np.int64)
        out[p + "rows"] = float(len(rows))
        out[p + "trajectories"] = float(len(trs))
        if task == WEBSHOP:
            put(p + "capped_share", [float(tr.capped) for tr in trs])
            # Capped trajectories that WON, a count: 0 if goal_capped is what it claims (no policy can
            # win the goal); any win says the flag is not a proof, and the V = 0 and the exclusion from
            # the table that rest on it are wrong for that goal. The trainer prints a warning on > 0.
            out[p + "capped_won"] = float(sum(1 for tr in trs if tr.capped and tr.R > 0))
        # The rows the table scores (and is updated with): every real row but a capped trajectory's.
        scored = np.array([i for tr in trs if not tr.capped for i in tr.rows], dtype=np.int64)
        win = np.array([tr.R > 0 for tr in trs if not tr.capped for _ in tr.rows], dtype=bool)
        put(p + "fallback_share", fallback[scored].astype(np.float64))
        put(p + "missing_feature_share", [float(feature_missing[i]) for i in scored])
        put(p + "mean_v", value[scored])
        # Calibration on this batch: the table's prediction against the target it will be updated with.
        if len(scored):
            out[p + "calibration"] = float(np.mean(value[scored]) - np.mean(y_row[scored]))
        put(p + "mean_abs_a_rl", np.abs(a_rl[scored]))
        put(p + "a_rl_success", a_rl[scored[win]])
        put(p + "a_rl_failure", a_rl[scored[~win]])
        put(p + "failure_pos_share", (a_rl[scored[~win]] > 0).astype(np.float64))
        put(p + "mean_abs_a_fmt", np.abs(a_fmt[rows]))
        cells, mass = table.task_stats(task)
        out[p + "table_cells"] = float(cells)
        out[p + "table_mass"] = mass
        if k_before is not None and k_after is not None:
            # A progress turn raised k_hist; a stagnant one left it (k_hist never falls).
            prog, stag = [], []
            for i in scored:
                kb, ka = _finite(k_before[i]), _finite(k_after[i])
                if kb is not None and ka is not None:
                    (prog if ka > kb else stag).append(delta[i])
            put(p + "delta_progress", prog)
            put(p + "delta_stagnant", stag)
    return out
