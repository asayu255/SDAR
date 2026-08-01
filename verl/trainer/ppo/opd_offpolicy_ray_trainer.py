"""
Off-policy multitask distillation (offline KD) — Stage 1 generator + Stage 2 trainer.

This is the *off-policy* sibling of the on-policy OPD trainer
(``verl/trainer/ppo/opd_ray_trainer.py``). The standard form of off-policy
distillation trains the student on a *fixed dataset of sequences sampled from the
teacher* (sequence-level / GKD off-policy branch), rather than on the student's
own on-policy rollouts.

Two stages, sharing all data / env / config / teacher / loss machinery with OPD:

* **Stage 1 — ``TeacherTrajectoryGenerator``**: for a single task, a frozen
  per-task teacher (loaded as the ``actor_rollout`` model) rolls out multi-turn
  trajectories in that task's environment, then scores its own top-k
  (``teacher_topk_logprobs`` / ``teacher_topk_ids``) over the generated
  responses. The result is dumped to ``<out_dir>/<task>.pt`` as a ``DataProto``.

* **Stage 2 — ``OffPolicyOPDRayTrainer``**: loads the per-task ``.pt`` files,
  concatenates them into one fixed off-policy dataset, and trains the student in
  3-task-balanced batches with the *same* top-k teacher-KL used by OPD
  (``teacher_kl_loss_type=topk_kl``, ``teacher_kl_topk=20``). No generation and
  no teacher forward pass happen during training — the teacher top-k is already
  baked into the dataset. Only validation rolls the student out (identical to
  OPD via the inherited ``_validate``).
"""

import glob
import os
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.protocol import DataProtoConfig
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.opd_ray_trainer import compute_opd_data_metrics, compute_opd_data_metrics_by_task
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    compute_response_mask,
)
from verl.utils.metric import reduce_metrics

from agent_system.multi_turn_rollout import adjust_batch, batch_size_divisor

# Tensor fields persisted per trajectory (Stage 1 -> Stage 2). These are exactly
# the keys update_policy's top-k distillation path consumes, plus prompts/
# response_mask for monitoring and balance_batch.
_SAVE_TENSOR_KEYS = [
    "responses",
    "input_ids",
    "attention_mask",
    "position_ids",
    "prompts",
    "response_mask",
    "teacher_topk_logprobs",
    "teacher_topk_ids",
]
_SAVE_NON_TENSOR_KEYS = ["task_name", "traj_uid"]

# Saved by Stage 1 but never read by Stage 2, and together a quarter of every row
# (prompts is 4096 int64 columns, response_mask another 512), so they are dropped
# as each file is loaded rather than carried in host RAM for the whole run:
#   prompts       - only read by the *on-policy* trainer's rollout_data_dir dump
#                   (opd_ray_trainer), which works off live rollouts, not this pool.
#   response_mask - the training loop recomputes it from attention_mask every step
#                   via compute_response_mask, and _compute_response_info slices
#                   attention_mask directly, so the stored copy is never consulted.
# Dropping happens at load, so pools already on disk need no regeneration.
# Subclasses whose loss reads *less* than the top-k KD loss extend this via the
# ``_drop_tensor_keys`` class attribute.
_DROP_TENSOR_KEYS = ("prompts", "response_mask")

# Keys popped from the prompt batch to form the generation input (mirrors OPD).
_GEN_BATCH_KEYS = ["input_ids", "attention_mask", "position_ids"]
_GEN_NON_TENSOR_KEYS = ["raw_prompt_ids", "data_source"]


def find_padding_duplicates(traj_uids: np.ndarray) -> np.ndarray:
    """Boolean mask of the rows an earlier Stage 1 appended as adjust_batch padding.

    Pools generated before that call was made optional carry duplicated turn-rows:
    every gen step was padded up to a multiple of lcm(log_prob_micro*W,
    actor_micro*W) by copying randomly chosen rows onto the end of the step's
    block. The copies were saved, so those turns are trained on twice.

    They are identified structurally rather than by hashing. A Stage-1 block is
    laid out trajectory-major -- ``gather_rollout_data`` iterates trajectories in
    the outer loop and turns in the inner one -- so each trajectory's turn-rows are
    contiguous, and ``adjust_batch`` concatenates its copies after all of them.
    A traj_uid starting a *second* run can therefore only be padding.

    The one row this misses is a copy that landed immediately after its own
    trajectory's run and merged into it, which needs the copy to be drawn from the
    block's last trajectory: ~1/120 per block, so ~0.5 rows per file against tens
    of thousands of padding rows. The error direction is safe -- a missed copy is
    kept (trained twice, as today), never an original dropped.
    """
    n = len(traj_uids)
    if n == 0:
        return np.zeros(0, dtype=bool)
    uids = np.asarray(traj_uids)
    change = np.ones(n, dtype=bool)
    change[1:] = uids[1:] != uids[:-1]
    run_starts = np.flatnonzero(change)
    run_ends = np.append(run_starts[1:], n)

    is_dup = np.zeros(n, dtype=bool)
    seen = set()
    for start, end in zip(run_starts.tolist(), run_ends.tolist()):
        uid = uids[start]
        if uid in seen:
            is_dup[start:end] = True
        else:
            seen.add(uid)
    return is_dup


def _build_gen_batch(batch: DataProto) -> DataProto:
    """Pop the model-input fields out of a prompt batch (same set as OPD/_validate)."""
    non_tensor_keys = list(_GEN_NON_TENSOR_KEYS)
    for opt in ("multi_modal_data", "raw_prompt", "tools_kwargs", "env_kwargs", "task_name"):
        if opt in batch.non_tensor_batch:
            non_tensor_keys.append(opt)
    return batch.pop(batch_keys=list(_GEN_BATCH_KEYS), non_tensor_batch_keys=non_tensor_keys)


class TeacherTrajectoryGenerator(RayPPOTrainer):
    """Stage 1: roll out one task's frozen teacher and dump trajectories + top-k.

    The teacher checkpoint is loaded as the ``actor_rollout`` model (set
    ``config.actor_rollout_ref.model.path`` to the teacher path before
    construction), so the existing rollout engine generates the teacher's
    trajectories and ``compute_actor_topk_log_prob`` scores its top-k.
    """

    def generate(self):
        gen_cfg = self.config.gen
        task = gen_cfg.task
        out_dir = gen_cfg.out_dir
        topk = int(gen_cfg.get("topk", self.config.actor_rollout_ref.actor.get("teacher_kl_topk", 20)))
        # Optional cap on the number of unique trajectories to collect; default:
        # one pass over the loader.
        target = gen_cfg.get("num_trajectories", None)
        target = int(target) if target else None

        # Incremental sharding. Without it every turn-row of the whole run is held
        # in this actor's RAM until the end, and then DataProto.concat allocates a
        # second full copy -- so the run can die at its very last step, after
        # hours. Rows are padded to data.max_prompt_length + data.max_response_length
        # for every task (the per-task max_prompt_length override applies only to
        # the initial dataset load, not to the rollout's per-turn retokenization),
        # which at 4096+512 and topk=20 is ~275 KB/row.
        # shard_every_steps=0 keeps the single-file behavior.
        shard_every = int(gen_cfg.get("shard_every_steps", 0) or 0)
        shard_idx = 0
        shard_paths = []

        # Padding each step up to a DP/micro-divisible size duplicates random rows
        # and writes the copies into <task>.pt, so it is off by default: Stage 2
        # needs no divisibility here (the only worker call pads and unpads itself),
        # and the copies would otherwise be trained on twice.
        #
        # It exists as an option for one case: extending a pool whose other tasks
        # were generated when this was unconditional. The loader drops the padding
        # at load, so it makes no difference to a run on this branch -- but a pool
        # should be uniform in the format it was written in.
        pad_batches = bool(gen_cfg.get("adjust_batch", False))
        pad_divisor = batch_size_divisor(self.config) if pad_batches else None
        if pad_batches:
            print(f"[OPD-offpolicy gen][{task}] adjust_batch ON, padding each step up "
                  f"to a multiple of {pad_divisor}", flush=True)
            # The divisor comes from the micro batch sizes and world size, so a
            # config that differs from the one the rest of the pool was written
            # with silently changes how much padding lands on disk. Checking it
            # here costs seconds; discovering it afterwards costs the whole run.
            expected = gen_cfg.get("expect_pad_divisor", None)
            assert expected is None or int(expected) == pad_divisor, (
                f"adjust_batch would pad to a multiple of {pad_divisor}, but "
                f"gen.expect_pad_divisor={int(expected)}. The divisor is "
                f"lcm(rollout.log_prob_micro_batch_size_per_gpu, "
                f"actor.ppo_micro_batch_size_per_gpu) * n_gpus_per_node * nnodes -- "
                f"set those to match the pool this task is joining "
                f"(scripts/inspect_teacher_pool.py reports the pool's divisor)."
            )

        self.global_steps = 0
        collected = []
        seen_trajs = set()
        n_collected = 0
        n_rows_total = 0
        os.makedirs(out_dir, exist_ok=True)

        # Stage 2 globs *.pt, so this task's previous output would be loaded
        # alongside the new one -- and traj_uid is a uuid4, so the duplicate check
        # would not catch it: the pool would simply hold twice the trajectories.
        # Regenerating a task replaces it, so clear what is there first. Runs
        # whether or not this pass shards, because the two layouts have different
        # filenames and either can be what the previous pass left behind.
        stale = sorted(glob.glob(os.path.join(out_dir, f"{task}_[0-9]*.pt")))
        legacy = os.path.join(out_dir, f"{task}.pt")
        if os.path.exists(legacy):
            stale.append(legacy)
        for path in stale:
            print(f"[OPD-offpolicy gen][{task}] removing stale output {path}", flush=True)
            os.remove(path)

        def _flush_shard():
            nonlocal collected, shard_idx
            if not collected:
                return
            shard = DataProto.concat(collected)
            path = os.path.join(out_dir, f"{task}_{shard_idx:04d}.pt")
            shard.save_to_disk(path)
            shard_paths.append(path)
            print(f"[OPD-offpolicy gen][{task}] wrote shard {path} ({len(shard)} turn-rows)", flush=True)
            collected = []
            shard_idx += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # Phase timing. The names matter beyond the printed seconds: _timer
                # pushes each name as a gpu_profiler phase, and the profiler's
                # background sampler is started lazily by that first push — so
                # without this instrumentation GPU_PROFILER=1 is a silent no-op for
                # Stage 1 and ROLLOUT_TURN_TIMING's genGPU%/DP columns stay empty.
                # Popping "step" (gpu_profiler's default GPU_PROFILER_BOUNDARY) is
                # what emits the per-step utilization report.
                timing_raw = {}
                reached_target = False
                with _timer("step", timing_raw):
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    gen_batch = _build_gen_batch(batch)
                    del batch

                    # Teacher generates the trajectories (is_train=True -> repeat by env.rollout.n).
                    with _timer("gen", timing_raw):
                        gen_out = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )
                    # Off by default -- nothing in this stage needs the divisibility
                    # (the worker call below pads and unpads itself via
                    # auto_padding_key) and the copies it appends get written into
                    # the dataset. See gen.adjust_batch above for the one reason to
                    # turn it back on.
                    if pad_batches:
                        gen_out = adjust_batch(self.config, gen_out)
                    gen_out.batch["response_mask"] = compute_response_mask(gen_out)

                    # Teacher scores its own top-k over the generated responses.
                    with _timer("teacher_topk", timing_raw):
                        topk_in = gen_out.select(
                            batch_keys=["responses", "input_ids", "attention_mask", "position_ids"]
                        )
                        topk_in.meta_info = dict(topk_in.meta_info)
                        topk_in.meta_info["topk_k"] = topk
                        # Per-call batch is not generally divisible by the DP world size.
                        topk_in.meta_info[DataProtoConfig.auto_padding_key] = True
                        topk_out = self.actor_rollout_wg.compute_actor_topk_log_prob(topk_in)
                        gen_out.batch["teacher_topk_logprobs"] = topk_out.batch["teacher_topk_logprobs"]
                        gen_out.batch["teacher_topk_ids"] = topk_out.batch["teacher_topk_ids"]

                    # Backfill task_name if the rollout dropped it (single-task generation).
                    if "task_name" not in gen_out.non_tensor_batch:
                        gen_out.non_tensor_batch["task_name"] = np.array(
                            [task] * len(gen_out), dtype=object
                        )

                    with _timer("collect", timing_raw):
                        tensor_keys = [k for k in _SAVE_TENSOR_KEYS if k in gen_out.batch]
                        non_tensor_keys = [k for k in _SAVE_NON_TENSOR_KEYS if k in gen_out.non_tensor_batch]
                        keep = gen_out.select(batch_keys=tensor_keys, non_tensor_batch_keys=non_tensor_keys)
                        keep = keep.to("cpu")
                        collected.append(keep)
                        n_rows_total += len(keep)
                    if "traj_uid" in keep.non_tensor_batch:
                        seen_trajs.update(keep.non_tensor_batch["traj_uid"].tolist())
                        n_collected = len(seen_trajs)
                    else:
                        # No uid to deduplicate on; count turn-rows. Uses the running
                        # total rather than the buffer, which sharding empties.
                        n_collected = n_rows_total
                    self.global_steps += 1
                    if target is not None and n_collected >= target:
                        reached_target = True

                # Printed after the "step" timer closes so its total is available.
                print(f"[OPD-offpolicy gen][{task}] step {self.global_steps} "
                      f"collected {n_collected} trajectories "
                      f"({n_rows_total} turn-rows)  "
                      f"[step {timing_raw.get('step', 0):.1f}s = "
                      f"gen {timing_raw.get('gen', 0):.1f}s + "
                      f"topk {timing_raw.get('teacher_topk', 0):.1f}s + "
                      f"collect {timing_raw.get('collect', 0):.1f}s]", flush=True)

                if shard_every > 0 and (reached_target or self.global_steps % shard_every == 0):
                    _flush_shard()

                if reached_target:
                    break
            else:
                continue
            break

        if shard_every > 0:
            _flush_shard()  # remainder since the last flush
            print(f"[OPD-offpolicy gen][{task}] saved {n_collected} trajectories "
                  f"({n_rows_total} turn-rows) -> {len(shard_paths)} shards in {out_dir}")
            return out_dir

        data = DataProto.concat(collected)
        out_path = os.path.join(out_dir, f"{task}.pt")
        data.save_to_disk(out_path)
        print(f"[OPD-offpolicy gen][{task}] saved {n_collected} trajectories "
              f"({len(data)} turn-rows) -> {out_path}")
        return out_path


class OffPolicyOPDRayTrainer(RayPPOTrainer):
    """Stage 2: off-policy distillation on the fixed teacher-trajectory dataset.

    Reuses the base worker setup (student ``actor_rollout`` only; no reference
    policy, no teacher workers) and the inherited ``_validate`` /
    ``_save_checkpoint``. Only the training loop differs from OPD: it iterates a
    fixed off-policy dataset in 3-task-balanced batches and applies the *same*
    top-k teacher-KL via ``update_actor`` (no rollout, no teacher forward).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.use_reference_policy, "off-policy OPD must not create a reference-policy worker"
        opd_cfg = self.config.algorithm.get("opd", {})
        self.teacher_data_dir = opd_cfg.get("teacher_data_dir", None)
        assert self.teacher_data_dir is not None, (
            "off-policy OPD requires algorithm.opd.teacher_data_dir "
            "(directory of Stage-1 <task>.pt files)"
        )
        # Per-step, per-task TRAJECTORY count == OPD's per-task prompts * group size
        # (15 * 8 = 120). Each step draws this many whole trajectories per task and
        # expands them to all their turn-rows, matching OPD's per-step composition.
        per_task_prompts = int(self.config.data.task_balance.per_task_batch_size)
        group_size = int(self.config.env.rollout.n)
        self.per_task_traj_per_step = per_task_prompts * group_size
        self._load_offpolicy_data()

    # Tensor columns discarded as each Stage-1 file is loaded. A subclass whose
    # loss consumes fewer columns than the top-k KD loss overrides this to drop
    # more; it must never drop fewer, since everything listed here is already
    # unread by *any* Stage-2 loss.
    _drop_tensor_keys = _DROP_TENSOR_KEYS

    @classmethod
    def _load_offpolicy_file(cls, path: str) -> DataProto:
        """Load one Stage-1 ``<task>.pt``, dropping padding rows and dead columns.

        Pools written while Stage 1 still called adjust_batch unconditionally carry
        duplicated turn-rows. Dropping them here rather than zero-weighting them also
        corrects the row count, and with it the number of mini-batches a step is
        split into: left in, they inflate the optimizer-update count for the same
        real data (measured ~+10% overall, driven by the short-episode tasks where
        the fixed ~120 padding rows per gen step are a large fraction of the block).

        Removal cannot lose data: adjust_batch concatenated its copies onto the
        *retained* batch, so the row a copy was made from is always still present.
        Row order is preserved, so the first-seen order of traj_uids -- and hence
        the trajectory sampling sequence -- is unchanged.

        ``cls._drop_tensor_keys`` go too. They are a quarter of every row and nothing
        in this stage reads them, so keeping them only costs host RAM -- see
        ``_DROP_TENSOR_KEYS`` for why each is dead, and the class attribute for how a
        subclass with a narrower loss drops more.

        Both happen in one column-wise pass that releases each source column as soon
        as its replacement exists. Doing it with select_idxs instead would hold the
        whole file and its filtered copy at once, which at 36k trajectories is a
        larger transient than the steady-state footprint this is trying to reduce.

        A pool already filtered by ``scripts/cache_teacher_pool.py`` passes straight
        through: there is no padding left to find and no listed column left to drop.
        """
        data = DataProto.load_from_disk(path)
        name = os.path.basename(path)

        keep_idx = None
        n_rows = len(data)
        if "traj_uid" in data.non_tensor_batch:
            is_dup = find_padding_duplicates(data.non_tensor_batch["traj_uid"])
            n_dup = int(is_dup.sum())
            if n_dup:
                keep_idx = torch.from_numpy(np.flatnonzero(~is_dup))
                print(f"[OPD-offpolicy] {name}: dropping {n_dup} Stage-1 padding rows "
                      f"({n_dup / n_rows:.1%}), keeping {n_rows - n_dup}")
        keep_np = keep_idx.numpy() if keep_idx is not None else None

        dropped_cols = []
        tensors = {}
        source = data.batch
        for key in list(source.keys()):
            column = source[key]
            if key in cls._drop_tensor_keys:
                dropped_cols.append(key)
            elif keep_idx is None:
                tensors[key] = column
            else:
                tensors[key] = column[keep_idx]
            del source[key]
            del column

        non_tensors = {
            key: (value if keep_np is None else value[keep_np])
            for key, value in data.non_tensor_batch.items()
        }
        if dropped_cols:
            print(f"[OPD-offpolicy] {name}: dropped unused columns {dropped_cols}")
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=data.meta_info)

    def _load_offpolicy_data(self):
        files = sorted(glob.glob(os.path.join(self.teacher_data_dir, "*.pt")))
        assert files, f"no Stage-1 <task>.pt files found in {self.teacher_data_dir}"
        parts = [self._load_offpolicy_file(f) for f in files]
        # Which file each row came from, so a trajectory that spans two of them can
        # be caught below. sorted() puts a task's shards in write order, so the
        # concatenated row order is the order a single unsharded file would have had.
        file_of_row = np.repeat(np.arange(len(parts)), [len(p) for p in parts])
        self.offpolicy_data = DataProto.concat(parts)
        del parts
        task_names = self.offpolicy_data.non_tensor_batch["task_name"]
        normalized = np.array([self._normalize_task_name(t) for t in task_names])
        assert "traj_uid" in self.offpolicy_data.non_tensor_batch, (
            "off-policy dataset must carry traj_uid for trajectory-level sampling"
        )
        traj_uids = self.offpolicy_data.non_tensor_batch["traj_uid"]
        # Group row indices by trajectory within each task, so a step can draw whole
        # trajectories (all their turn-rows) exactly like OPD's per-step rollout.
        self._task_to_traj_rows = {}  # task -> {traj_uid: np.ndarray(row_idx)}
        self._task_to_trajs = {}      # task -> np.ndarray(traj_uid) sampling population
        for task in sorted(set(normalized.tolist())):
            traj_to_rows = {}
            for ridx in np.where(normalized == task)[0]:
                traj_to_rows.setdefault(traj_uids[ridx], []).append(int(ridx))
            for uid, rows in traj_to_rows.items():
                # Shard boundaries fall on generation-step boundaries and a
                # trajectory is produced entirely within one step, so its rows are
                # always in one file. traj_uid is a uuid4, so a repeat across files
                # is a corrupt pool (a shard written twice, or two runs' output
                # mixed in one directory), not a split trajectory -- and silently
                # merging the two would make one "trajectory" whose turns come from
                # different rollouts.
                sources = np.unique(file_of_row[rows])
                assert len(sources) == 1, (
                    f"traj_uid {uid} appears in more than one file under "
                    f"{self.teacher_data_dir} ({[os.path.basename(files[s]) for s in sources]}); "
                    "the directory holds duplicate or overlapping shards"
                )
            self._task_to_traj_rows[task] = {u: np.array(r, dtype=np.int64) for u, r in traj_to_rows.items()}
            self._task_to_trajs[task] = np.array(list(traj_to_rows.keys()), dtype=object)
        sizes = {
            t: (len(self._task_to_trajs[t]), int(sum(len(r) for r in self._task_to_traj_rows[t].values())))
            for t in self._task_to_trajs
        }
        print(f"[OPD-offpolicy] loaded {len(self.offpolicy_data)} rows from {len(files)} files; "
              f"per-task (trajectories, rows): {sizes}")
        for task, trajs in self._task_to_trajs.items():
            assert len(trajs) > 0, f"no trajectories for task {task}"

    def _offpolicy_batch_iter(self):
        """Yield one task-balanced DataProto per training step. Each step draws
        ``per_task_traj_per_step`` whole trajectories per task (all their turn-rows),
        matching OPD's per-step trajectory count. Trajectory pools are reshuffled and
        recycled across steps (replay)."""
        seed = int(self.config.data.get("seed", 1))
        rng = np.random.default_rng(seed)
        pools = {t: rng.permutation(trajs) for t, trajs in self._task_to_trajs.items()}
        cursors = {t: 0 for t in pools}

        def _draw_trajs(task, n):
            out = []
            while len(out) < n:
                pool = pools[task]
                c = cursors[task]
                take = min(n - len(out), len(pool) - c)
                out.extend(pool[c : c + take].tolist())
                cursors[task] = c + take
                if cursors[task] >= len(pool):
                    pools[task] = rng.permutation(self._task_to_trajs[task])
                    cursors[task] = 0
            return out

        for _ in range(self.total_training_steps):
            rows = []
            for task in pools:
                for uid in _draw_trajs(task, self.per_task_traj_per_step):
                    rows.extend(self._task_to_traj_rows[task][uid].tolist())
            yield self.offpolicy_data.select_idxs(rows)

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="OPD-offpolicy Training")
        self.global_steps += 1
        last_val_metrics = None

        for batch in self._offpolicy_batch_iter():
            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps

            with _timer("step", timing_raw):
                # Pad the per-step batch to a DP/micro-divisible size (same as OPD's
                # post-rollout adjust_batch), then recompute response_mask / token counts.
                batch = adjust_batch(self.config, batch)
                batch.batch["response_mask"] = compute_response_mask(batch)
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # tag rows with their task so the actor can split its metrics
                self._attach_task_ids(batch)

                with _timer("update_actor", timing_raw):
                    # update_policy scales student logits by this temperature (same value
                    # compute_log_prob would set); OPD sets it here for the thin loop too.
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

                test_start_step = self.config.trainer.get("test_start_step", 0)
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                    with _timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            metrics.update({
                "training/global_step": self.global_steps,
            })
            metrics.update(compute_opd_data_metrics(batch=batch))
            metrics.update(compute_opd_data_metrics_by_task(batch=batch))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            logger.log(data=metrics, step=self.global_steps)
            progress_bar.update(1)
            self.global_steps += 1
            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return
