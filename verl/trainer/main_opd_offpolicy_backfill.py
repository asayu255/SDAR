"""
top-k backfill: turn SFT teacher trajectories into off-policy KD data.

The SFT Stage-1 dataset (``teacher_traj_sft/<task>.pt``, generated with
``gen.collect_topk=False``) stores only the teacher's token sequences, not its
top-k distribution. Off-policy KD's Stage-2 loss (``topk_kl``) needs the teacher
top-k (``teacher_topk_logprobs`` / ``teacher_topk_ids``). Rather than regenerate
the (expensive) rollouts, this script *reuses* the SFT trajectories and only adds
the missing top-k: it loads ``<in_dir>/<task>.pt``, scores the stored responses
with that task's frozen teacher (a ``role="ref"`` worker — no vLLM, no envs), and
writes ``<out_dir>/<task>.pt`` with the top-k columns appended.

The top-k is a deterministic teacher forward over the exact stored
``(input_ids, responses)``, so it is identical to what Stage-1 would have
computed inline — no approximation. Run once per task, e.g.:

    python3 -m verl.trainer.main_opd_offpolicy_backfill \
        +backfill.task=alfworld \
        +backfill.teacher_path=/path/to/alfworld_teacher \
        +backfill.in_dir=$HOME/data/.../teacher_traj_sft \
        +backfill.out_dir=$HOME/data/.../teacher_traj \
        +backfill.topk=20 actor_rollout_ref.model.path=Qwen/Qwen3-1.7B ...
"""

import os

import hydra
import ray
from omegaconf import OmegaConf, open_dict


def _chunk_bounds(n: int, chunk_size: int):
    """[(start, end), ...] covering range(n) in <= chunk_size blocks, in order."""
    chunk_size = max(1, int(chunk_size))
    return [(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_backfill(config)


def run_backfill(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = BackfillRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class BackfillRunner:
    def run(self, config):
        import torch

        from verl import DataProto
        from verl.protocol import DataProtoConfig
        from verl.utils.fs import copy_to_local

        OmegaConf.resolve(config)

        bf = config.get("backfill", None)
        assert (
            bf is not None
            and bf.get("task", None)
            and bf.get("teacher_path", None)
            and bf.get("in_dir", None)
            and bf.get("out_dir", None)
        ), (
            "backfill requires +backfill.task=<task> +backfill.teacher_path=<ckpt> "
            "+backfill.in_dir=<sft_dir> +backfill.out_dir=<offpolicy_dir>"
        )
        task = bf.task
        topk = int(bf.get("topk", config.actor_rollout_ref.actor.get("teacher_kl_topk", 20)))
        chunk_size = int(bf.get("chunk_size", 4096))
        in_path = os.path.join(bf.in_dir, f"{task}.pt")
        out_path = os.path.join(bf.out_dir, f"{task}.pt")
        assert os.path.exists(in_path), f"SFT trajectory file not found: {in_path}"

        # Score with the task's frozen teacher; top-k is unsupported with fused kernels.
        with open_dict(config):
            config.actor_rollout_ref.model.path = bf.teacher_path
            config.actor_rollout_ref.model.lora_rank = 0
            config.actor_rollout_ref.model.use_fused_kernels = False

        copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        # ---- Build ONE teacher ref worker (FSDP; no vLLM, no envs). ----
        from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
        from verl.single_controller.ray.base import create_colocated_worker_cls
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {Role.ActorRollout: global_pool_id}
        rpm = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        rpm.create_resource_pool()
        resource_pool = rpm.get_resource_pool(Role.ActorRollout)

        teacher_cls = RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker),
            config=config.actor_rollout_ref,
            role="ref",
        )
        worker_dict_cls = create_colocated_worker_cls(class_dict={"ref": teacher_cls})
        wg_kwargs = {}
        if OmegaConf.select(config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = config.trainer.ray_wait_register_center_timeout
        wg_dict = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            device_name=config.trainer.device,
            **wg_kwargs,
        )
        teacher_wg = wg_dict.spawn(prefix_set={"ref"})["ref"]
        teacher_wg.init_model()

        # ---- Load SFT trajectories and append teacher top-k in chunks. ----
        data = DataProto.load_from_disk(in_path)
        assert "teacher_topk_logprobs" not in data.batch, (
            f"{in_path} already has teacher top-k; nothing to backfill"
        )
        n = len(data)
        print(f"[backfill][{task}] loaded {n} rows from {in_path}; scoring top-{topk} ...")

        tlp_list, tid_list = [], []
        for start, end in _chunk_bounds(n, chunk_size):
            sub = data.slice(start, end)
            topk_in = sub.select(batch_keys=["responses", "input_ids", "attention_mask", "position_ids"])
            topk_in.meta_info = dict(topk_in.meta_info)
            topk_in.meta_info["topk_k"] = topk
            # Chunk is not generally divisible by the DP world size.
            topk_in.meta_info[DataProtoConfig.auto_padding_key] = True
            out = teacher_wg.compute_ref_topk_log_prob(topk_in)
            tlp_list.append(out.batch["teacher_topk_logprobs"])
            tid_list.append(out.batch["teacher_topk_ids"])
            print(f"[backfill][{task}] scored {end}/{n} rows")

        teacher_topk_logprobs = torch.cat(tlp_list, dim=0)
        teacher_topk_ids = torch.cat(tid_list, dim=0)
        assert teacher_topk_logprobs.size(0) == n and teacher_topk_ids.size(0) == n, (
            f"top-k row count {teacher_topk_logprobs.size(0)} != trajectory rows {n}"
        )
        data.batch["teacher_topk_logprobs"] = teacher_topk_logprobs
        data.batch["teacher_topk_ids"] = teacher_topk_ids

        os.makedirs(bf.out_dir, exist_ok=True)
        data.save_to_disk(out_path)
        print(
            f"[backfill][{task}] wrote {n} rows with teacher top-{topk} "
            f"(shape {tuple(teacher_topk_logprobs.shape)}) -> {out_path}"
        )


if __name__ == "__main__":
    main()
