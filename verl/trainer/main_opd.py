"""
Main entry point for multitask OPD (On-Policy Distillation) training.

Each sample is distilled from a separate, single-task RL-trained teacher selected
by its ``task_name``. The student is trained purely by the KL to its per-task
teacher on the student's own on-policy responses; GRPO policy-gradient, entropy,
reference-KL and reward signals are all disabled so no other signal enters the loss.
"""

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.utils import actor_capture


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_opd(config)


def inject_distillation_config(config) -> None:
    """Apply the settings BOTH OPD arms share: what the teacher KL is, and what
    support it is computed on.

    Split from :func:`inject_opd_config` so the OPD+GRPO arm gets the identical
    distillation half rather than a copy of it. The arms are meant to differ in
    exactly one thing -- whether a policy gradient is in the loss -- and an A/B
    between them says nothing if the teacher KL was also configured twice.
    Everything policy-gradient-shaped is left to the callers.

    A module-level function rather than an inline block because it is the
    difference between what a script asks for and what the run actually gets:
    the intent lock validates the config AFTER this, and a test that wants to
    know whether an arm will start has to apply the same thing rather than a
    copy of it that can drift.
    """
    opd_cfg = config.algorithm.get("opd", {})
    with open_dict(config):
        config.actor_rollout_ref.actor.use_teacher_kl_loss = True
        config.actor_rollout_ref.actor.teacher_kl_loss_coef = opd_cfg.get("kl_loss_coef", 1.0)
        config.actor_rollout_ref.actor.teacher_kl_loss_type = opd_cfg.get("kl_loss_type", "low_var_kl")
        # top-k (+tail) dense KL support size; only used when kl_loss_type=topk_kl.
        config.actor_rollout_ref.actor.teacher_kl_topk = opd_cfg.get("topk", 20)
        # Equal per-task share of the teacher-KL loss, instead of the token-count
        # share the plain token-mean gives. Scientific knob, so it is surfaced
        # under algorithm.opd like the other loss settings and pinned in the
        # expectations file rather than left to the actor's default.
        config.actor_rollout_ref.actor.normalize_loss_by_task = opd_cfg.get(
            "normalize_loss_by_task", False
        )
        # Cross-teacher sign agreement. The weights are built inside the
        # actor's forward -- that is where the student's top-k exists -- so the
        # settings have to reach the actor config, while staying authored under
        # algorithm.opd with the other scientific knobs.
        sign_cfg = opd_cfg.get("sign_weight", None)
        if sign_cfg is not None:
            config.actor_rollout_ref.actor.sign_weight = sign_cfg
        if sign_cfg is not None and bool(sign_cfg.get("enable", False)):
            # The frozen models the weights read (base, and each row's off-task
            # teachers) are scored at ids nobody knows until the training forward
            # picks a support, so they have to keep their hidden states and
            # normaliser -- which is exactly what ref.student_indexed_topk buys.
            #
            # Forced rather than left to the ppo_trainer.yaml interpolation
            # ${actor_rollout_ref.actor.student_indexed_topk}, because that ties it
            # to the ACTOR's support choice, and the two are independent: a
            # teacher-indexed arm takes its support from the teacher and still
            # needs base and the off-task teachers resolvable at those ids. With
            # the interpolation, actor.student_indexed_topk=False turned the
            # caching off and the weights had nothing to read.
            config.actor_rollout_ref.ref.student_indexed_topk = True
            assert config.actor_rollout_ref.ref.get("response_only_logits", False), (
                "sign weighting needs ref.response_only_logits=True: the row map the "
                "hidden-state cache is packed against comes from there"
            )
        # Off on both arms: neither wants a reference-KL term or the SDL/SDAR
        # distillation paths, so they belong to the shared half rather than to
        # whichever arm happens to be running.
        config.actor_rollout_ref.actor.use_kl_loss = False       # no reference-KL term
        config.actor_rollout_ref.actor.use_sdl_loss = False
        config.actor_rollout_ref.actor.use_sdar_loss = False


def inject_opd_config(config) -> None:
    """Force the pure-distillation invariants onto the composed config.

    The shared distillation half, plus the part that makes this arm PURE: no
    signal other than the per-task teacher KL may reach the loss. The OPD+GRPO
    arm applies the same shared half and deliberately leaves these four alone.
    """
    inject_distillation_config(config)
    with open_dict(config):
        config.actor_rollout_ref.actor.pg_loss_coef = 0          # no GRPO policy gradient
        config.actor_rollout_ref.actor.entropy_coeff = 0         # no entropy bonus
        config.algorithm.use_kl_in_reward = False                # reward never shapes the loss


def init_ray_for_opd(config) -> None:
    """Stand up Ray with the runtime env both OPD arms use.

    Shared so the OPD+GRPO arm cannot end up under a different runtime env --
    the Nsight plugin below rewrites how every worker process is launched, which
    is not something one arm should get and the other not.
    """
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray_init_kwargs = OmegaConf.to_container(ray_init_kwargs)
        if actor_capture.nsys_enabled():
            # The ranks are Ray actors, not children of this launcher, so
            # wrapping the launch command in `nsys profile` reaches nothing that
            # touches a GPU. Ray's _nsight plugin wraps each worker process
            # instead; the capture ranges dp_actor opens are what keep the report
            # to a handful of micro-batches. Reports land in the Ray session's
            # logs/nsight directory, one per process.
            #
            # nsys_enabled(), not enabled(): the torch backend writes its own
            # trace from inside the rank and wants no plugin, so turning it on
            # must not drag every worker under Nsight as well.
            ray_init_kwargs["runtime_env"]["_nsight"] = actor_capture.nsight_runtime_env()
            print(f"[nsys] workers will run under Nsight: "
                  f"{ray_init_kwargs['runtime_env']['_nsight']}", flush=True)
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**ray_init_kwargs)


def run_opd(config) -> None:
    init_ray_for_opd(config)
    runner = OPDTaskRunner.remote()
    ray.get(runner.run.remote(config))


def build_and_fit(config, *, inject_fn, trainer_cls, tag: str, label: str, example_dir: str) -> None:
    """Compose the run and hand it to ``trainer_cls``.

    Shared by both OPD entrypoints. Everything here -- the env construction, the
    tokenizer, the worker mapping, the resource pools, the reward managers, the
    datasets -- is meant to be the same on the pure-OPD and the OPD+GRPO arm,
    because that sameness is the premise of comparing them. The two things that
    genuinely differ are parameters:

    * ``inject_fn`` -- which invariants are forced onto the composed config
      (pure OPD zeroes the policy gradient; OPD+GRPO leaves it to the script);
    * ``trainer_cls`` -- which ``fit()`` loop runs.

    ``tag``/``label``/``example_dir`` only name the arm in messages and in the
    intent lock's failure output.
    """
    from pprint import pprint

    from omegaconf import OmegaConf

    from verl.utils.fs import copy_to_local

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    opd_cfg = config.algorithm.get("opd", {})
    teacher_paths = opd_cfg.get("teacher_paths", None)
    assert teacher_paths is not None, (
        f"{label} requires algorithm.opd.teacher_paths.{{alfworld,search,webshop}} "
        "(pass via +algorithm.opd.teacher_paths.<task>=/path)"
    )

    inject_fn(config)

    # Fail-fast intent check: validate the EFFECTIVE config (after the
    # injection above) against the version-controlled expectations file.
    # Required — a run without a pinned intent is exactly how the
    # low_var_kl-instead-of-topk_kl mishap happened.
    from verl.utils.expected_config import enforce_expected_config

    expect_file = config.trainer.get("expected_config", None)
    assert expect_file is not None, (
        f"{label} requires +trainer.expected_config=<expectations yaml> "
        f"(see examples/{example_dir}/expected_multitask_config.yaml)"
    )
    enforce_expected_config(config, expect_file, tag=f"{tag} expected-config")

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )

    from agent_system.environments import make_envs

    envs, val_envs = make_envs(config)

    from verl.utils import hf_processor, hf_tokenizer

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    if config.actor_rollout_ref.rollout.name in ["vllm"]:
        from verl.utils.vllm_utils import is_version_ge

        if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
            if not is_version_ge(pkg="vllm", minver="0.7.3"):
                raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        assert config.critic.strategy in ["fsdp", "fsdp2"]
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

        actor_rollout_cls = (
            AsyncActorRolloutRefWorker
            if config.actor_rollout_ref.rollout.mode == "async"
            else ActorRolloutRefWorker
        )
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy in ["fsdp", "fsdp2"]:
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # NOTE: neither OPD arm registers Role.RefPolicy. Teachers are created inside
    # OPDRayTrainer.init_workers as additional role="ref" worker groups, one per task.

    reward_manager_name = config.reward_model.get("reward_manager", "episode")
    if reward_manager_name == "episode":
        from agent_system.reward_manager import EpisodeRewardManager

        reward_manager_cls = EpisodeRewardManager
    else:
        raise NotImplementedError

    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    assert config.actor_rollout_ref.rollout.n == 1, (
        "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. "
        "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
    )

    from agent_system.multi_turn_rollout import TrajectoryCollector

    traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
    train_sampler = create_rl_sampler(config.data, train_dataset)

    teacher_paths_plain = (
        OmegaConf.to_container(teacher_paths, resolve=True)
        if OmegaConf.is_config(teacher_paths)
        else dict(teacher_paths)
    )
    print(f"[{label}] teacher_paths: {teacher_paths_plain}")
    print(f"[{label}] teacher_kl_loss_coef: {config.actor_rollout_ref.actor.teacher_kl_loss_coef}")
    print(f"[{label}] teacher_kl_loss_type: {config.actor_rollout_ref.actor.teacher_kl_loss_type}")
    # Printed on both arms: on pure OPD it is the injection's receipt that the
    # policy gradient really is off, and on OPD+GRPO it is the ratio between the
    # two terms -- the single number the arms are meant to differ by.
    print(f"[{label}] pg_loss_coef: {config.actor_rollout_ref.actor.pg_loss_coef}")
    print(f"[{label}] adv_estimator: {config.algorithm.adv_estimator}")

    trainer = trainer_cls(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        device_name=config.trainer.device,
        traj_collector=traj_collector,
        envs=envs,
        val_envs=val_envs,
    )
    trainer.init_workers()
    trainer.fit()


@ray.remote(num_cpus=1)
class OPDTaskRunner:
    def run(self, config):
        from verl.trainer.ppo.opd_ray_trainer import OPDRayTrainer

        build_and_fit(
            config,
            inject_fn=inject_opd_config,
            trainer_cls=OPDRayTrainer,
            tag="opd",
            label="OPD",
            example_dir="opd_trainer",
        )


if __name__ == "__main__":
    main()
