# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 【このファイルの役割（日本語補足）】
# - verl の PPO/GRPO 学習の「エントリポイント（起動口）」。
# - Hydra で設定(config)を読み込み、Ray クラスタを起動し、実際の学習ループを持つ
# RayPPOTrainer を組み立てて走らせる。
# - さらに、SDAR の3タスク同時学習で使う「TaskBalancedSampler（タスク均等サンプラ）」も
# このファイルで定義されている。ここが単タスク学習との分岐点になる。
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

# 環境変数 TASK_BALANCE_INTERLEAVE の読み取りに使う（後述の Sampler）
import os

# 設定管理フレームワーク。@hydra.main で config を注入する
import hydra
# 分散実行フレームワーク。学習を Ray アクター上で回す
import ray
# Hydra が使う設定オブジェクト（dict 風）の操作ユーティリティ
from omegaconf import OmegaConf

# 実体の学習ループを持つクラス
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
# 報酬マネージャのローダ（このファイルでは直接は未使用だが再エクスポート用に import）
from verl.trainer.ppo.reward import load_reward_manager
# Ray 実行環境（環境変数など）の既定値
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    # Hydra が config/ppo_trainer.yaml を読み込み、CLI 上書き（例: data.train_batch_size=45）を
    # マージした最終 config を引数に渡してくる。ここでは実処理を run_ppo に委譲するだけ。
    run_ppo(config)


def run_ppo(config) -> None:
    # Check if Ray is not initialized
    # Ray がまだ起動していなければ、ここでローカルクラスタを立ち上げる。
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        # get_ppo_ray_runtime_env(): tokenizer 並列制御・NCCL/VLLM のログレベルなど、
        # 学習全体に効かせたい環境変数の既定セットを返す。
        default_runtime_env = get_ppo_ray_runtime_env()
        # config 内に ray_init セクションがあれば取り出す（無ければ空 dict）。
        ray_init_kwargs = config.get("ray_init", {})
        # そのうち runtime_env（ユーザ指定の追加環境変数など）を取り出す。
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        # 既定の runtime_env に、ユーザ指定分を上書きマージする。
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        # マージ後の runtime_env を ray_init_kwargs に差し戻して、最終的な起動引数を作る。
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        # デバッグ用に起動引数を表示
        print(f"ray init kwargs: {ray_init_kwargs}")
        # OmegaConf を素の Python dict に変換して ray.init に渡し、クラスタを起動。
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # 学習本体を持つ Ray アクター TaskRunner をリモート生成する（.remote() はアクター化）。
    runner = TaskRunner.remote()
    # runner.run(config) をリモート実行し、ray.get で完了を待つ（同期実行）。
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
# ↑ この TaskRunner は Ray アクターとして 1CPU で動く。学習全体のオーケストレーション役。
# ヘッドノードに載らないよう注意、というコメント（GPU 資源はワーカ側が別途確保する）。
class TaskRunner:
    def run(self, config):
        # print initial config
        # 設定を見やすく表示するため
        from pprint import pprint

        from omegaconf import OmegaConf

        # HDFS 等のリモートパスをローカルに落とすユーティリティ
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        # ↑ ${...} 参照などを解決した最終 config を表示（実際に使われる値を確認）。
        # config 内の参照を実値に確定させる（以降 in-place で解決済み）。
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        # モデル本体（重み）をローカルへコピー。use_shm=True なら共有メモリ上に置く。
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        from agent_system.environments import make_envs
        # 学習用・検証用の環境を構築。env_name が "multitask" の場合はここで
        # MultiTaskEnvironmentManager（3サブ環境を束ねる）が作られる（単タスクは単一環境）。
        envs, val_envs = make_envs(config)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        # HF のカスタムコード許可フラグ
        trust_remote_code = config.data.get("trust_remote_code", False)
        # テキスト⇔トークンID変換器
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # ↑ マルチモーダル（画像等）用の前処理器。テキストのみのタスクでは None になり得る。
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # vllm early verify
        # 推論エンジンに vLLM を使う場合の事前検証。
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            # LoRA を使う設定なら vLLM 0.7.3 以上が必要（未満なら明示的にエラー）。
            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        # 分散戦略（FSDP か Megatron か）に応じて、使うワーカクラス群を選ぶ。
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            # actor と critic の戦略は揃える
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            # ロールアウトが async モードなら非同期版のワーカを使う（生成と学習を重ねる高速化）。
            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # 同上（戦略統一）
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            # 未知の分散戦略は非対応
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # 役割(Role) → 実行するワーカクラス、の対応表。ActorRollout（生成＋学習）と Critic。
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # GPU 資源プールの識別名
        global_pool_id = "global_pool"
        # 資源プールの仕様: 1ノードあたり n_gpus_per_node GPU を nnodes 台分。
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # 各役割をどの資源プールに載せるか（ここでは全役割を同一プールに同居させる）。
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        # 報酬モデル（学習済みモデルで報酬を出す方式）を使う場合のワーカ登録。
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        # 参照方策（KL 正則化の基準となる固定モデル）が必要なら RefPolicy ワーカを登録。
        # use_kl_in_reward（報酬に KL を混ぜる）か use_kl_loss（損失に KL を足す）のどちらかで必要。
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # 報酬マネージャの種類を config から取得（既定は "episode" = エピソード単位の報酬集計）。
        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == 'episode':
            from agent_system.reward_manager import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        # 学習用の報酬関数。num_examine=0 = ログに詳細サンプルを出さない。
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)

        # Note that we always use function-based RM for validation
        # 検証用の報酬関数。num_examine=1 = 1件だけ中身をログ表示して目視確認できるように。
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        # 資源プール仕様と役割マッピングから、GPU を実際に確保・管理するマネージャを作る。
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # verl 本体では rollout.n>1 が GRPO のグループ数だが、verl+env（エージェント）では
        # rollout.n は必ず 1 に固定し、GRPO のグループは env.rollout.n（=8 など）で実現する、という制約。
        assert config.actor_rollout_ref.rollout.n == 1, "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        from agent_system.multi_turn_rollout import TrajectoryCollector
        # 環境との複数ターン対話（ロールアウト）を回して軌跡(trajectory)を集める役。
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        # 複数サンプルを1バッチに結合する関数
        from verl.utils.dataset.rl_dataset import collate_fn

        # 学習用・検証用データセット（parquet を読み、プロンプトをトークナイズして保持）。
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        # サンプラ生成。task_balance.enable=True なら TaskBalancedSampler、そうでなければ通常の RandomSampler。
        # ★ここが「3タスク同時学習」と「単タスク学習」の分岐点。
        train_sampler = create_rl_sampler(config.data, train_dataset)
        # ここまでに用意した全部品（ワーカ・報酬・環境・データ・サンプラ）を RayPPOTrainer に渡す。
        trainer = RayPPOTrainer(
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
            # DataLoader がバッチ結合に使う関数
            collate_fn=collate_fn,
            # DataLoader が index の順序決めに使うサンプラ
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        # GPU ワーカ群を実際に起動・初期化（モデルロード等）
        trainer.init_workers()
        # 学習ループ本体を開始（ロールアウト→advantage→loss→更新の反復）
        trainer.fit()


# 【日本語補足】parquet ファイル群から RL 用データセットを作る。
# 通常は RLHFDataset を使うが、config.custom_cls でユーザ独自クラスも差し込める。
def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # custom_cls.path が指定されていれば、外部モジュールから独自データセットクラスを動的ロード。
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # 独自クラスは必ず torch の Dataset を継承していること（していなければ型エラー）。
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        # 既定のデータセットクラス
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # データセット本体を生成。ここで parquet 読み込み・（設定に応じ）長すぎるプロンプトの除外などが走る。
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


# 【日本語補足：このクラスの目的】
# 3タスク同時学習で、「毎バッチのタスク構成を必ず均等（例: alfworld15 + search15 + webshop15）に
# する」ためのサンプラ。PyTorch の Sampler と同じく「index を並べて yield するだけ」の薄い部品で、
# 実際にそれを batch_size 個ずつ束ねてバッチ化するのは DataLoader 側の仕事。
# ここが崩れると、あるステップが search だけ・別ステップが alfworld だけ…と偏り、
# 共同学習（joint 最適化）の前提が壊れる。
class TaskBalancedSampler:
    """Yield indices so every dataloader batch has the same task mix."""

    def __init__(self, dataset, task_balance_config, batch_size: int, shuffle: bool, seed: int):
        # 乱数（シャッフル）に使う
        import numpy as np
        # config を素の dict に変換するのに使う
        from omegaconf import OmegaConf

        # データセットに task_name 列が無ければタスク判定できないので即エラー。
        # （prepare_sdar_multitask.py が全行に task_name を付与している前提）
        if not hasattr(dataset, "dataframe") or "task_name" not in dataset.dataframe.column_names:
            raise ValueError("TaskBalancedSampler requires a dataset column named 'task_name'.")

        # task_balance の設定を素の Python dict にする（OmegaConf のままだと扱いにくいので）。
        cfg = OmegaConf.to_container(task_balance_config, resolve=True) if OmegaConf.is_config(task_balance_config) else dict(task_balance_config)
        # 対象タスク一覧（既定は3タスク）。この順序が非interleave時のバッチ内並び順になる。
        self.tasks = list(cfg.get("tasks", ["alfworld", "search", "webshop"]))
        # 1タスクあたり1バッチに何件入れるか（例: 15）。
        self.per_task_batch_size = int(cfg.get("per_task_batch_size", 0))
        if self.per_task_batch_size <= 0:
            raise ValueError("data.task_balance.per_task_batch_size must be a positive integer.")
        # 総バッチ数（=総学習ステップ数）。明示指定があればそれを使う（後で確定）。
        requested_num_batches = cfg.get("num_batches", None)

        # 期待されるバッチサイズ = 1タスク件数 × タスク数（例: 15 × 3 = 45）。
        expected_batch_size = self.per_task_batch_size * len(self.tasks)
        # DataLoader の batch_size と一致していないと均等分割が成立しないので、ここで厳格に検証。
        # （train_batch_size を変えて per_task_batch_size を直し忘れる、等の設定ミスを早期に弾く）
        if int(batch_size) != expected_batch_size:
            raise ValueError(
                f"Balanced task batch size mismatch: got batch_size={batch_size}, "
                f"expected {expected_batch_size} ({self.per_task_batch_size} * {len(self.tasks)} tasks)."
            )

        # データセット全行の task_name を取り出す（行 index 順）。
        task_names = list(dataset.dataframe["task_name"])
        # タスク別のインデックス台帳を作る:
        # {alfworld: [0..14], search: [15..4514], webshop: [4515..4529]} のような形。
        self.task_to_indices = {
            task: [idx for idx, task_name in enumerate(task_names) if task_name == task]
            for task in self.tasks
        }
        # どのタスクも「最低1バッチ分（per_task_batch_size 件）」は無いと均等バッチを作れない。
        missing = [task for task, indices in self.task_to_indices.items() if len(indices) < self.per_task_batch_size]
        if missing:
            raise ValueError(f"Not enough samples to build one balanced batch for tasks: {missing}")

        # データ量だけから作れる最大バッチ数 = 各タスクの (件数 // per_task) の最小値。
        # 例: min(15//15, 4500//15, 15//15) = min(1, 300, 1) = 1。
        available_batches = min(len(indices) // self.per_task_batch_size for indices in self.task_to_indices.values())
        # num_batches: 明示指定(例:300)があればそれを採用し、無ければ available(1) を使う。
        # ★ここが肝: 15件しかないダミータスクでも num_batches=300 を明示すれば、
        # 後述の _indices_for_required_size がサイクル補充して 300 ステップ分に引き伸ばす。
        self.num_batches = int(requested_num_batches) if requested_num_batches is not None else available_batches
        if self.num_batches <= 0:
            raise ValueError("data.task_balance.num_batches must be a positive integer when configured.")
        # タスク内の並びをシャッフルするか
        self.shuffle = bool(shuffle)
        # 乱数シード（再現性のため）
        self.seed = int(seed)
        # __iter__ が呼ばれるたびに増える（epochごとに並びを変える）
        self._epoch = 0
        # numpy 参照を保持
        self._np = np
        # Fix1: interleave tasks within each batch (round-robin) so the
        # data-parallel split of the generation batch is task-balanced.
        # 環境変数 TASK_BALANCE_INTERLEAVE=1 のとき、バッチ内をラウンドロビン配置にする。
        # そうするとデータ並列で分割した各GPUチャンクにも3タスクが均等に入る。
        self._interleave = os.environ.get("TASK_BALANCE_INTERLEAVE", "0").strip().lower() in ("1", "true", "yes", "on")

    def _indices_for_required_size(self, indices, required: int, rng):
        # 与えられた indices（1タスク分の index リスト）を、required 件になるまで
        # 「1周ごとに（shuffle 時は並べ替えつつ）繰り返し連結」して、最後に required 件へ切り詰める。
        # - search(4500件) は required(4500) と同数なので 1周＝重複なし。
        # - alfworld/webshop(15件) は 300周して各行が300回登場（=毎バッチ全15行）。
        # 破壊的操作を避けるためコピー
        indices = list(indices)
        output = []
        # 必要数に達するまで周回
        while len(output) < required:
            # この周回分（1周＝元リスト全部）
            epoch_indices = list(indices)
            if self.shuffle:
                # この周回だけの並べ替え（周回ごとに順序が変わる）
                rng.shuffle(epoch_indices)
            # 連結
            output.extend(epoch_indices)
        # required 件ちょうどに切り詰めて返す
        return output[:required]

    def __iter__(self):
        # DataLoader が1エポック分イテレートするたびに呼ばれる。ここで index 列を生成して yield する。
        # seed+epoch で毎エポック別の乱数系列
        rng = self._np.random.RandomState(self.seed + self._epoch)
        # 次に __iter__ が呼ばれたら別のシャッフルになるよう epoch を進める
        self._epoch += 1

        task_indices = {}
        # 各タスクに必要な総index数（例: 300×15=4500）
        required = self.num_batches * self.per_task_batch_size
        # タスクごとに「required 件までサイクル補充した index 列」を用意する。
        for task, indices in self.task_to_indices.items():
            task_indices[task] = self._indices_for_required_size(indices, required, rng)

        # バッチ0 … バッチ(num_batches-1) まで、各バッチ分の index を順に吐き出す。
        for batch_idx in range(self.num_batches):
            # このバッチが使う区間の開始位置
            start = batch_idx * self.per_task_batch_size
            # 終了位置（開始+per_task）
            end = start + self.per_task_batch_size
            if self._interleave:
                # Round-robin task layout: alf0, search0, webshop0, alf1, ...
                # -> each DP chunk gets an equal task mix. Reordering only; the
                # sample set and the GRPO groups (formed later by uid) are
                # unchanged, so training is unaffected.
                # ラウンドロビン: i を 0..per_task-1 と進め、各 i で全タスクから1件ずつ吐く。
                # 結果: alf0, search0, web0, alf1, search1, web1, ... という交互並び。
                # 並べ替えるだけで、サンプル集合も（後で uid 単位に作られる）GRPO グループも不変。
                for i in range(self.per_task_batch_size):
                    for task in self.tasks:
                        yield task_indices[task][start + i]
            else:
                # 非interleave（既定）: [alf を15件] → [search を15件] → [web を15件] の塊で吐く。
                # どちらのモードでも「連続 per_task×タスク数(=45) 件 = 各タスク15件ずつ」は保証される。
                for task in self.tasks:
                    yield from task_indices[task][start:end]

    def __len__(self):
        # このサンプラが1エポックで吐く index の総数（= 総バッチ数 × バッチサイズ）。
        # 例: 300 × 15 × 3 = 13500。DataLoader が長さ計算に使う。
        return self.num_batches * self.per_task_batch_size * len(self.tasks)


# 【日本語補足】data.task_balance.enable の有無で、
# 「3タスク均等サンプラ」か「通常サンプラ」かを選ぶ分岐点。
def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # task_balance セクションを取得（無ければ空 dict）。
    task_balance_config = data_config.get("task_balance", {})
    # enable=True のとき、3タスク同時学習用の TaskBalancedSampler を使う。
    if task_balance_config and task_balance_config.get("enable", False):
        # DataLoader に渡すバッチサイズ = gen_batch_size（無ければ train_batch_size）。
        # これが per_task_batch_size × タスク数 と一致している必要がある（Sampler 側で検証）。
        batch_size = data_config.get("gen_batch_size", data_config.train_batch_size)
        sampler = TaskBalancedSampler(
            dataset=dataset,
            task_balance_config=task_balance_config,
            batch_size=batch_size,
            # タスク内シャッフルの有無
            shuffle=data_config.shuffle,
            # 再現性シード（multitask スクリプトでは 1）
            seed=data_config.get("seed", 1),
        )
        # 起動ログに、確定したタスク一覧・1タスク件数・総バッチ数を出す。
        print(
            "Using TaskBalancedSampler: "
            f"tasks={sampler.tasks}, per_task_batch_size={sampler.per_task_batch_size}, "
            f"num_batches={sampler.num_batches}"
        )
        return sampler

    # use sampler for better ckpt resume
    # 単タスク学習（task_balance 無効）はここ。チェックポイント再開の再現性のためサンプラを明示指定。
    if data_config.shuffle:
        # シャッフルする場合: シード固定の Generator を持つ RandomSampler。
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # シャッフルしない場合: 先頭から順に返す SequentialSampler。
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    # スクリプトとして直接起動されたら Hydra の main を実行
    main()
