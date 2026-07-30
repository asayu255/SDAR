# 【このファイルの役割（日本語補足）】
# - SDAR / SkillSD の「学習ループ本体」。RLSDRayTrainer を継承している。
# - RLSD が advantage をトークン単位で教師で書き換えるのに対し、SkillSD は
# GRPO の advantage はそのまま残し、教師 log-prob を actor に渡して
# 「蒸留損失（SDL/SDAR loss）」を追加項として足す方式。
# - main_sdar.py はこのクラスをインスタンス化して fit() を呼ぶ（=SDAR 学習の実体）。
# - 3タスク同時学習で効くのは主に2箇所:
# (A) _task_kl_loss_coef_tensor : サンプルごとに task 別 KL 係数を割り当てる
# (B) fit() 内の task 別スキルによる教師フォワード / task 別無効行動ペナルティ
"""
SkillSD (Skill-based Self-Distillation) Trainer.

Extends RLSDRayTrainer. Unlike RLSD which replaces advantages with token-level
teacher-weighted advantages, SkillSD keeps the original GRPO advantages and
adds an auxiliary SDL loss computed inside the actor's update_policy().
"""

# 検証メトリクスなどを見やすく表示
from pprint import pprint

# SDAR デバッグ出力（per-token gap）を JSONL で保存するのに使う
import json
# 環境変数（ENV_RESET_PREFETCH, SAVE_SDAR_DEBUG）の読み取り
import os

import numpy as np
import ray
import torch
# 学習進捗バー
from tqdm import tqdm

# verl のバッチデータ構造（tensor + 非tensor をまとめて持つ）
from verl import DataProto
# 損失をマスク付きで集約するユーティリティ
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    # 全 trainer の基底（PPO ループの共通処理）
    RayPPOTrainer,
    # with 文で区間の実行時間を測るコンテキストマネージャ
    _timer,
    # 無効行動にペナルティ報酬を与える関数（task別係数対応）
    apply_invalid_action_penalty,
    # 報酬に KL ペナルティを混ぜる関数（use_kl_in_reward 用）
    apply_kl_penalty,
    # GRPO 等の advantage を計算する関数
    compute_advantage,
    # 応答トークン部分だけを 1 にするマスクを作る
    compute_response_mask,
)
# 報酬計算（同期/非同期）
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
# スキル（特権情報）を供給するクラス
from verl.trainer.ppo.rlsd_utils import SkillProvider
# 継承元と教師バッチ構築関数
from verl.trainer.ppo.rlsd_ray_trainer import RLSDRayTrainer, build_teacher_batch
# ワーカから返るメトリクスを集約
from verl.utils.metric import reduce_metrics
# マスクを考慮した平均
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    # バッチのデータ統計（報酬・長さ等）
    compute_data_metrics,
    compute_data_metrics_by_task,
    # スループット（tokens/sec 等）
    compute_throughout_metrics,
    # 各区間の所要時間
    compute_timing_metrics,
    normalize_task_name,
    task_row_indices,
)

# adjust_batch: ロールアウト結果のバッチ整形。
# compute_log_prob_with_prefetch: ロールアウト中に先読みした log-prob があれば再利用する高速化版。
from agent_system.multi_turn_rollout import adjust_batch, compute_log_prob_with_prefetch

# Overlap envs.reset() for the next rollout with this step's GPU training phases.
# The reset is pure CPU / subprocess / HTTP work and the env managers are idle
# between rollouts; the reset still runs exactly once per rollout and in the same
# order, so stateful env schedules (alfworld's game-file iterator) are unchanged.
# Opt-in; see TrajectoryCollector.prefetch_env_reset.
# 【日本語補足】ENV_RESET_PREFETCH=1 のとき、次ステップ用の envs.reset() を
# バックグラウンドで先行実行し、今ステップの GPU 学習と時間的に重ねる高速化。
# reset は CPU/サブプロセス/HTTP 主体で、ロールアウト間は env が空いているため重ねられる。
# reset の回数・順序は不変なので、alfworld のゲームファイル順など状態的スケジュールは変わらない。
_ENV_RESET_PREFETCH = os.environ.get("ENV_RESET_PREFETCH", "0").strip().lower() in ("1", "true", "yes", "on")


class SkillSDRayTrainer(RLSDRayTrainer):
    """
    SkillSD trainer that extends RLSDRayTrainer.

    Keeps the original GRPO advantages unchanged and passes teacher_log_probs
    to the actor, where the SDL loss is computed inside update_policy().
    """

    def __init__(self, *args, skill_provider: SkillProvider = None, **kwargs):
        # 基底（RLSDRayTrainer→RayPPOTrainer）の初期化に委譲。skill_provider も引き継ぐ。
        super().__init__(*args, skill_provider=skill_provider, **kwargs)
        # SkillSD 固有ハイパラを config から取得。
        skillsd_cfg = self.config.algorithm.get("skillsd", {})
        # 蒸留損失の重み（初期値）
        self.sdl_lambda = skillsd_cfg.get("sdl_lambda", 0.1)
        # 重みを線形減衰させるステップ数（-1=減衰なし）
        self.sdl_warmdown_steps = skillsd_cfg.get("warmdown_steps", -1)

    def _get_sdl_lambda(self, step: int) -> float:
        # 現ステップにおける蒸留損失の重み λ を返す。
        if self.sdl_warmdown_steps <= 0:
            # 減衰無効なら常に一定
            return self.sdl_lambda
        if step >= self.sdl_warmdown_steps:
            # 減衰期間を過ぎたら 0（蒸留オフ）
            return 0.0
        # それ以外は線形に減衰
        return self.sdl_lambda * (1.0 - step / self.sdl_warmdown_steps)

    @staticmethod
    def _normalize_task_name(task_name):
        return normalize_task_name(task_name)

    # どれにも当てはまらなければそのまま返す
    def _task_kl_loss_coef_tensor(self, batch: DataProto):
        # ★3タスク同時学習の要(A): サンプルごとに task 別の KL 係数を並べたテンソルを作る。
        # 単タスクなら kl_loss_coef はスカラ1値だが、同時学習では
        # {alfworld:0.01, search:0.001, webshop:0.01} のように task で変えたい。
        from omegaconf import OmegaConf

        # config に kl_loss_coef_by_task が無ければ None を返す（=従来のスカラ運用）。
        coef_by_task = self.config.actor_rollout_ref.actor.get("kl_loss_coef_by_task", None)
        if not coef_by_task:
            # 生の task_name 文字列を、正規の3種（alfworld/webshop/search）に正規化する。
            # 例: "alfworld/AlfredTWEnv" や "Webshop" など表記ゆれを吸収して判定に使う。
            return None
        if OmegaConf.is_config(coef_by_task):
            # 素の dict に変換
            coef_by_task = OmegaConf.to_container(coef_by_task, resolve=True)

        # 未知 task 用のフォールバック係数（従来のスカラ kl_loss_coef）。
        fallback = float(self.config.actor_rollout_ref.actor.kl_loss_coef)
        # 各サンプルの task_name を取得。まず non_tensor_batch から直接。
        task_names = batch.non_tensor_batch.get("task_name", None)
        if task_names is None:
            # 無ければ env_kwargs の中の task_name を拾う（フォールバック経路）。
            env_kwargs = batch.non_tensor_batch.get("env_kwargs", None)
            if env_kwargs is not None:
                task_names = [
                    item.get("task_name") if isinstance(item, dict) else None
                    for item in env_kwargs
                ]

        # それでも task_name が取れないなら、全サンプル一律 fallback 係数のテンソルを返す。
        if task_names is None:
            return torch.full((len(batch),), fallback, dtype=torch.float32)

        # サンプルごとに task を正規化し、対応する係数を並べる。
        coefs = []
        for task_name in task_names:
            task = self._normalize_task_name(task_name)
            coefs.append(float(coef_by_task.get(task, fallback)))
        # 形状 (バッチサイズ,) の係数テンソル。後で dp_actor がサンプル別重み付けに使う。
        return torch.tensor(coefs, dtype=torch.float32)

    # 【日本語補足】1ステップ = 「ロールアウト → 報酬 → 旧log_prob → 教師フォワード →
    # 参照log_prob → advantage → (critic) → actor更新 → 検証/保存」の流れ。
    def fit(self):
        """
        The training loop of SkillSD. Identical to RLSD except:
        - Advantages are NOT replaced with token-level advantages
        - teacher_log_probs are passed through to the actor for SDL loss
        """
        from omegaconf import OmegaConf
        # WandB / console へのログ出力ラッパ
        from verl.utils.tracking import Tracking

        # ロガー初期化（プロジェクト名・実験名・出力先・全 config を記録）。
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # 現在の学習ステップ数
        self.global_steps = 0
        # 途中再開用: チェックポイントがあれば復元
        self._load_checkpoint()
        self._fast_forward_env_schedules()

        # 学習前に検証するか（val_before_train）。multitask スクリプトでは False。
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                # 検証だけして終了するモード
                return

        # 進捗バー。総ステップ数まで進める。
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training")
        # ステップ番号は 1 始まり
        self.global_steps += 1
        last_val_metrics = None

        # エポックループ（total_epochs 回）。中の while が dataloader を1周する。
        for epoch in range(self.config.trainer.total_epochs):
            # DataLoader のイテレータを明示的に作る
            batch_iter = iter(self.train_dataloader)
            peeked_batch_dict = None
            while True:
                # 先読み済みバッチがあればそれを今ステップで使う。無ければ次を取り出す。
                if peeked_batch_dict is not None:
                    batch_dict = peeked_batch_dict
                    # 先読み（prefetch）した次バッチの一時置き場
                    peeked_batch_dict = None
                else:
                    batch_dict = next(batch_iter, None)
                    if batch_dict is None:
                        # このエポックのバッチを使い切ったら while を抜ける
                        break
                # Reset the pre-peek dataloader snapshot each step; it is set
                # again below if this step peeks ahead (see _save_checkpoint).
                # 先読みした場合の dataloader 位置スナップショットを毎ステップ初期化。
                self._pre_peek_dataloader_state = None
                # このステップのメトリクス
                metrics = {}
                # このステップの各区間時間
                timing_raw = {}
                # dict のバッチを verl の DataProto 構造へ変換。
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # 生成(gen)に渡す用に、テンソル系キーと非テンソル系キーを batch から取り出す(pop)。
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                # マルチモーダル等、存在するものだけ pop 対象に追加。
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                # ★同時学習: task_name もロールアウト用バッチに引き渡す（env ルーティングに必要）。
                if "task_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("task_name")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # このステップが最終ステップか（検証/保存の判定に使う）。
                is_last_step = self.global_steps >= self.total_training_steps

                # ステップ全体の時間計測
                with _timer("step", timing_raw):
                    # ロールアウト（環境との複数ターン対話）
                    with _timer("gen", timing_raw):
                        # multi_turn_loop: gen_batch のプロンプトから env と対話し、軌跡を生成。
                        # 同時学習では envs が MultiTaskEnvironmentManager で、task_name により
                        # 各サンプルが該当サブ環境へルーティングされる。
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    # The train envs are idle from here until the next rollout;
                    # kick off their reset for the next step in a background
                    # thread so it overlaps the GPU training phases below.
                    # 【高速化】ロールアウト後、env は次のロールアウトまで暇なので、
                    # 次ステップ用の reset をここでバックグラウンド起動して以降の GPU 処理と重ねる。
                    if (
                        _ENV_RESET_PREFETCH
                        # 最終ステップでは先読み不要
                        and not is_last_step
                        # グループ絞り込み時は先読みしない
                        and not self.config.algorithm.filter_groups.enable
                    ):
                        # Snapshot the dataloader state before peeking: the peeked
                        # batch is trained on the NEXT step, so a checkpoint saved
                        # this step must record the pre-peek position or a resumed
                        # run would skip that batch (see _save_checkpoint).
                        # 先読みで dataloader を1つ進めてしまうので、今ステップで保存する
                        # チェックポイントには「先読み前の位置」を記録する必要がある。
                        # そのため現在位置をスナップショットしておく（_save_checkpoint が使う）。
                        if hasattr(self.train_dataloader, "state_dict"):
                            self._pre_peek_dataloader_state = self.train_dataloader.state_dict()
                        # 次バッチを実際に取り出しておく（今ステップでは学習に使わない）。
                        peeked_batch_dict = next(batch_iter, None)
                        if peeked_batch_dict is not None and "env_kwargs" in peeked_batch_dict:
                            # Same repeat the next multi_turn_loop applies to its
                            # gen_batch (repeat(n, interleave=True) on non-tensors
                            # is an element-wise np.repeat).
                            # 次ロールアウトが gen_batch に施すのと同じ「n回複製」を env_kwargs に適用し、
                            # その env_kwargs で次ステップ分の env を先行 reset する。
                            next_env_kwargs = np.repeat(
                                peeked_batch_dict["env_kwargs"], self.config.env.rollout.n
                            )
                            self.traj_collector.prefetch_env_reset(self.envs, next_env_kwargs)

                    # 元 batch はもう不要
                    del batch
                    # 以降はロールアウト結果を batch とする
                    batch = gen_batch_output

                    # バッチ整形
                    batch = adjust_batch(self.config, batch)
                    # 応答部分マスクを付与
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # データ並列で各GPUの系列長を均すためのバランス処理（有効時）。
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # 全トークン数（スループット計算・ログ用）を記録。
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # 報酬計算
                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            # 報酬モデルを使う場合はスコアを計算して結合。
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            # 非同期報酬: future を受け取り、あとで ray.get する（GPU処理と重ねる）。
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            # 同期報酬: ここで即計算。
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # 生成方策（student）の log-prob 計算
                    with _timer("old_log_prob", timing_raw):
                        # Reuse any per-row log probs prefetched during the rollout
                        # (ROLLOUT_PREFETCH_LOGPROB); computes everything normally
                        # when nothing was prefetched.
                        # ロールアウト中に先読みした log-prob があれば再利用、無ければ通常計算。
                        old_log_prob = compute_log_prob_with_prefetch(
                            self.actor_rollout_wg,
                            batch,
                            self.traj_collector.take_prefetched_log_probs(),
                            temperature=self.config.actor_rollout_ref.rollout.temperature,
                        )
                        # トークン別エントロピー
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        old_log_prob_metrics = self._entropy_loss_metrics(batch, entropys, response_masks, loss_agg_mode)
                        # エントロピー損失（モニタ用に集約）。
                        metrics.update(old_log_prob_metrics)
                        # エントロピーは記録済みなので落とす
                        old_log_prob.batch.pop("entropys")
                        # old_log_probs を batch に結合
                        batch = batch.union(old_log_prob)

                    # ---- SkillSD: Teacher forward pass (same as RLSD) ----
                    with _timer("teacher_forward", timing_raw):
                        # ★同時学習の要(B): 教師フォワード。
                        # 同一モデルにスキル付き入力を与えて teacher_log_probs を計算する。
                        # 内部(build_teacher_batch)で各サンプルの task_name に応じた
                        # 正しいドメインのスキルが prepend される（skills_dirs によるルーティング）。
                        teacher_log_probs = self._compute_teacher_log_probs(batch)
                        batch.batch["teacher_log_probs"] = teacher_log_probs

                    # 参照方策の log-prob（KL 正則化用）
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                # 参照方策が別ワーカにある場合。
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                # 参照方策を actor ワーカ内で計算する場合。
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # critic（価値関数）を使う場合のみ
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # advantage 計算
                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            # 非同期報酬なら、ここで結果を回収。
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        # トークン単位スコア
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            # 追加報酬情報を非テンソルバッチに反映。
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # 無効行動ペナルティ（フォーマット違反等）を適用。
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                # ★同時学習: task 別のペナルティ係数（{alfworld:0.1,search:0.01,webshop:0.1}）。
                                invalid_action_penalty_coef_by_task=self.config.actor_rollout_ref.actor.get(
                                    "invalid_action_penalty_coef_by_task", None
                                ),
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            # 報酬に KL ペナルティを混ぜる方式（SDAR スクリプトは False）。
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            # 混ぜない場合はスコアをそのままトークン報酬とする。
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # GRPO で advantage を標準偏差で正規化するか。
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        # advantage 計算本体。GRPO は uid（プロンプト）単位のグループで正規化する。
                        # → グループは必ず単一 task に閉じるので、同時学習でも task 間で advantage は混ざらない。
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                        # ---- SkillSD: Do NOT replace advantages ----
                        # Advantages stay as standard GRPO sequence-level advantages.
                        # The SDL loss is computed inside dp_actor.update_policy().

                        # Log teacher-student gap metrics
                        # ↑ RLSD と違い advantage は書き換えない。蒸留損失は actor 側の update で足す。
                        # 教師と生徒の log-prob 差（gap）をモニタリング用に集計。
                        response_mask = batch.batch["response_mask"]
                        student_log_probs = batch.batch["old_log_probs"]
                        teacher_lp = batch.batch["teacher_log_probs"]
                        # 応答部分のみの gap
                        delta_t = (teacher_lp - student_log_probs) * response_mask
                        current_sdl_lambda = self._get_sdl_lambda(self.global_steps)
                        metrics["skillsd/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                        metrics["skillsd/teacher_student_gap_std"] = masked_mean(delta_t ** 2, response_mask).sqrt().item()
                        metrics["skillsd/sdl_lambda"] = current_sdl_lambda
                        for task, rows in task_row_indices(batch).items():
                            task_rows = torch.from_numpy(rows)
                            task_delta_t, task_mask = delta_t[task_rows], response_mask[task_rows]
                            metrics[f"skillsd/teacher_student_gap_mean/{task}"] = masked_mean(task_delta_t, task_mask).item()
                            metrics[f"skillsd/teacher_student_gap_std/{task}"] = masked_mean(task_delta_t ** 2, task_mask).sqrt().item()

                        # Save per-token gap data if SAVE_SDAR_DEBUG=1, at test_freq interval
                        # デバッグ用: SAVE_SDAR_DEBUG=1 のとき、test_freq 間隔で per-token の gap を JSONL 保存。
                        if os.environ.get("SAVE_SDAR_DEBUG", "0") == "1" and \
                                self.config.trainer.test_freq > 0 and \
                                self.global_steps % self.config.trainer.test_freq == 0:
                            save_dir = os.environ.get(
                                "SAVE_SDAR_DEBUG_DIR",
                                "outputs/sdar_debug"
                            )
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"step_{self.global_steps}.jsonl")

                            # バッチサイズ
                            bs = response_mask.shape[0]
                            # 各種メタ情報（無ければゼロ埋め）を取得。
                            turn_steps = batch.non_tensor_batch.get("turn_step", np.zeros(bs, dtype=object))
                            traj_uids = batch.non_tensor_batch.get("traj_uid", np.array([""] * bs, dtype=object))
                            episode_rewards_arr = batch.non_tensor_batch.get("episode_rewards", np.zeros(bs, dtype=object))
                            episode_lengths_arr = batch.non_tensor_batch.get("episode_lengths", np.zeros(bs, dtype=object))
                            response_ids = batch.batch["responses"]

                            with open(save_path, "w") as f:
                                # サンプルごとに1行 JSON を書く
                                for i in range(bs):
                                    mask_i = response_mask[i].bool()
                                    valid_count = mask_i.sum().item()
                                    if valid_count == 0:
                                        # 有効トークンが無いサンプルはスキップ
                                        continue
                                    token_ids_i = response_ids[i][mask_i].cpu().tolist()
                                    # トークン文字列
                                    tokens_i = [self.tokenizer.decode([tid]) for tid in token_ids_i]
                                    gaps_i = delta_t[i][mask_i].cpu().tolist()
                                    # 教師 log-prob
                                    teacher_lps_i = teacher_lp[i][mask_i].cpu().tolist()
                                    # 生徒 log-prob
                                    student_lps_i = student_log_probs[i][mask_i].cpu().tolist()

                                    record = {
                                        "global_step": self.global_steps,
                                        "sample_idx": i,
                                        "turn_step": int(turn_steps[i]),
                                        "traj_uid": str(traj_uids[i]),
                                        "episode_reward": float(episode_rewards_arr[i]),
                                        "episode_length": float(episode_lengths_arr[i]),
                                        "tokens": tokens_i,
                                        "token_ids": token_ids_i,
                                        "gaps": gaps_i,
                                        "teacher_log_probs": teacher_lps_i,
                                        "student_log_probs": student_lps_i,
                                    }
                                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            print(f"[SDAR Debug] Saved per-token gap data to {save_path} ({bs} samples)")

                    # tag rows with their task so the actor can split its metrics
                    self._attach_task_ids(batch)

                    # critic を使う場合は critic を更新
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # critic ウォームアップ期間を過ぎたら actor（方策）を更新。
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # ★同時学習: サンプル別の KL 係数テンソルを作り、batch に載せる。
                            # dp_actor 側でこの係数を使ってサンプルごとに KL 損失を重み付けする。
                            kl_loss_coef = self._task_kl_loss_coef_tensor(batch)
                            if kl_loss_coef is not None:
                                batch.batch["kl_loss_coef"] = kl_loss_coef
                            # actor 更新: ここで GRPO 損失 + SDAR 蒸留損失 + KL 損失を合算して
                            # 1回のバックワード＆オプティマイザステップ（=3タスク共同更新）が走る。
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # ロールアウト内容をディスクにダンプする設定なら書き出す（デバッグ/分析用）。
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # 検証(validation)タイミング判定: 最終ステップ or test_freq 間隔。
                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            # 同時学習では test をタスク順ソートしてあるため、各 val バッチが単一 task になる。
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # チェックポイント保存タイミング判定: 最終ステップ or save_freq 間隔。
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            # 先読みした場合は pre-peek 位置で保存（RLSD側でオーバライド）
                            self._save_checkpoint()

                # ステップ番号・エポックをメトリクスに追加。
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                # データ統計・時間・スループットを集計してログ。
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_data_metrics_by_task(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # WandB/console へ出力
                logger.log(data=metrics, step=self.global_steps)

                # 進捗バーを1進める
                progress_bar.update(1)
                # ステップ番号を進める
                self.global_steps += 1
                # 最終ステップに達したら学習終了
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
