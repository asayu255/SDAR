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

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.distributed
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from tensordict import TensorDict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.distributed.device_mesh import init_device_mesh

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl.utils.distributed import initialize_global_process_group


# [EXPLAIN] `test_trainer_forward_consistency` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def test_trainer_forward_consistency(trainer: FSDPSFTTrainer, total_steps: int = 4):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Test consistency between original forward pass and SP+rmpad forward passes.

    Args:
        trainer: The FSDPSFTTrainer instance to test
        total_steps: Number of steps to test (default: 4)
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if trainer.device_mesh.get_rank() == 0:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("\nStarting debug comparison between original and SP+rmpad forward passes...")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Sequence parallel size: {trainer.config.ulysses_sequence_parallel_size}")
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print(f"Remove padding: {trainer.use_remove_padding}\n")

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    steps_remaining = total_steps

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for epoch in range(1):  # Just one epoch for testing
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        trainer.train_sampler.set_epoch(epoch=epoch)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for data in trainer.train_dataloader:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            data = TensorDict(data, batch_size=trainer.config.data.train_batch_size).cuda()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            trainer.fsdp_model.train()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            micro_batches = data.split(trainer.config.data.micro_batch_size_per_gpu)

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for idx, micro_batch in enumerate(micro_batches):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if trainer.device_mesh.get_rank() == 0:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"\nProcessing micro batch {idx + 1}/{len(micro_batches)}")

                # Compute losses using both methods
                # Disable SP and rmpad
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                trainer.use_remove_padding = False
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                old_sp = trainer.config.ulysses_sequence_parallel_size
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                trainer.config.ulysses_sequence_parallel_size = 1
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss_ref = trainer._compute_loss_and_backward(micro_batch.copy(), do_backward=False)

                # Do SP and rmpad
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                trainer.config.ulysses_sequence_parallel_size = old_sp
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                trainer.use_remove_padding = True
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss_sp = trainer._compute_loss_and_backward(micro_batch.copy(), do_backward=False)

                # Collect losses across all ranks
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss_ref_all = loss_ref.clone()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                loss_sp_all = loss_sp.clone()
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                torch.distributed.all_reduce(loss_ref_all, op=torch.distributed.ReduceOp.AVG)
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                torch.distributed.all_reduce(loss_sp_all, op=torch.distributed.ReduceOp.AVG)

                # Calculate relative difference of averaged losses
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                rel_diff = torch.abs(loss_ref_all - loss_sp_all) / (torch.abs(loss_ref_all) + 1e-8)

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if trainer.device_mesh.get_rank() == 0:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print("\nComparison Results (Averaged across ranks):")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"Reference Loss: {loss_ref_all.item():.6f}")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"SP+rmpad Loss: {loss_sp_all.item():.6f}")
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print(f"Relative Difference: {rel_diff.item():.6f}")

                    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
                    assert rel_diff.item() < 1e-2, "Significant difference detected between averaged losses!"
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    print("Loss difference is within the acceptable range.")

                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                steps_remaining -= 1
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if steps_remaining == 0:
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    break
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if steps_remaining == 0:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        break

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if trainer.device_mesh.get_rank() == 0:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        print("\nDebug comparison completed successfully.")


# [EXPLAIN] `create_trainer` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def create_trainer(config):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Create and initialize a trainer instance with the given config.

    Args:
        config: Configuration object with training parameters

    Returns:
        FSDPSFTTrainer: Initialized trainer instance
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_rank, rank, world_size = initialize_global_process_group()

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dp_size = world_size // config.ulysses_sequence_parallel_size
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ulysses_device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))

    # build tokenizer and datasets first
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.trainer.fsdp_sft_trainer import create_sft_dataset
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils import hf_tokenizer
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from verl.utils.fs import copy_to_local

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return FSDPSFTTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)


# [EXPLAIN] `main` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def main(config):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Main function to run trainer tests.

    Args:
        config: Configuration object with training parameters
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    trainer = create_trainer(config)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    test_trainer_forward_consistency(trainer)


# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import hydra
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from omegaconf import DictConfig

    # [EXPLAIN] 直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。
    @hydra.main(config_path="../../../verl/trainer/config", config_name="sft_trainer")
    # [EXPLAIN] `hydra_entry` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def hydra_entry(cfg: DictConfig) -> None:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        main(cfg)

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    hydra_entry()
