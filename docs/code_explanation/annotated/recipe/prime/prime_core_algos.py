# Copyright 2024 PRIME team and/or its affiliates
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
import verl
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import verl.utils.torch_functional as verl_F


# [EXPLAIN] `compute_rloo_advantage_return` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_rloo_advantage_return(data: verl.DataProto, response_mask: torch.Tensor, n_samples, config):
    # calculate rloo reward on different reward sources, and sum again
    # [EXPLAIN] `masked_rloo` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def masked_rloo(reward_tensor_original, mask_tensor):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        reward_tensor = reward_tensor_original.clone()
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        reward_tensor[~mask_tensor] = 0
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for start_pos in range(0, reward_tensor.shape[0], n_samples):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cur_rewards_mean = torch.cat(
                [reward_tensor[pos : pos + 1][mask_tensor[pos : pos + 1]].mean(dim=0, keepdim=True) for pos in range(start_pos, start_pos + n_samples)],
                dim=0,
            )
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cur_rewards_sum = cur_rewards_mean.sum()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cur_reward_baseline = cur_rewards_sum / (n_samples - 1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward_tensor[start_pos : start_pos + n_samples][mask_tensor[start_pos : start_pos + n_samples]] = reward_tensor[start_pos : start_pos + n_samples][mask_tensor[start_pos : start_pos + n_samples]] * (n_samples / (n_samples - 1)) - cur_reward_baseline

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return reward_tensor

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    reward_tensors = []

    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.no_grad():
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "rm_scores" in data.batch.keys() and config.algorithm.reward_dpo_coef != 0.0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward_tensor = data.batch["rm_scores"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward_mask = response_mask.bool()

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            reward_tensors.append(masked_rloo(reward_tensor, reward_mask) * config.algorithm.reward_dpo_coef)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if "acc" in data.batch.keys() and config.algorithm.reward_gt_coef != 0.0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            reward_mask = torch.zeros_like(response_mask, dtype=torch.bool)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prompt_ids = data.batch["prompts"]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            prompt_length = prompt_ids.shape[-1]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(-1)

            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            reward_mask[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1,
            ] = True
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            reward_tensor[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1,
            ] = data.batch["acc"]

            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            reward_tensors.append(masked_rloo(reward_tensor, reward_mask) * config.algorithm.reward_gt_coef)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        final_reward_tensor = sum(reward_tensors)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        returns = (final_reward_tensor * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        advantages = returns.clone()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        advantages = verl_F.masked_whiten(advantages, response_mask)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return advantages, returns


# [EXPLAIN] `compute_ce_dpo_loss_rm` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cur_scores = ((token_level_scores * response_mask).sum(dim=1) * beta).sigmoid()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cur_dpo_loss = torch.nn.functional.binary_cross_entropy(cur_scores, acc)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return cur_dpo_loss


# [EXPLAIN] `compute_detach_dpo_loss_rm` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_detach_dpo_loss_rm(token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"):
    # we always assume that the BoN size equals n_samples
    # mode1: use acc as rm
    # mode2: use Q as rm
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    cur_Q = (token_level_scores * response_mask).sum(dim=1) * beta
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    other_Q = torch.zeros_like(cur_Q)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i in range(token_level_scores.shape[0]):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        Q_chosen = Q_bc[i][acc_bc[i] < acc[i]] if acc[i] > 0 else Q_bc[i][acc_bc[i] > acc[i]]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(Q_chosen) > 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            other_Q[i] = Q_chosen.mean() * beta
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            other_Q[i] = 0
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dpo_loss = -torch.log(torch.sigmoid((cur_Q - other_Q) * ((acc > 0).float() * 2 - 1)))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if bon_mode == "none":
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dpo_loss = dpo_loss.mean()
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        weight = torch.zeros_like(dpo_loss)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_samples = acc_bc.shape[1]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if bon_mode == "bon_rm":
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(token_level_scores.shape[0]):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                weight[i] = n_samples * torch.pow((Q_bc[i] * beta <= cur_Q[i]).float().mean(), n_samples - 1)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif bon_mode == "bon_acc":
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for i in range(token_level_scores.shape[0]):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                weight[i] = n_samples * torch.pow((acc_bc[i] <= acc[i]).float().mean(), n_samples - 1)
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
            raise NotImplementedError
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dpo_loss = (dpo_loss * weight).sum()

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return dpo_loss


# [EXPLAIN] `compute_dpo_accuracy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dpo_acc = []
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for start_id in range(0, token_level_scores.shape[0], n_samples):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cur_scores = (token_level_scores[start_id : start_id + n_samples] * response_mask[start_id : start_id + n_samples]).sum(dim=1)

        # [EXPLAIN] `get_upper_triangle` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def get_upper_triangle(tensor_x):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            diff_matrix = tensor_x.unsqueeze(1) - tensor_x.unsqueeze(0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            upper_tri_indices = torch.triu(torch.ones_like(diff_matrix).bool(), diagonal=1)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return diff_matrix[upper_tri_indices]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cur_acc_diff = get_upper_triangle(acc[start_id : start_id + n_samples])  # in range [-1,1]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cur_score_diff = get_upper_triangle(cur_scores)  # in R
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cur_score_prediction = (cur_score_diff > 0).float()  # in [0,1]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if cur_acc_diff.abs().sum() == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cur_acc = torch.zeros_like(cur_score_prediction[0]) + 0.5
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            cur_acc = (((cur_score_diff > 0) == (cur_acc_diff > 0)).float() * cur_acc_diff.abs()).sum() / cur_acc_diff.abs().sum()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        dpo_acc.append(cur_acc.unsqueeze(0))

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return torch.cat(dpo_acc, dim=0).mean()


# [EXPLAIN] `compute_dpo_abs_accuracy` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples):
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return (torch.sign((token_level_scores * response_mask).sum(dim=-1)) == torch.sign(acc * 2 - 1)).float().mean()
