# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Core functions to implement HGPO (History-based Group Policy Optimization) algorithms.
Self-contained in recipe for upstream verl-agent submission.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict, Counter
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from verl import DataProto
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from functools import reduce


# [EXPLAIN] `to_hashable` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def to_hashable(x):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if isinstance(x, (int, float, str, bool)):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return x
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif isinstance(x, (np.integer, np.floating)):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return x.item()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif isinstance(x, np.ndarray):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tuple(x.flatten())
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif isinstance(x, (list, tuple)):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tuple(to_hashable(e) for e in x)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif isinstance(x, dict):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise TypeError(f"Unsupported type: {type(x)}")

# [EXPLAIN] `compute_step_discounted_returns` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_step_discounted_returns(batch: DataProto, gamma: float):

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print("compute_step_discounted_returns")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    traj_uids = batch.non_tensor_batch['traj_uid']
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    returns_by_traj = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    unique_traj_uids = np.unique(traj_uids)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_rewards = rewards[traj_indices]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_active_masks = active_masks[traj_indices]
        # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        # Calculate returns
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_returns = np.zeros_like(traj_rewards)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        running_return = 0
        
        # Calculate returns from the end to the start
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for t in reversed(range(len(traj_rewards))):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            running_return = traj_rewards[t] + gamma * running_return
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            traj_returns[t] = running_return
        
        # Store the results
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        returns_by_traj[uid] = traj_returns
    
    # Recombine the returns into the original batch order
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_returns = np.zeros_like(rewards)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for i, uid in enumerate(traj_uids):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_indices = np.where(traj_uids == uid)[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        all_returns[i] = returns_by_traj[uid][idx_in_traj]
    
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return all_returns

# [EXPLAIN] `compute_hgpo_outcome_advantage` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_hgpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.array,
    index: np.array,
    traj_index: np.array,
    history_length: int,
    epsilon: float = 1e-6,
    mode: str = 'mean_std_norm',
    length_weight_alpha: float = 1.0,
    base_group: bool = False,
):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute HGPO step-level advantages.

    token_level_rewards: (bs, response_length)
    step_rewards: [N], scalar per sample
    base_group: If True, add episode_advantages as initial group to the aggregate_items weight advantage computation
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    step_advantages = hgpo_advantage_estimate(
        token_level_rewards,
        step_rewards,
        response_mask,
        anchor_obs,
        index,
        traj_index,
        history_length,
        epsilon=epsilon,
        mode=mode,
        length_weight_alpha=length_weight_alpha,
        base_group=base_group,
    )
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return step_advantages, step_advantages

# [EXPLAIN] `trajectory_advantages` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def trajectory_advantages(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        ):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute trajectory-level advantage using mean-std normalization from GiGPO.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = token_level_rewards.shape[-1]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    scores = token_level_rewards.sum(dim=-1)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    id2score = defaultdict(list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    id2mean = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    id2std = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    seen_pairs = set()
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.no_grad():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bsz = scores.shape[0]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(bsz):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if (index[i], traj_index[i]) in seen_pairs:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            id2score[index[i]].append(scores[i])
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not compute_mean_std_cross_steps:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                seen_pairs.add((index[i], traj_index[i]))

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for idx in id2score:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(id2score[idx]) == 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                id2mean[idx] = torch.tensor(0.0)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                id2std[idx] = torch.tensor(1.0)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif len(id2score[idx]) > 1:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"no score in prompt index: {idx}")
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(bsz):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if remove_std:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                scores[i] = scores[i] - id2mean[index[i]]
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return episode_advantages

# [EXPLAIN] `hgpo_advantage_estimate` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def hgpo_advantage_estimate(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    history_length: int,            # max history steps to look at (excluding current), so max k = history_length + 1
    epsilon: float = 1e-6,
    mode: str = "mean_std_norm",    # "mean_std_norm" / "mean_norm"
    length_weight_alpha: float = 1.0,
    base_group: bool = False,
):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    device = response_mask.device
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    response_length = response_mask.shape[-1]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rewards_all = step_rewards.detach().to(device=device, dtype=torch.float32)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    N = anchor_obs.shape[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    all_step_adv = torch.zeros(N, device=device, dtype=torch.float32)

    # ----- optional base (trajectory-level) -----
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    base_adv_scalars = None
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if base_group:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        base_adv = trajectory_advantages(
            token_level_rewards,
            response_mask,
            index,
            traj_index,
            epsilon=epsilon,
            remove_std=(mode == "mean_norm"),
            compute_mean_std_cross_steps=True,
        ).to(device=device, dtype=torch.float32)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask_sum = response_mask.sum(dim=1).clamp(min=1e-8)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        base_adv_scalars = (base_adv * response_mask).sum(dim=1) / mask_sum  # [N]

    # ----- aggregator: items are (k, adv), L=k+1 used for weight -----
    # [EXPLAIN] `aggregate_items` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def aggregate_items(items):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not items:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return torch.zeros((), device=device, dtype=torch.float32)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ks = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        advs = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for (k, a) in items:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            ks.append(k)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            advs.append(a.to(torch.float32))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        advs = torch.stack(advs, dim=0)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ks_t = torch.tensor(ks, device=device, dtype=torch.float32)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        valid_mask = (advs != 0)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if valid_mask.any():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            advs_v = advs[valid_mask]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            ks_v = ks_t[valid_mask]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            L = ks_v + 1
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            w_raw = L ** length_weight_alpha
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            w = w_raw / (w_raw.sum() + epsilon)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            items_val = (w * advs_v).sum()
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            items_val = torch.zeros((), device=device, dtype=torch.float32)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return items_val

    # ===================== main =====================
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for gid in np.unique(index):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        group_indices = np.flatnonzero(index == gid)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if group_indices.size == 0:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            continue

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        group_obs = anchor_obs[group_indices]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        group_traj_ids = traj_index[group_indices]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        group_idx_t = torch.as_tensor(group_indices, device=device, dtype=torch.long)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        group_rewards = rewards_all.index_select(0, group_idx_t)  # [G]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        uniq_traj, inv = np.unique(group_traj_ids, return_inverse=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        n_traj = len(uniq_traj)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_positions = [[] for _ in range(n_traj)]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for pos, t in enumerate(inv):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            traj_positions[t].append(pos)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_positions = [np.asarray(p, dtype=np.int64) for p in traj_positions]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_obs = [group_obs[p] for p in traj_positions]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_rewards = [
            group_rewards.index_select(0, torch.as_tensor(p, device=device, dtype=torch.long))
            for p in traj_positions
        ]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_gidx = [group_indices[p] for p in traj_positions]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        clusters = defaultdict(list)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        traj_h = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for ti in range(n_traj):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            traj_h.append([to_hashable(s) for s in traj_obs[ti]])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        max_k = history_length + 1

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for ti in range(n_traj):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            T = len(traj_h[ti])
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for si in range(T):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                k_upper = min(max_k, si + 1)
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for k in range(1, k_upper + 1):
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    seq = traj_h[ti][si - k + 1 : si + 1]
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    clusters[(k, tuple(seq))].append((ti, si))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        per_step_items = defaultdict(list)

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for (k, _key), members in clusters.items():
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            nL = len(members)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if nL <= 1:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            r = torch.stack([traj_rewards[ti][si] for (ti, si) in members], dim=0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mean = r.mean()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            std = r.std(unbiased=False)

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if mode == "mean_std_norm":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                a = (r - mean) / (std + epsilon)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif mode == "mean_norm":
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                a = (r - mean)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
                raise ValueError(f"Invalid mode: {mode}")

            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for j, (ti, si) in enumerate(members):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                per_step_items[(ti, si)].append((k, a[j]))

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for ti in range(n_traj):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            T = len(traj_obs[ti])
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for si in range(T):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                global_idx = int(traj_gidx[ti][si])
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                items = per_step_items.get((ti, si), [])

                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if base_group and base_adv_scalars is not None:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    base_item = (0, base_adv_scalars[global_idx])
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    items = [base_item] + items

                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                all_step_adv[global_idx] = aggregate_items(items)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    scores = all_step_adv.unsqueeze(-1).expand(-1, response_length) * response_mask
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return scores
