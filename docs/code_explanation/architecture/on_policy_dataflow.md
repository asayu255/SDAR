# On-policy Multitask Dataflow

## Preparation

`prepare_sdar_multitask.py` は train/test parquet を task 混在形式で作る。
AlfWorld と WebShop は dummy prompt row を使い、実際の game/task は environment
reset 時の内部 schedule から決まる。Search は question、ground truth、
data source を保持し、`env_kwargs` へ渡す。

```text
prepare_sdar_multitask.py
  → train.parquet / test.parquet
  → RLHFDataset
  → task-specific max_prompt_length / truncation
  → TaskBalancedSampler
  → collate_fn
  → DataProto
  → TrajectoryCollector
```

train では指定 seed で必要件数を sample し、test は評価用の固定構成を作る。
dataset row の集合と、environment が episode 内容を決める schedule は別の状態である。

## Dataset と metadata

`RLHFDataset` は task_name を canonical 名へ正規化し、task override を global
data config より優先する。token Tensor に加えて、`task_name`、Search の
question/ground truth/data source を含む `env_kwargs` を non-tensor batch に
残す。`collate_fn` 後もこれらは row と同じ順序で `DataProto` へ入る。

## Prompt、trajectory、turn-row

`data.train_batch_size=45` は prompt batch、`env.rollout.n=8` は各 prompt の
rollout group size であるため、生成する trajectory 数には `45 × 8` が関係する。
ただし multi-turn 展開では1 trajectory が複数 env turn row を持つ。actor
update の row 数は trajectory 数と同一とは限らない。

各 turn row は同じ `traj_uid` で trajectory へ再集約でき、response mask と
turn ごとの response 長を持つ。trajectory-level episode reward/success と、
turn-row token length を同じ batch dimension の metric として混同しない。
