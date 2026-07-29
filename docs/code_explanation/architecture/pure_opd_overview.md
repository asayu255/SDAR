# Pure Multitask OPD の全体像

## 結論

この branch の中心は、student 自身が生成した on-policy response を、row の
`task_name` に対応する単一 task teacher で評価し、その teacher KL だけで
student を更新する pure multitask OPD である。offline KD、OPSD、SDAR、
通常の GRPO policy-gradient とは区別する必要がある。

## 実行経路

1. `main_opd.py` が Hydra config を compose した後、Pure OPD の不変条件を
   actor config へ注入する。
2. `OPDRayTrainer.fit()` が student rollout を生成する。
3. multi-turn 展開後の各 row にある `task_name` を正規化し、AlfWorld、
   Search、WebShop の teacher worker group へ分配する。
4. teacher は student と同じ prompt/response を `no_grad` forward し、
   sampled-token log-prob または top-k 分布を返す。
5. teacher 出力を元の global row 順へ戻し、入力 `DataProto` へ書き込む。
6. actor が同じ response 上の student log-prob を計算し、teacher KL を
   response mask 付き token-mean で scalar 化して backward する。

## 実効不変条件

- `use_teacher_kl_loss=True`
- `pg_loss_coef=0`
- `entropy_coeff=0`
- `use_kl_loss=False`
- `use_sdl_loss=False`
- `use_sdar_loss=False`
- `algorithm.use_kl_in_reward=False`
- teacher parameter は CPU offload され、teacher 側へ gradient は流れない
- student log-prob だけが backward graph を保持する

`algorithm.adv_estimator=grpo` は config に存在するが、Pure OPD の thin loop は
old log-prob、critic value、advantage、returns を生成しない。したがって
「設定に存在する」ことは「`fit()` から実行される」ことを意味しない。

## Thin loop と通常 loop の差

Pure OPD は `gen → adjust_batch → response_mask → monitoring reward →
teacher_forward → task ID attachment → update_actor` を主経路とする。通常の
PPO/GRPO にある reference policy forward、reward KL、critic、advantage、
returns、policy-gradient phase は省略される。

validation、checkpoint、metrics は学習信号とは独立に残る。環境 reset prefetch
や rollout session は phase の重なり・同期回数を変える性能機構であり、
teacher KL の科学的定義を変更しない。
