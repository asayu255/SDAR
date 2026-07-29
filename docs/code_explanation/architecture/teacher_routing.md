# Task 別 Teacher Routing

## Worker 配置

`OPDRayTrainer` は `algorithm.opd.teacher_paths` を task 名で正規化し、
AlfWorld、Search、WebShop ごとに別の `RayClassWithInitArgs` を作る。
全 teacher は `role="ref"` だが shared reference policy ではなく、それぞれ
異なる checkpoint を持つ独立 worker group である。

actor/rollout と同じ resource pool に colocate されるため、同じ GPU 資源を
逐次利用できる。`role="ref"` は FSDP の CPUOffload を有効にし、teacher
parameter は平常時 CPU、forward 時に GPU へ集約され、計算後に reshard/
CPU 戻しされる。colocate は全 model の同時 GPU 常駐を意味しない。

## Row routing

1. `batch.non_tensor_batch["task_name"]` を canonical task 名へ正規化する。
2. teacher ごとに、元 batch 上の global row index `idxs` を抽出する。
3. `DataProto.select_idxs(idxs)` で task-local sub-batch を作る。
4. `meta_info` は共有参照になり得るため shallow copy する。
5. `_verl_auto_padding=True` を sub-batch だけに設定する。
6. `DP_COMPUTE_PROTO` dispatch が teacher world size の倍数へ pad し、戻りを
   元の task row 数へ unpad する。
7. teacher 出力は CPU へ移される。
8. task-local 順 `j` から元 global row `i` へ scatter する。
9. 全 row が処理されたか `seen` で検証し、未知 task または teacher 欠落を
   `ValueError` にする。

## 出力と mutation

通常方式は `(batch,response_length)` の `teacher_log_probs`、top-k 方式は
`(batch,response_length,k)` の `teacher_topk_logprobs` と
`teacher_topk_ids` を、入力 `batch.batch` へ直接書き込む。戻り値は `None`
であり、caller は mutation 済み batch をそのまま actor update へ渡す。
