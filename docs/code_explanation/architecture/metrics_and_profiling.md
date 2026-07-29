# metricとGPU profiling

## row metricとtrajectory metric

multi-turn batchでは一trajectoryが複数turn rowになる。`response_length` はrow単位、
`episode/response_tokens` は同じ `traj_uid` のrowを合計したtrajectory単位である。
episode reward/lengthも重複rowをそのまま平均せず、trajectoryの代表indexで集約する。

## task別metric

driver側は `task_name`/`env_kwargs` からcanonical taskを得て、DataProtoをrow sliceして
全体metricと同じ関数を再実行する。worker側は文字列を直接運ばず、`task_ids` tensorと
`task_id_names` mappingでmicro-batch内maskを作る。欠損taskは `-1` となりtask bucketへ入らない。

success rateのようにbatch-wide値を全rowへbroadcastしたmetricは、後からrow sliceしても
task別値を復元できない。そのためenv managerが報告するtask別successを使い、汎用slice側では除外する。
task throughputは共有wall-clockを分母にtoken数だけ分けた寄与であり、task固有latencyではない。

## GPU profiler

profilerはNVMLを優先し、利用できない場合は `nvidia-smi` backendを使う。phase stackで
`gen`、`old_log_prob`、`teacher_forward`、`update_actor` 等の時間窓を記録し、samplingした
GPU utilization/memoryを集約する。これは非同期samplingなので正確なkernel profilerではなく、
phase間のボトルネックとdata-parallel imbalanceを観測する診断値である。
