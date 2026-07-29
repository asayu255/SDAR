# Pure OPD call graph

Pure OPD production pathを中心にした実効call graph。破線相当の「実験」は固定sourceでproduction接続が確認できない経路。

```mermaid
flowchart TD
  A["run_multitask_qwen3.sh"] --> B["main_opd.py"]
  B --> C["TaskRunner.run"]
  C --> D["OPDRayTrainer.fit"]
  D --> E["TrajectoryCollector.multi_turn_loop"]
  E --> F["MultiTaskEnvironmentManager.reset / step"]
  D --> G["compute_teacher_log_probs (batch mutation)"]
  G --> H["task-specific ref worker groups"]
  H --> I["teacher_topk_ids / logprobs / logsumexp"]
  D --> J["actor_rollout_wg.update_actor"]
  J --> K["DataParallelPPOActor.update_policy"]
  K --> L["compute_topk_kl"]
  L --> M["teacher KL scalar loss"]
  N["async_rollout_core.collect_async"] -. "test/scaffolding only" .-> E
```

Pure OPD thin loopは通常PPOのcritic value、advantage/returns、reward-KL、policy-gradient phaseを通らない。`compute_teacher_log_probs`はTensorを返すAPIではなく、入力DataProtoへteacher signalをscatterする副作用境界である。
