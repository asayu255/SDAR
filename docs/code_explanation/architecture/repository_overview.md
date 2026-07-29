# リポジトリ全体像

このブランチは VERL の分散 RL 基盤へ、環境との multi-turn interaction、
SDAR 系 trainer、Pure OPD、task-balanced sampling、task-specific teacher を
重ねている。Pure OPD の入口は
`examples/opd_trainer/run_multitask_qwen3.sh` → `verl/trainer/main_opd.py` →
`verl/trainer/ppo/opd_ray_trainer.py` である。

## 主要ディレクトリ

| パス | 責務 |
|---|---|
| `examples/` | Hydra override を含む実験起動、データ前処理、期待設定 |
| `agent_system/environments/` | task 固有 env と mixed batch の分配・再結合 |
| `agent_system/multi_turn_rollout/` | action生成、env step、turn row の収集 |
| `verl/trainer/` | worker graph、学習 loop、loss 入力の組み立て |
| `verl/workers/` | actor/critic/teacher/rollout の FSDP・Megatron 実装 |
| `verl/models/` | model forward と tensor parallel 対応 |
| `tests/` | config、routing、metric、async equivalence の契約 |

## Pure OPDで有効な最短経路

1. task-balanced sampler が AlfWorld/Search/WebShop を混ぜた prompt batch を作る。
2. multitask environment manager が row を task 別 env へ送り、元順序へ戻す。
3. rollout loop が trajectory を turn row へ展開する。
4. teacher router が `task_name` で row を三つの teacher worker group へ分配する。
5. teacher の top-k token id/log-prob と tail mass を global row 順へ復元する。
6. actor が同じ token support 上の KL を計算し、response mask で平均して更新する。

共有 PPO/SDAR コードには advantage、reward KL、critic、entropy などの経路もあるが、
Pure OPD entrypoint が係数と flag を上書きするため、存在だけで実効 loss と判断してはならない。
