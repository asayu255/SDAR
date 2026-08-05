"""品質修正計画で手動レビューしたコード固有注釈をmapへ反映する。"""

from __future__ import annotations

from annotation_map_utils import (
    annotation_type_for,
    comment_style_for,
    load_annotation_map,
    save_annotation_map,
)
from common import ROOT, language_for


def entry(
    curated_id: str,
    priority: str,
    path: str,
    start: int,
    end: int,
    *lines: str,
    annotation_type: str | None = None,
) -> dict:
    source_lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    if start < 1 or start > len(source_lines):
        raise ValueError(f"{path}:{start} is outside source")
    source_line = source_lines[start - 1]
    indent = source_line[: len(source_line) - len(source_line.lstrip())]
    language = language_for(path)
    return {
        "original_path": path,
        "source_start_line": start,
        "source_end_line": end,
        "annotation_lines": list(lines),
        "annotation_type": annotation_type or annotation_type_for(lines),
        "review_status": "reviewed",
        "legacy_disposition": "rewrite",
        "indent": indent,
        "comment_style": comment_style_for(language),
        "sequence": 0,
        "curated_id": curated_id,
        "priority": priority,
    }


CURATED = [
    # Priority A: executable experiment definition.
    entry(
        "A-run-data-prep",
        "A",
        "examples/opd_trainer/run_multitask_qwen3.sh",
        44,
        51,
        "`prepare_sdar_multitask`が各task 15 prompt × 300 batch分のtrain rowと、"
        "task別126件のvalidation rowを固定seedで作る。",
        "AlfWorld/WebShopのrowは環境側でepisodeを選ぶdummy prompt、Searchのrowは"
        "questionとground truthを`env_kwargs`へ運ぶ。",
        annotation_type="configuration",
    ),
    entry(
        "A-run-hydra",
        "A",
        "examples/opd_trainer/run_multitask_qwen3.sh",
        53,
        150,
        "このHydra commandがQwen3-1.7B student、45 prompt、`env.rollout.n=8`、"
        "task別teacher checkpoint、top-k=20を一つの実効runへcomposeする。",
        "`actor_rollout_ref.rollout.n=1`はengine側の複製数であり、1 prompt当たりの"
        "trajectory数は`env.rollout.n=8`が担う。multi-turn展開後のactor row数は360 trajectoryより多くなり得る。",
        "actor micro-batchは2/GPU、teacher/ref forwardは4/GPU、rollout log-probは16/GPUで、"
        "同じ「batch size」ではない。",
        annotation_type="configuration",
    ),
    entry(
        "A-expect-loss",
        "A",
        "examples/opd_trainer/expected_multitask_config.yaml",
        20,
        28,
        "この4組はOPD algorithm設定と、`main_opd.py`がactorへ注入した実効設定の一致を検査する。",
        "`topk_kl`とK=20のどちらかがdefaultやoverrideで変われば、学習開始前にfail-fastする。",
        annotation_type="configuration",
    ),
    entry(
        "A-expect-invariants",
        "A",
        "examples/opd_trainer/expected_multitask_config.yaml",
        30,
        43,
        "`pg_loss_coef=0`、entropy=0、reference-KL/SDL/SDAR/reward-KL=falseが"
        "teacher KL以外のgradient寄与を閉じる。",
        "invalid-action penaltyは有効だが、Pure OPD thin loopはadvantage/returnsを作らないため、"
        "変更後rewardはmonitoringとdumpに留まりactor lossへ入らない。",
        annotation_type="configuration",
    ),
    entry(
        "A-expect-teachers",
        "A",
        "examples/opd_trainer/expected_multitask_config.yaml",
        45,
        48,
        "3つのpathはshared reference policyではなく、`task_name`で選ばれる独立teacher checkpointである。",
        "各pathは`OPDRayTrainer.init_workers`でdeepcopyしたref-role設定の`model.path`へ入る。",
        annotation_type="configuration",
    ),
    entry(
        "A-expect-batching",
        "A",
        "examples/opd_trainer/expected_multitask_config.yaml",
        56,
        73,
        "45 promptはtask別15件を合わせたglobal prompt batchであり、`TaskBalancedSampler`が"
        "各DP chunkのtask構成を保つ。prompt長とtruncationはtask overrideがglobal値より優先される。",
        annotation_type="configuration",
    ),
    entry(
        "A-expect-environment",
        "A",
        "examples/opd_trainer/expected_multitask_config.yaml",
        75,
        88,
        "`env.rollout.n=8`が各promptのtrajectory複製数を決める。task別max_stepsは"
        "AlfWorld=50、Search=4、WebShop=15で、mixed batch内でも各sub-managerへ個別に渡る。",
        annotation_type="configuration",
    ),
    entry(
        "A-main-hydra",
        "A",
        "verl/trainer/main_opd.py",
        15,
        17,
        "`@hydra.main`が`ppo_trainer.yaml`へCLI overrideをcomposeし、未解決のDictConfigを"
        "Pure OPD専用`run_opd`へ渡すentrypointである。",
        annotation_type="configuration",
    ),
    entry(
        "A-main-ray-init",
        "A",
        "verl/trainer/main_opd.py",
        20,
        31,
        "既存Ray runtimeがなければ、VERL既定runtime_envと`config.ray_init.runtime_env`をmergeしてdriverからclusterへ接続する。",
        "`ray.init`はactorを作る処理ではなく、後続のremote task/worker groupが使うruntimeを初期化する境界である。",
        annotation_type="execution_context",
    ),
    entry(
        "A-main-ray-block",
        "A",
        "verl/trainer/main_opd.py",
        33,
        34,
        "`OPDTaskRunner.remote()`で1 CPUのRay actorを作り、`runner.run.remote(config)`を非同期投入する。",
        "外側の`ray.get`がdriverの明示的blocking pointであり、worker初期化・training・validation・checkpointが終了するまで`run_opd`は戻らない。",
        annotation_type="synchronization",
    ),
    entry(
        "A-main-resolve",
        "A",
        "verl/trainer/main_opd.py",
        46,
        54,
        "`OmegaConf.resolve`後の値を基準にteacher pathの存在を確認するため、Hydra interpolationやCLI overrideを反映した実効設定を検査する。",
        annotation_type="configuration",
    ),
    entry(
        "A-main-open-dict",
        "A",
        "verl/trainer/main_opd.py",
        58,
        69,
        "`open_dict(config)`はOmegaConfのstruct制約をこのblockだけ一時解除し、compose済み設定へPure OPD固有keyと実効値を書き込む。",
        "resource、autocast、no-gradの寿命を管理するcontext managerではない。",
        annotation_type="configuration",
    ),
    entry(
        "A-main-intent-check",
        "A",
        "verl/trainer/main_opd.py",
        75,
        82,
        "検証順はHydra compose → Pure OPD invariant注入 → expected-config比較である。",
        "`enforce_expected_config`はraw defaultではなくworkerが読む直前のeffective configをdot pathごとに照合する。",
        annotation_type="configuration",
    ),
    entry(
        "A-main-student-model",
        "A",
        "verl/trainer/main_opd.py",
        84,
        97,
        "`actor_rollout_ref.model.path`だけをlocalizeしてstudent tokenizer/processorを構築する。",
        "task別teacher pathはここでstudent pathへ混ぜず、後の`OPDRayTrainer.init_workers`で各ref-role copyへ設定する。",
        annotation_type="execution_context",
    ),
    entry(
        "A-main-worker-backend",
        "A",
        "verl/trainer/main_opd.py",
        106,
        127,
        "actor strategyに応じてFSDPまたはMegatron worker group実装を選ぶ。現在runはFSDPで、"
        "同じ`ActorRolloutRefWorker` classをactor_rollout roleとtask別ref roleに使い分ける。",
        annotation_type="distributed",
    ),
    entry(
        "A-main-role-map",
        "A",
        "verl/trainer/main_opd.py",
        131,
        169,
        "global resource poolへActorRolloutとCriticのclassを登録するが、`Role.RefPolicy`は登録しない。",
        "task別teacherはshared ref policyではなく、OPDRayTrainerが同じpool上へ追加する独立worker groupである。",
        "reward managerはmonitoring用scoreを作るが、Pure OPD thin loopのscalar lossには接続されない。",
        annotation_type="distributed",
    ),
    entry(
        "A-main-rollout-count",
        "A",
        "verl/trainer/main_opd.py",
        171,
        178,
        "engine側`actor_rollout_ref.rollout.n`を1に固定し、GRPO groupの8 trajectoryは環境側`env.rollout.n`で複製する。",
        "この分離によりmulti-turn collectorがtrajectory UIDとturn-rowを一貫して管理する。",
        annotation_type="configuration",
    ),
    entry(
        "A-main-trainer",
        "A",
        "verl/trainer/main_opd.py",
        198,
        217,
        "`OPDRayTrainer`はcompose済みconfigからteacher pathを読み、`init_workers`でactorの前に3 teacherを初期化する。",
        "`fit`は通常PPO loopではなく、rollout → monitoring reward → teacher forward → actor updateのthin loopを実行する。",
        annotation_type="execution_context",
    ),
    entry(
        "A-trainer-init",
        "A",
        "verl/trainer/ppo/opd_ray_trainer.py",
        119,
        141,
        "`teacher_paths`のkeyをcanonical task名へ正規化し、未知taskとshared reference-policy生成を拒否する。",
        "`teacher_topk_kl`がtrueならteacherは各response位置のK個のID/log-probを返し、falseなら生成token位置だけのsingle-token signalを返す。",
        annotation_type="configuration",
    ),
    entry(
        "A-trainer-teacher-workers",
        "A",
        "verl/trainer/ppo/opd_ray_trainer.py",
        166,
        182,
        "student設定をtaskごとにdeepcopyし、`model.path`だけを対応checkpointへ差し替えるためactor本体のpathは変わらない。",
        "各copyを`role=\"ref\"`で同一resource poolへ登録する。colocationはGPU常時同時常駐を意味せず、FSDP CPUOffload下でforward時に逐次利用される。",
        annotation_type="distributed",
    ),
    entry(
        "A-trainer-colocated-spawn",
        "A",
        "verl/trainer/ppo/opd_ray_trainer.py",
        196,
        210,
        "`create_colocated_worker_cls`が同じRay process/GPU配置へactorとteacher roleを束ね、`spawn`がtask別worker group handleを返す。",
        "`wg.init_model()`はRay側でblockingして各teacher checkpointのFSDP構築完了を待つ。teacherを先に初期化し、vLLMのKV-cache見積りを行うactor_rolloutを最後に初期化する。",
        annotation_type="synchronization",
    ),
    entry(
        "A-trainer-monitoring-metrics",
        "A",
        "verl/trainer/ppo/opd_ray_trainer.py",
        56,
        101,
        "`compute_opd_data_metrics`はadvantage/returnsを読まず、turn-row長、trajectory token数、episode reward/successだけを集計する。",
        "ここで読む`token_level_scores`とinvalid-action penalty後rewardはmonitoring値であり、`update_actor`へloss入力として渡されない。",
        annotation_type="semantic",
    ),
    entry(
        "A-fsdp-mesh",
        "A",
        "verl/workers/fsdp_workers.py",
        80,
        97,
        "1次元meshは全rankでparameterをshardするFULL_SHARD、2次元`(ddp, fsdp)` meshはnode/group内shardとreplicationを組み合わせるHYBRID_SHARDを選ぶ。",
        "これは明示的barrierではないが、forward/backwardのAllGather/ReduceScatterがrank間のcollective待機点になる。",
        annotation_type="distributed",
    ),
    entry(
        "A-fsdp-batch-normalization",
        "A",
        "verl/workers/fsdp_workers.py",
        146,
        169,
        "global `ppo_mini_batch_size`と旧global micro-batch指定を、DP world size / Ulysses SP sizeでrank-local値へ正規化する。",
        "現在runは既に`ppo_micro_batch_size_per_gpu=2`、teacher/refは4/GPUを指定するため、両者を同じmicro-batchとして扱わない。",
        annotation_type="configuration",
    ),
    entry(
        "A-fsdp-role-offload",
        "A",
        "verl/workers/fsdp_workers.py",
        310,
        350,
        "`role=\"actor\"`はgradient accumulationの正確性のためparameter CPUOffloadを無効にし、task別teacherの`role=\"ref\"`はCPUOffloadを有効にする。",
        "FSDP1/FSDP2ともforward前のparameter materializationと計算後reshardを行うため、colocated teacherが常時GPU residentとは限らない。",
        annotation_type="distributed",
    ),
    entry(
        "A-fsdp-update-actor",
        "A",
        "verl/workers/fsdp_workers.py",
        600,
        641,
        "`DP_COMPUTE_PROTO`がglobal DataProtoを各DP rankへ配送し、Ulysses preprocess後に`update_policy`を同期実行する。",
        "各micro-batch backwardのFSDP collectiveを経てoptimizer stepが完了してから、CPU上のmetric DataProtoがdriverへ戻る。",
        annotation_type="synchronization",
    ),
    entry(
        "A-fsdp-rollout-session",
        "A",
        "verl/workers/fsdp_workers.py",
        643,
        715,
        "通常は各turnでsharding manager enter/exitに伴うFSDP state gather、vLLM weight sync、wake/sleepを行う。",
        "session modeはrollout境界で1回だけenter/exitし、凍結actor weightを全turnで共有する。per-turn DataProto shardingは残るためrow順は変わらない。",
        annotation_type="synchronization",
    ),
    entry(
        "A-fsdp-teacher-topk",
        "A",
        "verl/workers/fsdp_workers.py",
        796,
        822,
        "teacher sub-batchをGPUへ移し、ref用micro-batch=4/GPUとUlysses shardingを適用して`compute_topk_log_prob`をno-grad実行する。",
        "返却する`teacher_topk_logprobs`/`teacher_topk_ids`は`(task_batch,response_length,K)`で、postprocess後CPUへ移動する。",
        "FSDP1では`reshard(True)`がroot parameterを再shardする。これは明示的`dist.barrier`とは別のcollective境界である。",
        annotation_type="tensor_shape",
    ),
    entry(
        "A-actor-forward-rmpad",
        "A",
        "verl/workers/actor/dp_actor.py",
        77,
        241,
        "`_forward_micro_batch`は`input_ids(B,S)`をattention maskで`input_ids_rmpad(1,nnz)`へ変換し、FlashAttention varlen forward後にresponse位置へ戻す。",
        "teacher modeはlogitsから自前top-k ID/log-probを返し、student modeは`teacher_topk_ids(B,R,K)`と同じ位置をgatherする。",
        "top-k distillationはfull logitsを必要とするためfused-kernel経路をguardし、Ulysses使用時はpad/sliceとgather/unpadの順序を維持する。",
        annotation_type="tensor_shape",
    ),
    entry(
        "A-actor-teacher-forward",
        "A",
        "verl/workers/actor/dp_actor.py",
        362,
        407,
        "`compute_topk_log_prob`はteacher/ref forward全体を`torch.no_grad()`で囲み、micro-batchごとの"
        "`(B_i,R,K)`をconcatしてdynamic-batch並べ替えを元row順へ戻す。",
        "IDはint64のまま返し、teacher側Tensorからstudentへのgradient edgeは作らない。",
        annotation_type="tensor_shape",
    ),
    entry(
        "A-actor-effective-inputs",
        "A",
        "verl/workers/actor/dp_actor.py",
        410,
        450,
        "`pg_loss_coef==0`なら`old_log_probs`/`advantages`をselectせず、teacher signal、response mask、task IDだけをupdate inputへ残す。",
        "`teacher_kl_loss_type=topk_kl`では`teacher_topk_ids/logprobs`をstudent micro-batchと同じrow sliceで運ぶ。",
        annotation_type="configuration",
    ),
    entry(
        "A-core-topk-kl",
        "A",
        "verl/trainer/ppo/core_algos.py",
        682,
        725,
        "`student_topk_logprob`と`teacher_topk_logprob`はともに`(B,R,K)`で、teacherが選んだ同一support上のreverse KLを計算する。",
        "K個の確率質量にtop-k外をまとめたtail bucketを加え、出力`teacher_kld(B,R)`をresponse maskでtoken-mean集約する。",
        "teacher入力はdetachされ、single sampled tokenの`low_var_kl`とは異なりstudentのK位置すべてへgradientが流れる。",
        annotation_type="tensor_shape",
    ),
    entry(
        "A-config-actor-defaults",
        "A",
        "verl/trainer/config/ppo_trainer.yaml",
        47,
        73,
        "ここは共有PPOのraw defaultであり、Pure OPDの実効値ではない。`main_opd.py`がteacher KLを有効化し、"
        "PG/entropy/reference-KLを無効化した後にexpected-configが照合する。",
        annotation_type="configuration",
    ),
    entry(
        "A-config-ref-teacher",
        "A",
        "verl/trainer/config/ppo_trainer.yaml",
        99,
        112,
        "task別teacherはこの`ref` micro-batch/offload系列を再利用するが、通常のshared RefPolicy roleではない。",
        "現在runの4/GPUはteacher forward専用で、actor updateの2/GPUとは独立に正規化される。",
        annotation_type="configuration",
    ),
    entry(
        "A-config-rollout",
        "A",
        "verl/trainer/config/ppo_trainer.yaml",
        113,
        161,
        "rollout backendのsampling、KV-cache、vLLM sessionに関する性能defaultであり、"
        "expected-configが固定する科学的loss設定とは分離される。",
        "`rollout.n`はengine側sample数で、Pure OPD multitask runでは1に固定する。",
        annotation_type="configuration",
    ),
    entry(
        "A-config-shared-algorithm",
        "A",
        "verl/trainer/config/ppo_trainer.yaml",
        244,
        268,
        "GAE/GRPO、reward KL、filter_groupsはshared trainer用defaultである。Pure OPD thin loopは"
        "advantage/returnsを作らず、entrypointが`use_kl_in_reward=false`を強制するため実効lossへ到達しない。",
        annotation_type="configuration",
    ),
    entry(
        "A-config-env",
        "A",
        "verl/trainer/config/ppo_trainer.yaml",
        302,
        325,
        "global env defaultに対しmultitask runがtask別max_stepsをoverrideする。mixed batchは"
        "AlfWorld/Search/WebShop sub-managerへ分割され、結果だけ元global row順へscatterされる。",
        annotation_type="configuration",
    ),
    entry(
        "B-prepare-dummy-rows",
        "B",
        "examples/data_preprocess/prepare_sdar_multitask.py",
        31,
        51,
        "AlfWorld/WebShopのepisode内容はdataset行ではなく環境内部のseeded scheduleが選ぶため、"
        "ここでは空promptと`task_name`を持つ制御行だけを作る。",
        "`extra_info.index`はsplit内の追跡用、`env_kwargs.task_name`はmixed environment routerのdispatch keyであり、"
        "どちらも教師入力そのものではない。",
        annotation_type="configuration",
    ),
    entry(
        "B-prepare-search-normalization",
        "B",
        "examples/data_preprocess/prepare_sdar_multitask.py",
        57,
        87,
        "Searchだけはquestion/ground truthが各episodeの実体なので、旧parquetの複数schemaから値を回収し、"
        "`env_kwargs`へ正規化して環境へ渡す。",
        "ground truthの優先順はenv_kwargs→reward_model→golden_answersで、欠損schemaを吸収しても"
        "`task_name=search`というrouting invariantは必ず付与する。",
        annotation_type="configuration",
    ),
    entry(
        "B-prepare-balanced-splits",
        "B",
        "examples/data_preprocess/prepare_sdar_multitask.py",
        89,
        124,
        "test Searchはsingle-task baselineと同じ先頭母集団、trainはseed付き一様sampleを使う。"
        "連結後の行順は固定し、学習時の混合順序は`TaskBalancedSampler`だけに委ねる。",
        "3 taskの件数を直後に検証するため、欠落・過剰samplingをparquet生成時点で検出できる。",
        annotation_type="experimental",
    ),
    entry(
        "B-main-ppo-balanced-sampler",
        "B",
        "verl/trainer/main_ppo.py",
        226,
        269,
        "samplerは各taskのindex poolを分離し、1 global generation batchが"
        "`per_task_batch_size × task数`になることを初期化時に強制する。",
        "最小poolより長いrunでは各task内だけを再shuffleして循環利用するため、task比率は崩れない。"
        "一方、dataset行数やsample集合自体は変更しない。",
        annotation_type="configuration",
    ),
    entry(
        "B-main-ppo-dp-interleave",
        "B",
        "verl/trainer/main_ppo.py",
        271,
        303,
        "`TASK_BALANCE_INTERLEAVE`有効時はtask-major配列をround-robinへ並べ替える。"
        "これは各data-parallel rankの連続chunkにも同じtask比率を届けるためのlayout変換である。",
        "epochごとに`seed + epoch`でtask内shuffleを再現し、yield総数は"
        "`num_batches × per_task_batch_size × task数`から変わらない。",
        annotation_type="distributed",
    ),
    entry(
        "B-dataset-task-overrides",
        "B",
        "verl/utils/dataset/rl_dataset.py",
        165,
        186,
        "各rowの`task_name`を正規化し、task別`max_prompt_length`/`truncation`を共有dataset上で選ぶ。"
        "row直下にない旧形式は`env_kwargs.task_name`へfallbackする。",
        "overrideがないtaskはglobal defaultへ戻るため、single-task datasetの挙動も維持される。",
        annotation_type="configuration",
    ),
    entry(
        "B-dataset-item-tokenization",
        "B",
        "verl/utils/dataset/rl_dataset.py",
        248,
        362,
        "`__getitem__`はtask別上限を確定してからchat templateとmultimodal入力をtokenizeし、"
        "左padding済み`input_ids/attention_mask/position_ids`と環境用non-tensor metadataを同じrowへ束ねる。",
        "この境界より後ではtaskごとのprompt長差がtensor shapeへ反映済みなので、collate時はbatch最大長へのpaddingだけでよい。",
        annotation_type="tensor_shape",
    ),
    entry(
        "B-env-manager-reset-routing",
        "B",
        "agent_system/environments/env_manager.py",
        619,
        689,
        "mixed batchを`env_kwargs.task_name`でtask別sub-managerへ分割し、resetをthread poolへ同時投入する。"
        "`future.result()`はRayではなくホスト側threadの完了待ちで、I/O/IPC待ちをtask間で重ねる。",
        "sub-managerの戻り値は保存したglobal indexへscatterされ、呼出側から見えるrow順とbatch sizeは入力時のまま保たれる。",
        annotation_type="synchronization",
    ),
    entry(
        "B-env-manager-step-routing",
        "B",
        "agent_system/environments/env_manager.py",
        691,
        755,
        "stepもactive taskだけを並行実行し、既に全episode終了またはtask固有max_steps到達のtaskは"
        "最後の観測とdone-only結果でshort-circuitする。",
        "各taskのreward/done/infoをglobal rowへscatterした後に返すため、短いSearchと長いAlfWorldが混在しても"
        "trajectory indexの対応は変化しない。",
        annotation_type="execution_context",
    ),
    entry(
        "B-resume-schedule-invariant",
        "B",
        "agent_system/environments/resume.py",
        16,
        40,
        "checkpointはdataloader位置を復元しても、プロセス内で再生成されるWebShop/ALFWorldのepisode schedule位置までは保存しない。"
        "そこでcompleted global step数と「1 stepにつきreset 1回」の不変条件からreset回数を再生する。",
        "Searchはepisodeがrowの`env_kwargs`で決まるため対象外であり、dynamic filteringでreset回数が変わる構成にも適用できない。",
        annotation_type="experimental",
    ),
    entry(
        "B-resume-fast-forward",
        "B",
        "agent_system/environments/resume.py",
        64,
        92,
        "single-task managerとmultitask配下のleaf vector envを列挙し、`fast_forward`を公開するstateful envだけを進める。"
        "stateless envは明示的にskipする。",
        "再生失敗は再現性を損なうがcheckpoint復旧自体を止めない設計なので、例外をmessageへ変換して学習を継続する。",
        annotation_type="error_condition",
    ),
    entry(
        "B-rollout-prefetch-reset",
        "B",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        678,
        712,
        "次rolloutのCPU/subprocess resetを、現rollout終了後のGPU training区間へ重ねる。"
        "pending futureは同じenv instanceかつ同一kwargsでのみ消費できる。",
        "`future.result()`は次rollout開始時のthread合流点である。kwargs不一致時に再resetするとstateful scheduleが二重に進むため、"
        "黙ってfallbackせず例外にする。",
        annotation_type="synchronization",
    ),
    entry(
        "B-rollout-vanilla-loop",
        "B",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        714,
        864,
        "同期loopは未完了trajectoryだけをvLLM generationへ送り、出力をfull batch位置へscatterして履歴を更新する。"
        "taskごとの終了turnが違っても、done rowの不要なGPU生成を避けながらglobal indexを保つ。",
        "任意のlog-prob prefetchでは`envs.step`をhost threadへ投入し、そのI/O待ち中だけ同じ凍結actorで"
        "完了rowのold log-probを計算する。`future.result()`が両処理の合流点で、次のgenerationとは競合しない。",
        annotation_type="synchronization",
    ),
    entry(
        "B-rollout-entry-selection",
        "B",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        929,
        1015,
        "`multi_turn_loop`はfeature flagによりasync prototypeか既存同期loopを選ぶ公開境界である。"
        "async経路も最終的には同じDataProto field、uid grouping、reward/length統計へ変換する必要がある。",
        "比較実験ではsampling semanticsを変えずschedulerだけを切り替えることが前提で、未対応条件は同期経路へ戻す。",
        annotation_type="experimental",
    ),
    entry(
        "B-rollout-logprob-reuse",
        "B",
        "agent_system/multi_turn_rollout/utils.py",
        87,
        155,
        "`prefetched`は`(traj_uid, turn_step)`をkeyに、rollout中の凍結actorで得たtoken単位"
        "`old_log_probs/entropys`を再利用する。",
        "欠損rowだけをworker world sizeへpadして再計算し、最後に元batch順の`(B,R)` tensorへscatterする。"
        "response長不一致やkey欠損時は安全側で通常計算へfallbackする。",
        annotation_type="tensor_shape",
    ),
    entry(
        "B-rollout-batch-divisibility",
        "B",
        "agent_system/multi_turn_rollout/utils.py",
        157,
        202,
        "rollout/ref/actorそれぞれのmicro-batch制約の最小公倍数を求め、batch sizeを全経路で割り切れる値へ調整する。",
        "`copy`はrowを複製してsampleを保持し、`delete`はランダムに余剰rowを落とすため、"
        "両modeは統計量と再現性への影響が異なる。",
        annotation_type="distributed",
    ),
    entry(
        "B-async-collection-contract",
        "B",
        "agent_system/multi_turn_rollout/async_rollout_core.py",
        111,
        150,
        "各trajectoryが`generate_action → env_step`を独立に繰り返すasync schedulerで、"
        "終了条件とuid groupingは同期loopと同じに保つ。",
        "`max_in_flight`は生成中coroutine数だけをSemaphoreで制限し、収集するsampleやstep順序の意味は変更しない。",
        annotation_type="execution_context",
    ),
    entry(
        "B-async-gather",
        "B",
        "agent_system/multi_turn_rollout/async_rollout_core.py",
        152,
        190,
        "Semaphore acquire/releaseは各generationの同時実行枠を管理し、`finally`により生成例外時もpermitを返す。"
        "環境stepは同じtrajectory内では直列である。",
        "最後の`asyncio.gather`は全trajectory coroutineの合流点であり、Ray taskやOS threadの待機ではない。",
        annotation_type="synchronization",
    ),
    entry(
        "C-dispatch-register",
        "C",
        "verl/single_controller/base/decorator.py",
        490,
        548,
        "`register`は関数をその場で分散実行せず、dispatch/execute/blocking metadataをwrapperへ付与する。"
        "worker group構築時にこのmetadataから引数分割、remote call、結果collectの関数が結線される。",
        "`materialize_futures`はdispatch前に`DataProtoFuture.get()`を呼ぶホスト側依存解決であり、"
        "CUDA synchronizationやdistributed barrierではない。",
        annotation_type="distributed",
    ),
    entry(
        "C-ray-generated-method",
        "C",
        "verl/single_controller/ray/base.py",
        44,
        61,
        "動的worker-group methodはdispatch→Ray remote投入→必要なら`ray.get`→collect→auto-padding除去の順で実行する。",
        "`blocking=False`ならObjectRefをcollect側へ渡すためdriverは待たず、`blocking=True`の`ray.get`が明示的なdriver合流点になる。",
        annotation_type="synchronization",
    ),
    entry(
        "C-ray-execute-sync-async",
        "C",
        "verl/single_controller/ray/base.py",
        490,
        600,
        "`execute_*_async`は各Ray actorへ`.remote()`を投入してObjectRefを返す。"
        "`execute_*_sync`だけが`ray.get()`で完了と例外をdriverへmaterializeする。",
        "全引数がworker数と同長のlistならrankごとに対応要素をscatterし、それ以外は同じ引数をbroadcastする。"
        "これはcollective通信ではなくdriverによるRay task引数routingである。",
        annotation_type="synchronization",
    ),
    entry(
        "C-sharding-base-contract",
        "C",
        "verl/workers/sharding_manager/base.py",
        21,
        32,
        "sharding managerのcontext境界はtraining表現からinference表現への一時的な切替を表す。"
        "`__enter__`でweight/RNG/memoryをrollout側へ移し、`__exit__`でtraining状態へ戻す。",
        "`preprocess_data`/`postprocess_data`は同じglobal batchをinference TP/SP配置へgatherし、終了後に元rankのsliceへ戻す契約である。",
        annotation_type="distributed",
    ),
    entry(
        "C-fsdp-vllm-session",
        "C",
        "verl/workers/sharding_manager/fsdp_vllm.py",
        114,
        235,
        "entryではFSDP shardからstate dict（LoRA時はadapter、初回はbaseを含む）をmaterializeし、"
        "vLLM weight領域をwakeして同期した後にKV cacheをwakeする。",
        "FSDP full-param materializationはrank間AllGatherを伴い、`sync_model_weights/update_params`はtraining weightを"
        "inference engineへコピーする別の同期である。exitの`sleep(level=1)`はvLLMメモリ解放状態への遷移でbarrierではない。",
        "training RNGとgeneration RNGを交換してTP rank間のsamplingを一致させ、exitで元のtraining RNGと`train()`状態を復元する。",
        annotation_type="synchronization",
    ),
    entry(
        "C-fsdp-sglang-session",
        "C",
        "verl/workers/sharding_manager/fsdp_sglang.py",
        92,
        181,
        "FSDP state dictをGPUへmaterializeし、infer-TP各rankのserialized shardを`dist.gather_object`でrank 0へ集めて"
        "SGLang engineへweight単位でloadする。",
        "ここでのgatherはweight同期、`all_gather_data_proto`はrollout入力複製で目的が異なる。"
        "exitではSGLang memory occupationをreleaseし、actorをtrain modeとtraining RNGへ戻す。",
        annotation_type="distributed",
    ),
    entry(
        "C-fsdp-ulysses-reshard",
        "C",
        "verl/workers/sharding_manager/fsdp_ulysses.py",
        27,
        70,
        "context中だけglobal sequence-parallel groupをこのmodelのSP meshへ差し替え、終了時に必ず旧groupへ戻す。",
        "FSDPのDP shardとして届いたDataProtoをSP group内AllGatherして各SP rankへ同一入力を渡し、"
        "postprocessでSP rank対応chunkへ戻して外側のDP配置を復元する。",
        annotation_type="distributed",
    ),
    entry(
        "C-megatron-sglang-session",
        "C",
        "verl/workers/sharding_manager/megatron_sglang.py",
        84,
        155,
        "Megatron shardはconverterを通じてSGLangが読むtensor名/layoutへ逐次変換され、TP rank 0のengineへ反映される。"
        "cache flush後にrolloutし、exitで推論memoryをreleaseして全pipeline modelをtrain modeへ戻す。",
        "入力DataProtoはinfer TP group内AllGather、出力はrank対応chunkへ戻すため、weight変換とbatch reshardは独立した処理である。",
        annotation_type="distributed",
    ),
    entry(
        "C-megatron-vllm-session",
        "C",
        "verl/workers/sharding_manager/megatron_vllm.py",
        308,
        365,
        "Megatron parameter generatorがPP/TP shardをvLLM形式へ変換し、versionに応じて一括syncまたはwake後の直接loadを行う。",
        "新vLLMではweight領域→KV cacheの順にwakeし、rollout終了時に`sleep(level=1)`で推論memoryを解放してtrainingへ戻す。",
        "DataProtoのAllGather/chunkはinfer TPのbatch配置を変えるcollectiveであり、parameter loadとは別の同期点である。",
        annotation_type="synchronization",
    ),
    entry(
        "C-gpu-profiler-sampler",
        "C",
        "verl/utils/gpu_profiler.py",
        210,
        300,
        "daemon threadがNVML/SMIのdevice-wide counterを一定間隔で読むため、値は現在processだけでなく同GPU上の全workloadを含む。"
        "phase stack最上位のlabelを各sampleへ付け、lockはhost側のsample buffer整合性だけを守る。",
        "`mean_util_between`は指定wall-clock窓、`report_and_reset`はtrainer step境界の集計であり、"
        "どちらもCUDA synchronizeを挿入しないため測定自体でGPU pipelineを止めない。",
        annotation_type="execution_context",
    ),
    entry(
        "C-metrics-task-slicing",
        "C",
        "verl/trainer/ppo/metric_utils.py",
        94,
        163,
        "global batchの`task_name`からrow indexを作り、overallと同じmetric関数をtask sliceごとに再実行して"
        "`/.../{task}`系列を生成する。",
        "worker micro-batchではstringを送らず`task_ids` maskへ復元する。batch全体へbroadcast済みのsuccess rateは"
        "sliceしてもtask値にならないため除外し、env managerが直接出すtask別値を使う。",
        annotation_type="tensor_shape",
    ),
    entry(
        "C-metrics-trajectory-tokens",
        "C",
        "verl/trainer/ppo/metric_utils.py",
        165,
        210,
        "1 rowは1 environment turnなのでresponse maskのsumはturn token数である。"
        "`traj_uid`が同じrowを`np.bincount`で合算し、sample/trajectory単位の生成token数へ変換する。",
        "これによりtaskごとのturn数差があるmultitask rolloutでも、throughputをrow数ではなく実際の生成token量で比較できる。",
        annotation_type="tensor_shape",
    ),
    entry(
        "D-alfworld-ray-env",
        "D",
        "agent_system/environments/env_package/alfworld/envs.py",
        142,
        231,
        "各ALFWorld instanceをRay actorに隔離し、step/resetを全actorへ`.remote()`投入した後、"
        "単一`ray.get(futures)`でdriverへ集約する。これはenvironment IPCの待機でGPU collectiveではない。",
        "worker seedは`seed + group index`なので同じGRPO group内の複数trajectoryは同じ初期task scheduleを共有し、"
        "action差による結果を比較できる。admissible commandsは次turnの合法action制約として保持する。",
        annotation_type="execution_context",
    ),
    entry(
        "D-alfworld-resume-schedule",
        "D",
        "agent_system/environments/env_package/alfworld/envs.py",
        88,
        140,
        "resume時はTextWorld wrapper chainからseeded game-file iteratorを探索し、実際のgame loadをせずgeneratorだけを進める。"
        "1 manager resetが各workerで1 gameを消費するという不変条件に依存する。",
        "内部attributeがversion差で見つからない、またはiteratorが尽きた場合は部分進行を報告し、"
        "復旧を停止せず再現性低下を上位へ通知する。",
        annotation_type="uncertainty",
    ),
    entry(
        "D-search-env-thread-pool",
        "D",
        "agent_system/environments/env_package/search/envs.py",
        26,
        148,
        "Search episodeはseeded scheduleではなく各rowのquestion/ground truth/data_sourceでresetされる。"
        "retriever URLはenvへround-robin割当し、blocking HTTPをthread poolへ逃がす。",
        "`run_until_complete(asyncio.gather(...))`はhost threadsの全結果が揃うまで同期APIを待たせる境界である。"
        "不足row用dummyは固定capacityを満たすだけで、valid maskにより戻りbatchから除外される。",
        annotation_type="synchronization",
    ),
    entry(
        "D-search-api-retry",
        "D",
        "agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py",
        42,
        153,
        "retrieval HTTPはtimeoutとserver 5xx/connection failureを区別し、bounded retryを行う。"
        "`WAIT_FOR_SERVICE`有効時のconnection/timeoutだけはservice起動待ちとして無期限retryになり得る。",
        "4xx、JSON decode、その他例外は即座にerror resultへ変換する。sleepはこのhost threadだけを止め、"
        "他envのthreadやGPU実行を直接同期しない。",
        annotation_type="error_condition",
    ),
    entry(
        "D-search-cache-session",
        "D",
        "agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py",
        163,
        259,
        "全Search envでbase URL別`requests.Session`を共有してconnection poolを再利用し、"
        "`(URL, topk, query)`をkeyにbounded LRUで同一retrieval結果を共有する。",
        "session poolとquery cacheは別lockで保護される。cache hitはHTTPを完全に省くが、"
        "retriever indexがrun中に変わらないことを前提とするため実験中のindex更新時は無効化が必要である。",
        annotation_type="experimental",
    ),
    entry(
        "D-webshop-reward",
        "D",
        "agent_system/environments/env_package/webshop/envs.py",
        24,
        63,
        "workerはWebShopの連続scoreを`task_score`へ保存した上で、学習rewardを成功時10・それ以外0へ再定義する。"
        "したがって評価用元scoreとpolicy optimization用sparse rewardを混同しない。",
        "resetは指定session/goal indexを開き、available actionsをinfoへ同梱して追加IPCなしで次action制約を渡す。",
        annotation_type="experimental",
    ),
    entry(
        "D-webshop-goal-schedule",
        "D",
        "agent_system/environments/env_package/webshop/envs.py",
        87,
        215,
        "WebShop vector envはgoalをseeded NumPy RNGで選び、同じgroupの`group_n` workerへ同一goal indexをrepeatする。"
        "step/resetのRay taskは全workerへ投入後に`ray.get`でdriverへ合流する。",
        "`fast_forward`はworkerをresetせず同じ`rng.choice`だけをcompleted step数分再生し、"
        "checkpoint後の次goalを中断なしrunと一致させる。goal poolやenv_numが変わればこの再現性は成立しない。",
        annotation_type="synchronization",
    ),
]

# これらはユーザー提供の詳細解説から注釈mapを再構築しているため、
# 旧CURATED blockを再適用すると同義コメントが重複する。
ATTACHMENT_OVERRIDE_PATHS = {
    "agent_system/multi_turn_rollout/rollout_loop.py",
    "verl/trainer/main_ppo.py",
    "verl/trainer/ppo/core_algos.py",
    "verl/trainer/ppo/ray_trainer.py",
    "verl/trainer/ppo/skillsd_ray_trainer.py",
    "verl/protocol.py",
    "verl/utils/seqlen_balancing.py",
    "verl/workers/actor/dp_actor.py",
    "verl/workers/fsdp_workers.py",
}


def main() -> None:
    rows = load_annotation_map()
    active_curated = [
        row for row in CURATED
        if row["original_path"] not in ATTACHMENT_OVERRIDE_PATHS
    ]
    curated_ids = {row["curated_id"] for row in active_curated}
    rows = [row for row in rows if row.get("curated_id") not in curated_ids]
    # 新しいblock解説が覆う旧legacy注釈は、同義コメントの連続を避けるため置換する。
    covered_ranges: dict[str, list[tuple[int, int]]] = {}
    for curated in active_curated:
        covered_ranges.setdefault(curated["original_path"], []).append(
            (curated["source_start_line"], curated["source_end_line"])
        )
    rows = [
        row
        for row in rows
        if row.get("curated_id")
        or not any(
            start <= row["source_start_line"] <= end
            for start, end in covered_ranges.get(row["original_path"], [])
        )
    ]
    rows.extend(active_curated)
    save_annotation_map(rows)
    print(f"curated_total={len(active_curated)}")
    for priority in ("A", "B", "C", "D"):
        print(
            f"priority_{priority}="
            f"{sum(row.get('priority') == priority for row in active_curated)}"
        )


if __name__ == "__main__":
    main()
