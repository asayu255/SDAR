.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
verl x Ascend
===================================


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
我们在 verl 上增加对华为昇腾设备的支持。

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
硬件支持
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Atlas 200T A2 Box16

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Atlas 800T A2


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
安装
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
基础环境准备
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+-----------+-------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| software  | version     |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+-----------+-------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| Python    | == 3.10     |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+-----------+-------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| CANN      | == 8.1.RC1  |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+-----------+-------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| torch     | == 2.5.1    |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+-----------+-------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| torch_npu | == 2.5.1.RC1|
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+-----------+-------------+


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
vllm & vllm-ascend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
为了能够在 verl 中正常使用 vllm，需使用以下命令编译安装 vllm 和 vllm-ascend。请注意根据机器类型区分安装方式。

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash
    
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # vllm
    git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm.git
    cd vllm
    pip install -r requirements-build.txt

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # for Atlas 200T A2 Box16
    VLLM_TARGET_DEVICE=empty pip install -e . --extra-index https://download.pytorch.org/whl/cpu/
    
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # for Atlas 800T A2
    VLLM_TARGET_DEVICE=empty pip install -e .

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash
    
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # vllm-ascend
    git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    export COMPILE_CUSTOM_KERNELS=1
    python setup.py install

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
安装verl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    git clone https://github.com/volcengine/verl.git
    cd verl
    pip install -r requirements-npu.txt
    pip install -e .

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
其他三方库说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+--------------+---------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| software     | description   |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+--------------+---------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| transformers | >= v4.52.0    |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+--------------+---------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| flash_attn   | not supported |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+--------------+---------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| liger-kernel | not supported |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+--------------+---------------+

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. 支持通过 transformers 使能 --flash_attention_2， transformers 需大于等于 4.52.0版本。
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. 不支持通过 flash_attn 使能 flash attention 加速。
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. 不支持 liger-kernel 使能。


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
快速开始
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----------------------------------
正式使用前，建议您通过对Qwen2.5-0.5B GRPO的训练尝试以检验环境准备和安装的正确性。

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1.下载数据集并将数据集预处理为parquet格式，以便包含计算RL奖励所需的必要字段

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2.执行训练

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    set -x

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    export VLLM_ATTENTION_BACKEND=XFORMERS

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$HOME/data/gsm8k/train.parquet \
        data.val_files=$HOME/data/gsm8k/test.parquet \
        data.train_batch_size=128 \
        data.max_prompt_length=512 \
        data.max_response_length=128 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='verl_grpo_example_gsm8k' \
        trainer.experiment_name='qwen2_7b_function_rm' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=5 \
        trainer.total_epochs=1 \
        trainer.device=npu $@


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
支持现状
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+-----------+----------------------+-------------+-------------------+----------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| algorithm |         model        | rewards mae |  throughput ratio |        hardware      |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+-----------+----------------------+-------------+-------------------+----------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
|   GRPO    | Qwen2.5-7B-instruct  |    0.38%    |        0.588      |  Atlas 200T A2 Box16 |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+-----------+----------------------+-------------+-------------------+----------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
|   GRPO    | Qwen2.5-32B-instruct |    0.30%    |        0.685      |  Atlas 200T A2 Box16 |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+-----------+----------------------+-------------+-------------------+----------------------+

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
目前支持 Qwen2.5 的 GRPO 训练，Qwen2.5-VL GRPO 训练在 vllm-ascend 的修复后支持，涉及到的issue为：

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. `issues#809 <https://github.com/vllm-project/vllm-ascend/issues/809>`_

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. `issues#825 <https://github.com/vllm-project/vllm-ascend/issues/825>`_


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
精度对比说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
对于 SFT 类算法，我们期望在相同配置下华为昇腾设备与 A100 的 loss 平均绝对误差<= 2%。计算方式如下图。更多信息请参考 `精度计算说明 <https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/LMaccuracy_0001.html>`_。

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/loss_comparison.png?raw=true
   :alt: loss_comparison

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
根据经验，对于 GRPO 等 RL 类算法，我们期望在相同配置下华为昇腾设备与 A100 的 rewards 平均绝对误差<= 4%，计算方式参考上图。


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
吞吐对比说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Ascend npu 和 A100 分别取日志中前4个 step 的 "perf/throughput" 做平均， throughput ratio = npu 平均值 / A100 平均值。 



.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
计划
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
查看 `roadmap <https://github.com/volcengine/verl/discussions/900>`_ 获取更多特性的支持进度。



.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
声明
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----------------------------------
verl中提供的ascend支持代码皆为参考样例，商业使用请通过官方正式途径沟通，谢谢。