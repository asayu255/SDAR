.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
SGLang Backend
==============
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
**Authored By SGLang RL Team and listed alphabetically by last name**

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
`Jingyi Chen <https://github.com/fzyzcjy>`_, `Yitong Guan <https://github.com/minleminzui>`_, `Zhuobin Huang <https://zobinhuang.github.io/sec_about/>`_, `Jiajun Li <https://github.com/guapisolo>`_, `Ji Li <https://github.com/GeLee-Q>`_, `Shenggui Li <https://franklee.xyz/about>`_, `Junrong Lin <https://github.com/ocss884>`_, `Xiang Long <https://github.com/SwordFaith>`_, `Rui Lu <https://scholar.google.com/citations?user=-MGuqDcAAAAJ>`_, `Jin Pan <https://jhinpan.github.io/>`_, `Shuai Shi <https://github.com/shuaills>`_, `Yushen Su <https://yushengsu-thu.github.io/>`_, `Xinyuan Tong <https://github.com/JustinTong0323>`_, `Chendong Wang <https://github.com/cedricbeta>`_, `Hanchen Zhang <https://scholar.google.com/citations?user=pGcJcagAAAAJ>`_, `Haoran Wang <https://ubecc.github.io/about/>`_, `Yongan Xiang <https://github.com/BearBiscuit05>`_, `Chengxing Xie <https://yitianlian.github.io/>`_, `Yuhao Yang <https://github.com/yhyang201>`_, `Jinwei Yao <https://kivi-yao.github.io/>`_, `Qiaolin Yu <https://github.com/Qiaolin-Yu>`_, `Yuzhen Zhou <https://github.com/zyzshishui>`_, `Chenyang Zhao <https://github.com/zhaochenyang20>`_



.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Introduction
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------
`SGLang <https://github.com/sgl-project/sglang>`_ is an open-source state-of-the-art inference service engine, fully adopted by xAI to support all inference needs of Grok during research and serving processes.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Currently, verl fully supports using SGLang as the inference engine during the rollout phase. As a rollout engine, SGLang provides the same feature coverage as vLLM., including memory saving and multi-node rollout features. After installing verl and SGLang, simply add ``actor_rollout_ref.rollout.name=sglang`` at startup script to seamlessly switch between the two inference frameworks.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
In addition, the SGLang team is actively working on supporting features such as Multi-Turn Agentic RL, VLM RLHF, Server-Based RLHF, and Partial Rollout. You can track the related development progress in the `Tracking Roadmap <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/issues/74>`_.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Installation
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------
Please always follow the following command to install SGLang with verl. 

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash
    
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    pip install --upgrade pip
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Currently 0.4.6.post5, subject to updates at any time, please refer to the latest version specified in `setup.py`
    pip install -e ".[sglang]"

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
You can check the following dependencies are in your environment:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. note::

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **PyTorch**: 2.6.0+cu124
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **CUDA**: 12.4
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **flashinfer-python**: 0.2.5+cu124torch2.6
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **sgLang**: 0.4.6.post5
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **sgl-kernel**: 0.1.4

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Using SGLang as the Inference Backend for PPO Training on a Single Machine
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-------------------------------------------------------------------------
We use Qwen/Qwen2-7B-Instruct on the gsm8k dataset for a simple test.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Run the following command to prepare the gsm8k dataset:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    python3 examples/data_preprocess/gsm8k.py

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Run the following script to conduct a PPO experiment on a single machine with 4 GPUs:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        data.train_files=$HOME/data/gsm8k/train.parquet \
        data.val_files=$HOME/data/gsm8k/test.parquet \
        data.train_batch_size=4096 \
        data.max_prompt_length=4096 \
        data.max_response_length=4096 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        critic.optim.lr=1e-5 \
        critic.model.path=Qwen/Qwen2-7B-Instruct \
        critic.ppo_micro_batch_size_per_gpu=4 \
        critic.model.fsdp_config.param_offload=True \
        critic.model.fsdp_config.optimizer_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=['console'] \
        trainer.val_before_train=False \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=15 2>&1 | tee verl_demo.log

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Why export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. ``verl`` initializes a ``SGLangRollout`` module during rollout, which is used to evaluate/generate samples.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. ``SGLangRollout`` will initialize ``Engine``, and further initialize a ``torch.distributed.DeviceMesh``, used to support Tensor Parallel (TP).

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. ``DeviceMesh.init()`` internally checks the free GPU memory of all participating devices. If the difference is too large (more than ~10%), it directly reports an error to avoid initialization failures or deadlocks.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Why might there be inconsistent GPU memory?
"""""""""""""""""""""""""""""""""""""""""""

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
**1. Ray Distributed Actor loads the model at different times**

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
``verl`` uses Ray-based multi-process, multi-GPU concurrent training. Each ``WorkerDict`` may be called at different times:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: python

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    self.rollout = SGLangRollout(...)

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Different workers initialize the model at different times → different memory usage.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
**2. Delayed initialization causes memory bias**

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Some workers start model loading/inference (e.g., ``generate_sequences()``, ``compute_log_prob()``) earlier than others.  
Early workers already use up GPU memory → late workers still have empty memory → memory difference appears.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
**3. SGLang's TP init uses "all-device broadcast", but there's no uniform release timing**

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Although ``SGLangRollout`` may only involve subset of GPUs, its ``Engine`` initialization calls ``torch.distributed.init_process_group()`` and broadcasts weights, so:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- Non-rollout GPUs also join the communication.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- Later on, ``DeviceMesh`` init will fail due to "inconsistent memory".

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
**4. Different FSDP/TP loading behaviors also lead to mismatch**

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
If using:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    actor.fsdp_config.param_offload=True  
    ref.fsdp_config.param_offload=True

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Then some workers keep params on CPU while others already sharded to GPU → leads to asymmetric memory layout.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Using SGLang as the Inference Backend for PPO Training Across Multiple Machines
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------------------------------------------------------------------------
SGLang also supports running verl's RAY-based cross-machine inference in IPv4 and IPv6 scenarios. In the script below, we use TP=16 for cross-machine inference. Suppose we have two interconnected machines: node0 with IP 10.94.16.4 and node1 with IP 10.94.16.5.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Start Ray on node0:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    ray start --head --dashboard-host=0.0.0.0

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
You will see the following prompt:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    Usage stats collection is enabled. To disable this, add `--disable-usage-stats` to the command that starts the cluster, or run the following command: `ray disable-usage-stats` before starting the cluster. See https://docs.ray.io/en/master/cluster/usage-stats.html for more details.

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    Local node IP: 10.94.16.4

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    --------------------
    Ray runtime started.
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    --------------------

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    Next steps
    To add another node to this Ray cluster, run
        ray start --address='10.94.16.4:6379'

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Have node1 join the Ray cluster:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Run the following command on node1:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    ray start --address='10.94.16.4:6379'

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Run the following command to confirm that the Ray cluster now has two nodes:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    ray status

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
You can see that the cluster has two nodes with 16 GPUs:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    ======== Autoscaler status: 2025-04-09 09:25:37.694016 ========
    Node status
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    ---------------------------------------------------------------
    Active:
     1 node_ef382ffd687d8f6b060c1b68e63ada7341b936fe5b1901dd04de1027
     1 node_1eb4d7d07e793114c23a89d1a41f1f76acf6ef5b35af844a4ee8e4ba
    Pending:
     (no pending nodes)
    Recent failures:
     (no failures)

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    Resources
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    ---------------------------------------------------------------
    Usage:
     0.0/360.0 CPU
     0.0/16.0 GPU
     0B/3.39TiB memory
     0B/372.53GiB object_store_memory

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. Run the following script to train meta-llama/Llama-3.1-8B-Instruct with TP=16 across 2 machines using 16 GPUs:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    DATA_DIR=$HOME/data/gsm8k

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    python3 -m verl.trainer.main_ppo \
        actor_rollout_ref.rollout.name=sglang \
        data.train_files=$DATA_DIR/train.parquet \
        data.val_files=$DATA_DIR/test.parquet \
        data.train_batch_size=4096 \
        data.max_prompt_length=4096 \
        data.max_response_length=4096 \
        actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B-Instruct \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=16 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path=meta-llama/Llama-3.1-8B-Instruct \
        critic.model.enable_gradient_checkpointing=True \
        critic.ppo_micro_batch_size=16 \
        critic.model.fsdp_config.param_offload=True \
        critic.model.fsdp_config.optimizer_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.val_before_train=True \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=15 2>&1 | tee verl_demo.log
