.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Resource Needed for verl RL
==============================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Since RL requires more resources compared to regular training, 
determining how much resources are needed to successfully run it before training 
is a relatively difficult task. To provide more people with reference points for 
resource selection when dealing with different models and tasks, this section is 
mainly dedicated to introducing the environmental requirements based on experiments 
we have conducted.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
However, due to limited manpower and equipment resources, we also hope for more 
contributions from the open-source community. When submitting a PR, it is necessary 
to provide a script to be added to the example/tuning scripts.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
We need two types of scripts: one is the configuration that can run with the **minimum 
resources(min)**, and the other is the configuration that runs with **recommended resources(recommended)**. For the former, 
it can be understood as a script that can run after applying all memory optimization techniques 
(e.g., offload, gradient checkpointing). For the latter, it can be understood as a script that 
can run while avoiding operations that incur additional time overhead as much as possible (targetting best throughput).

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
When defining script names, please follow this format: 
``[model]_[task]_[gpunums]_[device]_[train]_[infer].sh``. This will effectively improve 
the script's recognizability. You can place the script under the ``examples/tuning/`` directory.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
If you happen to have a configuration that has already been tested, we welcome you to submit 
a PR and include a screenshot from Wandb or other verifiable evidence.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
----------------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
7B
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2-7B
      - GRPO
      - 2*H800
      - fsdp
      - vllm0.8.2
      - `qwen2-7b_grpo_2_h800_fsdp_vllm <https://github.com/volcengine/verl/blob/main/examples/tuning/7b/qwen2-7b_grpo_2_h800_fsdp_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_
      

14B
~~~

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. list-table::
    :widths: auto
    :header-rows: 1
    
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    * - Tag
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Model
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Task
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Resource
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Train
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Infer
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Link
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Contributor
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    * - MIN
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Qwen2-14B
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - GRPO
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - 4*H800
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - fsdp
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - vllm0.8.2
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - `qwen2-14b_grpo_4_h800_fsdp_vllm <https://github.com/volcengine/verl/blob/main/examples/tuning/14b/qwen2-14b_grpo_4_h800_fsdp_vllm.sh>`_
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - `Xiangyongan <xiangyongan@bytedance.com>`_


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
32B
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2-32B
      - GRPO
      - 8*H20
      - megatron
      - vllm0.8.2
      - `qwen2-32b_grpo_8_h20_megatron_vllm <https://github.com/volcengine/verl/tree/main/examples/tuning/32b/qwen2_32B_grpo_8_h20_megatron_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_

70B
~~~

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. list-table::
    :widths: auto
    :header-rows: 1

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    * - Tag
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Model
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Task
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Resource
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Train
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Infer
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Link
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Contributor
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    * - MIN
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Qwen2-70B
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - GRPO
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - 32*H20
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - fsdp
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - vllm0.8.2
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - `qwen2-70b_grpo_32_h20_fsdp_vllm <https://github.com/volcengine/verl/blob/main/examples/tuning/70b/qwen2-70b_grpo_32_h20_fsdp_vllm.sh>`_
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - `Xiangyongan <xiangyongan@bytedance.com>`_
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    * - MIN
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - Qwen2-70B
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - GRPO
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - 32*H800
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - fsdp
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - vllm0.8.3
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - `qwen2-70b_grpo_32_h800_fsdp_vllm <https://github.com/volcengine/verl/blob/main/examples/tuning/70b/qwen2-70b_grpo_32_h800_fsdp_vllm.sh>`_
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - `Xiangyongan <xiangyongan@bytedance.com>`_

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
405B
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~

.. table::
   :widths: auto

   ====== ====== ====== ======== ====== ====== ======
   tag    model  task   resource train  infer  link
   ====== ====== ====== ======== ====== ====== ======
   \      \      \        \      \      \
   ====== ====== ====== ======== ====== ====== ======


671B
~~~~

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. table::
   :widths: auto

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   ====== ====== ====== ======== ====== ====== ======
   tag    model  task   resource train  infer  link
   ====== ====== ====== ======== ====== ====== ======
   \      \      \        \      \      \
   ====== ====== ====== ======== ====== ====== ======
