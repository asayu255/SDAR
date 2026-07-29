.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Multi-Modal Example Architecture
=================================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Introduction
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Now, verl has supported multi-modal training. You can use fsdp and 
vllm/sglang to start a multi-modal RL task. Megatron supports is also 
on the way.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Follow the steps below to quickly start a multi-modal RL task.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Step 1: Prepare dataset
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # it will be saved in the $HOME/data/geo3k folder
    python examples/data_preprocess/geo3k.py

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Step 2: Download Model
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
----------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # download the model from huggingface
    python3 -c "import transformers; transformers.pipeline(model='Qwen/Qwen2.5-VL-7B-Instruct')"

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Step 3: Perform GRPO training with multi-modal model on Geo3K Dataset
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
---------------------------------------------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # run the task
    bash examples/grpo_trainer/run_qwen2_5_vl-7b.sh








