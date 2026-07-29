.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
RL(HF) algorithms with LoRA Support
===========================================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
We support LoRA (Low-Rank Adaptation) for reinforcement learning algorithms such as PPO, GRPO, and others.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
LoRA is a parameter-efficient fine-tuning technique that injects trainable low-rank matrices into pre-trained weights (typically linear layers). This reduces memory footprint and compute cost, making it possible to fine-tune large models with limited hardware.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The benefits this brings include:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- reinforcement learning with very large models (e.g. 70B+) with modest hardware (e.g. 8x80G GPUs),
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- enable larger batch sizes due to reduced memory usage,
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- simplify model transfer and deployment, as only LoRA adapters need to be saved,
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- Combine with techniques like `SLoRA <https://arxiv.org/abs/2311.03285>`_ or `CCoE <https://arxiv.org/abs/2407.11686>`_ to serve multiple LoRA adapters efficiently

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
This guide explains how to enable LoRA in RL training and configure related parameters.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Usage Guide
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------------------
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Lora is available in the `verl.trainer.ppo.ray_trainer.RayPPOTrainer`. Examples are provided via the `verl.trainer.main_ppo` entry point.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Currently, LoRA is supported via huggingface peft, only with fsdp and vllm backend (sglang support coming soon).

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- `strategy=fsdp`
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- `rollout.name=vllm`

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. Required configurations for LoRA:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- `actor_rollout_ref.model.lora_rank`: int, set to a reasonable value greater than 0 (e.g., 8, 16, 32, 64)
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- `actor_rollout_ref.model.lora_alpha`: float, the alpha term in LoRA
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- `actor_rollout_ref.rollout.load_format="safetensors"`: required. This enables vLLM to load the base model.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- `actor_rollout_ref.model.target_modules`: the target modules for LoRA. Typically set to "all-linear".

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
4. Recommend options:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- `actor_rollout_ref.model.use_shm=True`: preload the model into `/dev/shm` to improve model loading speed.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- `actor_rollout_ref.rollout.layered_summon=True`: this enables the actor-model to gather the FSDP shards per layers when synchronizing the LoRA Adapter to vLLM, thereby reducing GPU peak memory. Recommended if the model is very large (70B+) or the GPU memory is limited (< 48GB)


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Best Practices and Notes
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. **Learning rate**: it is recommended to increase the value of learning rate by an order of magnitude.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. **LoRA Rank**:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- Too small a rank can hurt convergence.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- LoRA rank recommendation from @thelongestusernameofall:

  .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
  - A very small lora_rank can lead to slower convergence or worse training performance. It is recommended to set lora_rank to be>=32. Tests have shown that for a 0.5B model, with lora_rank=32,the training convergence speed and final performance are almost identical to non-LoRA training
  .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
  - For a 32B model,with lora_rank=128,the training convergence speed and final performance are also almost identical to non-LoRA training.
  .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
  - More comprehensive reference results are coming soon.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. image:: https://github.com/eric-haibin-lin/verl-community/blob/f2b80b8b26829124dd393b7a795a0640eff11644/docs/lora.jpg?raw=true

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. Reference configuration for RL training with the Qwen2.5-72B model using 8 x 80GB GPUs (increase lora_rank if needed):

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block::

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    data.train_batch_size=64 \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Example Script
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
For an end-to-end example, refer to the script below:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh
