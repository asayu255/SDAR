.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Using Checkpoints to Support Fault Tolerance Training
=====================================================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
There could be training errors or machine failure during the whole RLHF training process, 
so it is recommended to enable checkpoints to minimize your loss.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The API Interface has already been listed in :ref:`config-explain-page`,
and we will not repeat them. But there are still some technique details
we hope to clarify.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. note:: 

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    Notice that the ``checkpoint.contents`` field has no effect to FSDP checkpoint except ``hf_model``, 
    the other 3 fields are binded together to save and load. We recommend to include ``model``, ``optimizer`` and ``extra`` all.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Checkpoint Saving Directory Structure
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-------------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Commonly, we use the ``default_local_dir`` declared in ``ppo_trainer.yaml`` or ``ppo_megatron_trainer.yml``
to work as preffix when saving checkpoints, which is ``checkpoints/${trainer.project_name}/${trainer.experiment_name}``.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
So the inner checkpoint structure of **FSDP** is like:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code::

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    │   ├── actor_huggingface
    │   ├── critic
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    │   └── critic_huggingface
    └── latest_checkpointed_iteration.txt

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
All model shards, optimizers and extra states are stored togather, in a sharded and distributed way.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
While **Megatron** current checkpoint structure is:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code::

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
    │   │   ├── huggingface     # default save tokenizer, save huggingface model if include ``hf_mode`` in checkpoint.contents
    │   │   ├── model           # save sharded model, naming the same as Megatron
    │   │   │   ├── mp_rank_xx_yyy          # xx is tp_rank in 2 digits, yyy is pp_rank in 3 digits
    │   │   │   │   └── model_states.pt
    │   │   │   └── mp_rank_xx_xxx
    │   │   ├── optim
    │   │   │   └── distrib_optim_pp{a}_tp{b}_cp{c}_dp{d}.pt
    │   │   └── rng_states
    │   └── critic
    │   │   ├── huggingface
    │   │   ├── model
    │   │   ├── optim
    │   │   └── rng_states
    └── latest_checkpointed_iteration.txt

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Convert FSDP and Megatron Checkpoints to HuggingFace Format Model
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----------------------------------------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
We provide a tool to convert the FSDP and Megatron checkpoints to HuggingFace format model.
The tool is located in ``scripts/model_merger.py``.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The script supports two main sub-commands: `merge` (to convert and save checkpoints) and `test` (to validate merged checkpoints against a reference model).
The arguments for the `merge` sub-command are as follows:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    usage: model_merger.py merge [-h] --backend {fsdp,megatron} --local_dir LOCAL_DIR [--hf_model_path HF_MODEL_PATH]
                                [--tie-word-embedding] [--is-value-model] [--target_dir TARGET_DIR]
                                [--hf_upload_path HF_UPLOAD_PATH] [--private]

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    options:
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    -h, --help            show this help message and exit
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    --backend {fsdp,megatron}
                            The backend of the model
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    --local_dir LOCAL_DIR
                            Path to the saved model checkpoints
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    --hf_model_path HF_MODEL_PATH
                            (Deprecated) Path to the original Hugging Face model for config.
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    --tie-word-embedding  Whether to tie word embedding weights (currently only Megatron supported)
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    --is-value-model      Whether the model is a value model (currently only Megatron supported)
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    --target_dir TARGET_DIR
                            Directory to save the merged huggingface model
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    --hf_upload_path HF_UPLOAD_PATH
                            Hugging Face repository ID to upload the model
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    --private             Whether to upload the model to a private Hugging Face repository

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Example usage for merging Megatron checkpoints:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    python scripts/model_merger.py merge \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --backend megatron \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --tie-word-embedding \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --target_dir /path/to/merged_hf_model

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Example usage for merging FSDP checkpoints:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    python scripts/model_merger.py merge \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --backend fsdp \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --local_dir checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --target_dir /path/to/merged_hf_model


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Megatron Merger details
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Current implement of decoder layers uses ``nn.ModuleList`` to store the layers, 
and thus the model layers on every PP rank and VPP rank starts their index from 0.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
There are 3 ways to correct this behavior:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Modify the decoder layer's state_dict, add ``offset`` to each layer's index, thus rewrite ``nn.ModuleList`` implementation.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Modify the layer index when saving checkpoint and recover them when loading checkpoint.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. The Checkpoint merger do this work, calculate the actual ``offset`` from ``state_dict`` only, a little complex.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Current implementation use solution 2.


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
HuggingFace to Megatron DistCheckpoint details
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
----------------------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
If your model is quite huge, we recommend you to use Megatron dist-checkpoint to load the model.
Megatron dist-checkpoint supports loading with different kinds of model parallelism,
and it is much faster than the original checkpoint loading.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
To convert original HuggingFace model to Megatron dist-checkpoint,
you can use the ``scripts/converter_hf_to_mcore.py`` script. Large MoE models are temporarily supported with CPU initialization,
which is a little slower. While we are working on a better solution to support large models.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Example command to convert the model is as follows:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    python scripts/converter_hf_to_mcore.py \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --hf_model_path Qwen/Qwen1.5-MoE-A2.7B-Chat \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --output_path /mnt/disk/Qwen/Qwen1.5-MoE-A2.7B-Chat \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --use_cpu_initialization    # Only work for MoE models


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Original Checkpoint Utils
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Original Checkpoint Utils refer to original checkpoint implementation in ``verl/models/[model]/megatron/checkpoint_utils``.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
We only need ``[model]_loader.py`` in original checkpoint utils now, since we get rid of storing ``hf_model`` every time (which is not recommended for large model training, try only saving sharded models if you can).

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. note:: 

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    Note that ``[model]_loader`` only support environments where **storage clusters are able to connect with every calculation nodes**. 
    Because it utilizes **sharded load way to minimize the loading checkpoint overhead**. 
    Every rank loads its own data from ``state_dict`` which can be accessed by all of them.
    While there is also no need to broadcast among DP ranks, since the saved state_dict is only produced by DP rank 0.

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    For users who can **only place the huggingface model on one device**, we keep the original costly implementation in ``[model]_loader_deprecated``. In this implementation, rank 0 broadcast all weights to each tp and pp rank, and then dp rank 0 broadcast to all dp ranks. There may be at risks of OOM.

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    To use deprecated loader, change the import package of ``load_state_dict_to_megatron_llama``.
