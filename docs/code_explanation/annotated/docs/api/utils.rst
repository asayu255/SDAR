.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Utilities
============

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
This section documents the utility functions and classes in the VERL library.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Python Functional Utilities
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. automodule:: verl.utils.py_functional
   :members: append_to_dict

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
File System Utilities
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. automodule:: verl.utils.fs
   :members: copy_to_local

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Tracking Utilities
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
---------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. automodule:: verl.utils.tracking
   :members: Tracking

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Metrics Utilities
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
---------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. automodule::  verl.utils.metric
   :members: reduce_metrics

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Checkpoint Management
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. automodule:: verl.utils.checkpoint.checkpoint_manager
   :members: find_latest_ckpt_path

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. automodule:: verl.utils.checkpoint.fsdp_checkpoint_manager
   :members: FSDPCheckpointManager

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Dataset Utilities
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
---------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. automodule:: verl.utils.dataset.rl_dataset
   :members: RLHFDataset, collate_fn

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Torch Functional Utilities
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. automodule:: verl.utils.torch_functional
   :members: get_constant_schedule_with_warmup, masked_whiten, masked_mean, logprobs_from_logits

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Sequence Length Balancing
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
----------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. automodule:: verl.utils.seqlen_balancing
   :members: get_reverse_idx, rearrange_micro_batches

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Ulysses Utilities
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
--------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. automodule:: verl.utils.ulysses
   :members: gather_outpus_and_unpad, ulysses_pad_and_slice_inputs

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
FSDP Utilities
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. automodule:: verl.utils.fsdp_utils
   :members: get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn, load_fsdp_model_to_gpu, load_fsdp_optimizer, offload_fsdp_model_to_cpu, offload_fsdp_optimizer,

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Debug Utilities
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. automodule:: verl.utils.debug
   :members: log_gpu_memory_usage, GPUMemoryLogger

