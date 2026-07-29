.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Megatron-LM Backend
===================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
We support Megatron Backend by implementing various workers for actor,
critic, reference, rollout and reward models. We also implement the
``3DHybridEngine`` using Megatron-LM and vLLM/SGLang in
`megatron_vllm.py <https://github.com/volcengine/verl/blob/main/verl/workers/sharding_manager/megatron_vllm.py>`_
and `megatron_sglang.py <https://github.com/volcengine/verl/blob/main/verl/workers/sharding_manager/megatron_sglang.py>`_.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
**Pros**

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- Support 5D parallelism (TP, EP, CP, DP, PP) and sequence parallelism
  for best scalablility and throughput.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- 3D HybridEngine can significantly reduce peak memory usage and reduce
  weight synchronize overhead between actor and rollout.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
**Cons**

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- Huggingface Models and Megatron checkpoints need tools for conversion.


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Development Progress
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
--------------------


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Note that [Deprecated] means that the feature is not supported in the latest
version of verl.
[To-Optimize] means that the feature is implemented but not optimized yet.
[WIP] means that the feature is working in progress.
[In-Release] means that the feature is ready and in review process,
coming at any time.


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [Deprecated]  | Megatron 3D Parallelism with custom models                |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [Done]        | Megatron 0.11.0 ``GPTModel`` support                      |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [Done]        | Megatron GRPO support                                     |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [Done]        | Megatron with vLLM 0.8.2, with per-tensor weights loading |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [Done]        | Megatron with Context Parallel                            |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [Done]        | Qwen2MoE model support                                    |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [To-Optimize] | Megatron dist Checkpoint                                  |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [To-Optimize] | Huggingface and Megatron Checkpoint Converter             |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [To-Optimize] | Efficient fused linear, entropy and cross entropy         |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [Done]        | Megatron offload(param, grad, optimizer)                  |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [Done]        | Megatron Profiler                                         |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [In-Release]  | Megatron 0.12.0, TE 2.2 with vLLM 0.8.3 and Fused Attn    |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [WIP]         | Moonlight/DeepSeek-V3 model support                       |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [WIP]         | Expert Parallel support                                   |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [WIP]         | Megatron support dynamic batch size                       |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [To-Do]       | Performance tuning                                        |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
| [MileStone]   | Runnable with DeepSeek-V3 671B post-training              |
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
+---------------+-----------------------------------------------------------+



.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Utils of Megatron Workers
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
MegatronWorker
^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
``MegatronWorker`` is the base class of different megatron worker
classes. In this class, ``get_megatron_global_info`` and
``get_megatron_rank_info`` function to retrive the 3D parallel world
size and rank of each ``Worker`` running on specific GPU. These information
will be used in transfer protocol for Megatron Backend.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The following ``Worker`` class for different models will be utilized to
construct the ``WorkerGroup`` .

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
We implement various of APIs for each ``Worker`` class decorated by the
``@register(dispatch_mode=)`` . These APIs can be called by the ray
driver process. The data can be correctly collect and dispatch following
the ``dispatch_mode`` on each function. The supported dispatch_model
(i.e., transfer protocols) can be found in `decorator.py <https://github.com/volcengine/verl/blob/main/verl/single_controller/base/decorator.py>`_.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
ActorRolloutRefWorker
^^^^^^^^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
This class is implemented for Actor/Rollout HybridEngine or for the
reference model to initialize their model and perform computation.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Actor/Rollout HybridEngine
''''''''''''''''''''''''''

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. HybridEngine, Actor and Rollout initialization API.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   @register(dispatch_mode=Dispatch.ONE_TO_ALL)
   def init_model(self):

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
``ONE_TO_ALL``: when calling the ``init_model`` function from the driver
process, each worker (on a GPU) will execute the following model
initialization process.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The initialization details of HybridEngine, Actor and Rollout are
highlighted below:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. ``AllGatherPPModel`` holds memory buffer for both Actor and Rollout
   and support weight resharding between actor and rollout.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. ``MegatronPPOActor`` implements the simple PPO computation logics
   when the model is built with Megatron, including compute log prob,
   model update.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. ``vLLMRollout`` support generation with vLLM. We modify the vLLM
   Engine and make it executed under SPMD to fit into our
   ``WorkerGroup`` design.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
4. ``MegatronVLLMShardingManager`` a context manager to perform actual
   resharding between actor and rollout.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
See `source code <https://github.com/volcengine/verl/blob/main/verl/workers/megatron_workers.py#L63>`_ for more information.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Initialize the 3D HybridEngine
   hybrid_engine = AllGatherPPModel(model_provider=megatron_actor_model_provider)
   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Fetch the model at current rank
   actor_module = hybrid_engine.this_rank_models
   ...

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # build actor model
   self.actor = MegatronPPOActor(config=self.config.actor,
                                 model_config=self.actor_model_config,
                                 megatron_config=megatron_config,
                                 actor_module=self.actor_module,
                                 actor_optimizer=self.actor_optimizer,
                                 actor_optimizer_config=self.actor_optim_config)

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # build rollout
   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # rollout initialization
   rollout = vLLMRollout(actor_module=params,
                        config=self.config.rollout,
                        tokenizer=self.tokenizer,
                        model_hf_config=self.actor_model_config,
                        train_tp=mpu.get_tensor_model_parallel_world_size())
   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # perform weight resharding between actor and rollout
   sharding_manager = MegatronVLLMShardingManager(module=self.hybrid_engine,
                                                  inference_engine=rollout.inference_engine,
                                                  model_config=self.actor_model_config,
                                                  layer_name_mapping=layer_name_mapping)
   ...

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Generate sequence and recompute log prob

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   @register(dispatch_mode=Dispatch.MEGATRON_PP_AS_DP_PROTO)
   def generate_sequences(self, prompts: DataProto):

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``Dispatch.MEGATRON_PP_AS_DP_PROTO``: The PP dimension of the actor
  model will be regarded as DP dimension. Then the driver process will
  dispatch and collect the data according to this reorganization. This
  is because, in HybridEngine, the actor weight, which usually applied
  larger 3D parallel sizes, will be gathered along the PP dimension and
  TP dimension. Therefore, the corresponding data should be dispatched
  and collected through the 3D parallel group of the rollout model,
  rather than the actor model. However, the world_size and rank
  information can only be retrived from ``get_megatron_global_info`` and
  ``get_megatron_rank_info``, which records the 3D information for the
  actor model. Moreover, the data resharding inside TP dimension will be
  processed within the HybridEngine.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- In this function, the rollout model will perform auto-regressive
  generation and the actor model will recompute the old log prob for the
  generated response.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. Update actor model

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def update_actor(self, data: DataProto):

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``Dispatch.MEGATRON_COMPUTE_PROTO``: User passes the data partitioned
  by DP dimension. The data is dispatched to all tp/pp ranks within the
  same dp group, and ultimately only collects output data from tp=0 and
  the last pp.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- Update the actor model weight using PPO & entropy loss.


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
..note:: 

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   Currently, training Tensor Parallel Size can be different from inference
   Tensor Parallel Size.


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
ReferenceModel
''''''''''''''

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Reference model initialization

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The reference model is initialized using the same function as the actor
model without initializing the HybridEngine and Optimizer. Then the
actor model is also wrapped by the ``MegatronPPOActor``.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Compute reference log prob

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def compute_ref_log_prob(self, data: DataProto):

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- In this function, the reference model will call the compute log prob
  function in ``MegatronPPOActor`` to compute the reference log prob.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
CriticWorker and RewardWorker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Model initialization

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Quite similar to reference model. The CriticWorker will perform
additional initialization for the Optimizer.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Compute Values for CriticWorker

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def compute_values(self, data: DataProto):

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. Update Critic

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def update_critic(self, data: DataProto):

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
4. Compute Reward

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def compute_rm_score(self, data: DataProto):


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Utils of Train Optimization
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
---------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Offload
^^^^^^^
When resources are tight, the offload method can lower GPU memory 
usage, helping training and inference frameworks work well under verl. 
It moves parameters, gradients, and optimizers to CPU memory and only 
loads them back to the GPU when needed.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
If you want to use the offload, you can add the following parameters 
for the actor and ref separately. 

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # For the actor
   actor_rollout_ref.actor.megatron.param_offload=True \
   actor_rollout_ref.actor.megatron.grad_offload=True \
   actor_rollout_ref.actor.megatron.optimizer_offload=True \
   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # For the ref w/o grad and optimizer
   actor_rollout_ref.ref.megatron.param_offload=True \


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
For the critic, you can include these parameters.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # For the critic
   critic.megatron.param_offload=True \
   critic.megatron.grad_offload=True \
   critic.megatron.optimizer_offload=True \

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Profiler
^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The profiler is a tool that helps you understand the performance of your 
model. It can be used to profile the time spent on different operations 
and identify the bottlenecks. You can get more information from 
`torch.profiler <https://pytorch.org/docs/stable/profiler.html>`_.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
In verl, now the profiler is only support for the actor role In Megatron. You can set 
the begin step and end step to profile. Notice, one step means one gradient update. And 
the profile result will be saved in the save_path. If you just want to profile in the 
specific rank, you can set the profile_ranks, by default, it will be [0].

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   actor_rollout_ref.actor.profile.use_profiler=True \
   actor_rollout_ref.actor.profile.profile_ranks=[0] \
   actor_rollout_ref.actor.profile.begin_step=0 \
   actor_rollout_ref.actor.profile.end_step=1 \
   actor_rollout_ref.actor.profile.save_path="./profile"


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Related MCore Document
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
----------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
There is also a detailed document of using MCore to train different
kinds of models, please refer to `MCore Document <https://github.com/volcengine/verl/blob/main/verl/models/mcore/readme.md>`_.