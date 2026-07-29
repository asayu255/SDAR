.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
PPO Ray Trainer
===============

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
We implement the RayPPOTrainer, which is a trainer runs on the driver
process on a single CPU/GPU node (default is CPU).

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The PPORayTrainer include 3 core functions for data preparation,
WorkerGroup initialization and PPO training loop.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Data Preparation
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
----------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The ``PPORayTrainer``, as a single process, is responsible for loading a
complete batch of samples (prompts) from the dataset and then dispatch
to different worker_groups running on different GPUs.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
To generalize the data loading, we implement the ``RLHFDataset`` class
to load the preprocessed parquet files, apply chat templates to the
prompts, add padding, truncate prompts that exceed max prompt length and
then tokenize.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   self.train_dataset = RLHFDataset(data_files=self.config.data.train_files,
                                       tokenizer=self.tokenizer,
                                       config=self.config.data)

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Then, the dataloader will iterate the dataset under PPO mini batch size.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
WorkerGroup Initialization
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
--------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
We first introduce a basic implementation of initializing the
``WorkerGroup`` of the actor model on a given set of GPUs.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
   resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
                                   use_gpu=True,
                                   max_colocate_count=1)
   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # define actor rollout cls to be init on remote
   actor_rollout_cls = RayClassWithInitArgs(cls=ActorRolloutWorker)
   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # define actor_rollout worker group
   actor_rollout_worker_group = MegatronRayWorkerGroup(resource_pool=resource_pool,
                                                       ray_cls_with_init=actor_rollout_cls,
                                                       default_megatron_kwargs=config.actor_rollout.megatron)

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Different WorkerGroups, like ``actor_rollout_worker_group`` ,
``critic_worker_group`` and ``ref_worker_group`` lies on a separate
process in the above implementation.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The driver process can then call the distributed compute function within
the ``actor_rollout_worker_group`` and other roles to construct the RL
training loop.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
For models colocated in the same set of GPUs, we further provide a
fine-grain optimization, which merge the ``worker_group`` of different roles
in the same process. This optimization can save the redundant
CUDA/distributed context in different processes.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # initialize WorkerGroup
   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # See TODO(url) for more information.
   all_wg = {}
   for resource_pool, class_dict in self.resource_pool_to_cls.items():
       worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
       wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
       spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
       all_wg.update(spawn_wg)

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   if self.use_critic:
       self.critic_wg = all_wg['critic']
       self.critic_wg.init_model()

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   if self.use_reference_policy:
       self.ref_policy_wg = all_wg['ref']
       self.ref_policy_wg.init_model()

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   if self.use_rm:
       self.rm_wg = all_wg['rm']
       self.rm_wg.init_model()

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
   self.actor_rollout_wg = all_wg['actor_rollout']
   self.actor_rollout_wg.init_model()

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. note:: For megatron backend, if we merge the ``worker_groups`` into the same processes, all the roles will utilize the same 3D parallel size. To optimize this, we may need to maintain several 3D process groups for each role in the same distributed context. If you want to use different 3D parallel size for different roles, please follow the similar architecture of the first code block to initialize each role's ``worker_group``


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
PPO Training Loop
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
We implement the PPO training loop by calling the functions in
worker_group of each role. The input and output data of each function is
a ``DataProto`` object implemented in `protocol.py <https://github.com/volcengine/verl/blob/main/verl/protocol.py>`_. In the training
loop, trainer will dispatch/collect the data to/from different GPUs
following the transfer protocols wrapped in the workers' functions. The
computation of PPO micro batches is processed in ``update_actor`` and
``update_critic`` functions.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
To extend to other RLHF algorithms, such as DPO, GRPO, please refer to
:doc:`../advance/dpo_extension`.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   def fit(self):
       """
       The training loop of PPO.
       The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
       The light-weight advantage computation is done on the driver process.
       """
       from verl.utils.tracking import Tracking
       from omegaconf import OmegaConf

       .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
       logger = Tracking(project_name=self.config.trainer.project_name,
                           experiment_name=self.config.trainer.experiment_name,
                           default_backend=self.config.trainer.logger,
                           config=OmegaConf.to_container(self.config, resolve=True))

       .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
       global_steps = 0

       .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
       # perform validation before training
       .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
       # currently, we only support validation using the reward_function.
       if self.val_reward_fn is not None:
           val_metrics = self._validate()
           pprint(f'Initial validation metrics: {val_metrics}')

       .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
       for epoch in range(self.config.trainer.total_epochs):
           for batch_dict in self.train_dataloader:
               metrics = {}

               .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
               batch: DataProto = DataProto.from_single_dict(batch_dict)
               .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
               # batch = batch.to('cuda')

               .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
               # pop those keys for generation
               gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

               .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
               # generate a batch
               with Timer(name='gen', logger=None) as timer:
                   gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
               metrics['timing/gen'] = timer.last

               .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
               batch = batch.union(gen_batch_output)

               .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
               if self.use_reference_policy:
                   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
                   # compute reference log_prob
                   with Timer(name='ref', logger=None) as timer:
                       ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                       batch = batch.union(ref_log_prob)
                   metrics['timing/ref'] = timer.last

               .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
               # compute values
               with Timer(name='values', logger=None) as timer:
                   values = self.critic_wg.compute_values(batch)
                   batch = batch.union(values)
               metrics['timing/values'] = timer.last

               .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
               with Timer(name='adv', logger=None) as timer:
                   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
                   # compute scores. Support both model and function-based.
                   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
                   # We first compute the scores using reward model. Then, we call reward_fn to combine
                   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
                   # the results from reward model and rule-based results.
                   if self.use_rm:
                       .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
                       # we first compute reward model score
                       reward_tensor = self.rm_wg.compute_rm_score(batch)
                       batch = batch.union(reward_tensor)

                   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
                   # we combine with rule-based rm
                   reward_tensor = self.reward_fn(batch)
                   batch.batch['token_level_scores'] = reward_tensor

                   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
                   # compute rewards. apply_kl_penalty if available
                   batch, kl_metrics = apply_kl_penalty(batch,
                                                           kl_ctrl=self.kl_ctrl_in_reward,
                                                           kl_penalty=self.config.algorithm.kl_penalty)
                   metrics.update(kl_metrics)

                   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
                   # compute advantages, executed on the driver process
                   batch = compute_advantage(batch,
                                               self.config.algorithm.gamma,
                                               self.config.algorithm.lam,
                                               adv_estimator=self.config.algorithm.adv_estimator)
               metrics['timing/adv'] = timer.last

               .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
               # update critic
               if self.use_critic:
                   with Timer(name='update_critic', logger=None) as timer:
                       critic_output = self.critic_wg.update_critic(batch)
                   metrics['timing/update_critic'] = timer.last
                   critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                   metrics.update(critic_output_metrics)

               .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
               # implement critic warmup
               if self.config.trainer.critic_warmup <= global_steps:
                   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
                   # update actor
                   with Timer(name='update_actor', logger=None) as timer:
                       actor_output = self.actor_rollout_wg.update_actor(batch)
                   metrics['timing/update_actor'] = timer.last
                   actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                   metrics.update(actor_output_metrics)

               .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
               # validate
               if self.val_reward_fn is not None and (global_steps + 1) % self.config.trainer.test_freq == 0:
                   with Timer(name='testing', logger=None) as timer:
                       val_metrics: dict = self._validate()
                       val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
                   metrics['timing/testing'] = timer.last
                   metrics.update(val_metrics)

               .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
               # collect metrics
               data_metrics = compute_data_metrics(batch=batch)
               metrics.update(data_metrics)

               .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
               # TODO: make a canonical logger that supports various backend
               logger.log(data=metrics, step=global_steps)

               .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
               if self.config.trainer.save_freq > 0 and (global_steps + 1) % self.config.trainer.save_freq == 0:
                   actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                                   f'global_step_{global_steps}')
                   actor_remote_path = os.path.join(self.config.trainer.default_hdfs_dir, 'actor')
                   self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

                   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
                   if self.use_critic:
                       critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                                           f'global_step_{global_steps}')
                       critic_remote_path = os.path.join(self.config.trainer.default_hdfs_dir, 'critic')
                       self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

               .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
               global_steps += 1

       .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
       # perform validation after training
       if self.val_reward_fn is not None:
           val_metrics = self._validate()
           pprint(f'Final validation metrics: {val_metrics}')
