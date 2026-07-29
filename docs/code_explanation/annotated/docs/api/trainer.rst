.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Trainer Interface
================================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Trainers drive the training loop. Introducing new trainer classes in case of new training paradiam is encouraged.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. autosummary::
   :nosignatures:

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   verl.trainer.ppo.ray_trainer.RayPPOTrainer


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Core APIs
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~~~

.. autoclass::  verl.trainer.ppo.ray_trainer.RayPPOTrainer
   :members: __init__, init_workers, fit


.. automodule:: verl.utils.tokenizer
   :members: hf_tokenizer


.. automodule:: verl.trainer.ppo.core_algos
   :members: agg_loss, kl_penalty, compute_policy_loss, kl_penalty


.. automodule:: verl.trainer.ppo.reward
   :members: load_reward_manager, compute_reward, compute_reward_async
