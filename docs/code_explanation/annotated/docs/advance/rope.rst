.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
RoPE Scaling override
=======================================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Some models such as `Qwen/Qwen2.5-7B-Instruct <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#processing-long-texts>`_ support RoPE Scaling but don't have it defined in their config.json file.
For example, this model supports this configuration:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: python

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    {
        ...,
        "rope_scaling": {
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "type": "yarn"
        }
    }



.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
In order to support a longer context for such models, you must override the model configs when starting the trainer.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
PPO example:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    +actor_rollout_ref.model.override_config.rope_scaling.type=yarn \
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    +actor_rollout_ref.model.override_config.rope_scaling.factor=4.0 \
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=32768 \


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
And for the critic model

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    +critic.model.override_config.rope_scaling.type=yarn \
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    +critic.model.override_config.rope_scaling.factor=4.0 \
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    +critic.model.override_config.rope_scaling.original_max_position_embeddings=32768 \
