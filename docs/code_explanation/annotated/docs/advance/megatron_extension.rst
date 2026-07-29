.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Add models with the Megatron-LM backend
=========================================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Model
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----------


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
If use latest verl, we have direct support of ``GPTModel`` for Megatron backend. 
You can use the similar way of using Megatron to pretrain custom models. 
We list the steps here:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Find `model_initializer.py <https://github.com/volcengine/verl/blob/main/verl/models/mcore/model_initializer.py>`_
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. If your model is configurable by ``TransformerLayerSpec`` , you can
   directly use ``GPTModel``. Otherwise, Please implement a new
   ``ModelLayerSpec`` and ``ModelLayer`` here.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. Use the right ``LayerSpec`` , ``TransformerConfig`` and ``HuggingfaceConfig`` 
   as arguments to initialize the GPTModel.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
4. Return the model at last.


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Add Models with old version of verl
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----------------------------------


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The most challenging aspect to use the Megatron-LM backend is implementing
the models for training. Currently, we implement Llama model that
support data parallelism, tensor parallelism, pipeline parallelism (also
vPP) and sequence parallelism. We also implement remove padding (sequence packing) on Llama
model, which can be found in `modeling_llama_megatron.py <https://github.com/volcengine/verl/blob/main/verl/models/llama/megatron/modeling_llama_megatron.py>`_.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
To support other model, users are required to implement:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Implemnt a model similar to ``modeling_llama_megatron.py`` that satisfy the
   parallelism requirements of Megatron-LM. Then register your model in
   the `registry.py <https://github.com/volcengine/verl/blob/main/verl/models/registry.py>`_.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Checkpoint utils that can load full checkpoint (e.g. huggingface
   checkpoint) to partitioned models during the runtime. Then register
   your loader to ``weight_loader_registry`` in `weight_loader_registry.py <https://github.com/volcengine/verl/blob/main/verl/models/weight_loader_registry.py>`_.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. Weight loader that synchronize the weight from Megatron to rollout
   (vLLM) model. Note that both the actor model and rollout model are
   partitioned during runtime. So, it's advisable to map the model name
   in actor model implementation. Otherwise, you may need an additional
   name mapping and even weight transformation. The weight loader implementation
   is in `megatron_weight_loaders.py <https://github.com/volcengine/verl/blob/main/verl/third_party/vllm/vllm_v_0_6_3/megatron_weight_loaders.py>`_.