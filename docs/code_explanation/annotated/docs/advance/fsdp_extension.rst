
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Add models with the FSDP backend
==================================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Model
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
--------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
In principle, our FSDP backend can support any HF model and we can
sychronoize the actor model weight with vLLM using `hf_weight_loader.py` under `third_party/vllm`.
However, ``hf_weight_loader`` is will gather the full state_dict of a
model during synchronization, which may cause OOM. We suggest using
``dtensor_weight_loader`` which gather the full model parameter layer by
layer to reduce the peak memory usage. We already support dtensor weight
loader for the models below in `dtensor_weight_loader.py` under `third_party/vllm`:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``GPT2LMHeadModel``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``LlamaForCausalLM``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``LLaMAForCausalLM``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``MistralForCausalLM``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``InternLMForCausalLM``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``AquilaModel``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``AquilaForCausalLM``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``Phi3ForCausalLM``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``GemmaForCausalLM``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``Gemma2ForCausalLM``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``GPTBigCodeForCausalLM``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``Starcoder2ForCausalLM``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``Qwen2ForCausalLM``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ``DeepseekV2ForCausalLM``

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
To implement ``dtensor_weight_loader`` of a model that's supported in
vLLM, follow the guide of gemma model below:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Copy the
   ``load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]])`` from the vllm model class
   to ``dtensor_weight_loaders.py``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Modify the arguments to
   ``(actor_weights: Dict, vllm_model: nn.Module)``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. Replace the ``self`` to ``vllm_model``
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
4. Add the
   ``local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)``
   before each ``param = params_dict[name]`` and modify the following
   weight loading using ``local_loaded_weight``.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
5. Register the implemented dtensor weight loader to ``__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__``.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: diff

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    + def gemma_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
        stacked_params_mapping = [
            .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    -   params_dict = dict(self.named_parameters())
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    +   params_dict = dict(vllm_model.named_parameters())
        loaded_params = set()
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    -   for name, loaded_weight in weights:
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    +   for name, loaded_weight in actor_weights.items():
            for (param_name, shard_name, shard_id) in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    +           local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
                param = params_dict[name]
                weight_loader = param.weight_loader
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    -           weight_loader(param, loaded_weight, shard_id)
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    +           weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
                break
            else:
                .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
                # lm_head is not used in vllm as it is tied with embed_token.
                .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
                # To prevent errors, skip loading lm_head.weight.
                if "lm_head.weight" in name:
                    continue
                .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    +           local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    -           weight_loader(param, loaded_weight)
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    +           weight_loader(param, local_loaded_weight.to(dtype=param.dtype))
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            raise RuntimeError(
                "Some weights are not initialized from checkpoints: "
                f"{unloaded_params}")