<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# verl Megatron-Core Models
The earlier versions of verl use `Megatron-LM` 0.4 and workaround huggingface model classes. To better use the latest features and speedup of modern Megatron, we are migrating to `Megatron-Core`(mcore), and use the recommended `GPTModel` class for all language models. With mcore `GPTModel`, we can use the latest features like `context parallel`, `expert parallel`, `dist_checkpointing`, etc. and we can update mcore with little effort in the future for new features.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
The migration has been successful with the help of the mcore team and the community. What we have done is:
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. update `Megatron` version to `0.11.0`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. migrate `LlamaForCausalLM` and `Qwen2ForCausalLM` to mcore `GPTModel`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. support sequence packing/thd format.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. support `tensor parallel`, `pipeline parallel`, `sequence parallel`, `virtual pipeline parallel`, `context parallel`.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5. support the mcore `dist_checkpointing` feature and a basic offline weighs conversion scipt from huggingface to mcore `dist_checkpointing` format.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
We are working on the following features:
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- support `Qwen2MoeForCausalLM`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- support `MixtralForCausalLM`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- support `DeepseekV3ForCausalLM`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- support `expert parallel`

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Features we invite the community to contribute:
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- better scipts for offline weights conversion from huggingface to mcore `dist_checkpointing` format.
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - conversion of large models with multiple GPUs
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - conversion of large models with single GPU
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- refactor the `megatron_checkpoint_manager.py` by `dist_checkpointing` format.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- support llama4
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- support qwen2.5-vl

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
To track the progress of verl mcore integration, please refer to the [mcore integration issue](https://github.com/volcengine/verl/issues/1033).

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## How things work now
To engage the community in contributing, here are the key steps in our mcore integration process and features under development. 

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
The huggingface `transformers` is the de facto standard of model zoo while mcore is good at computation efficiency. The main challenge is conversion between the two.
main steps:
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. modelling the huggingface model with mcore `GPTModel`
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - a. convert the huggingface config to mcore `TransformerConfig`
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - b. init the mcore `GPTModel` with the converted config
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - c. load the huggingface model weights to the `GPTModel`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. online weight conversion from mcore to huggingface (due the the rollout engine `vLLM` is using huggingface format)
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - a. bridge the gap between mcore and huggingface weights format and name mapping
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - b. online resharding the mcore weights to rollout engine
        <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
        - this part is very complicated with multiple parallel strategies composition between mcore and rollout engine
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. support the mcore features in verl
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - a. support `tensor parallel`, `pipeline parallel`, `sequence parallel`, `virtual pipeline parallel`, `context parallel`
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - b. support recompute and other mcore speed up features

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. checkpointing
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - a. support recovering the verl training.
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - b. support exporting the mcore checkpoint to huggingface format, for downstream inference.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Modelling the huggingface model with mcore `GPTModel`
The first step is to convert huggingface config to mcore `TransformerConfig` and init the mcore `GPTModel` with the converted config. See code in `verl/models/mcore/config_converter.py` and `verl/verl/models/mcore/models/model_initializer.py`. The corresponding model forward code is in `verl/verl/models/mcore/models/model_forward.py`.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
There are two ways of loading the huggingface model weights to the `GPTModel`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Runtime loading
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - every rank loads the entire huggingface model weights and then shard and convert to mcore weights.
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - speed is slow and memory consumption is high.
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - this way is deprecated and will not support new models.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Offline loading
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - use offline script to convert the huggingface model weights to mcore weights and save with mcore `dist_checkpointing` format.
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - online loading and sharding is automatically done by mcore `dist_checkpointing` format. The speed is fast and memory consumption is low.
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - the offline script is in `verl/scripts/converter_hf_to_mcore.py`.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### online weight conversion from mcore to huggingface
See function `convert_megatron_model_to_transformers_model` in `verl/utils/megatron_utils.py` for the details.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
It should be refatored for extensibility and better performance.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### support the mcore features in verl
Most of the features of `GPTModel` is out-of-the-box supported in verl through changing the `TransformerConfig`, except those about parallel strategies, such as `expert parallel`. 
Features about parallel strategies should be supported with changes about the online weights conversion(especially the resharding part) and verl work dispatching.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### checkpointing
The existing checkpointing code is in `verl/utils/checkpoint/megatron_checkpoint_manager.py`. And the script to convert checkpoint to huggingface format is in `verl/scripts/model_merger.py`.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
The existing checkpoint format is simplely save every rank's weights and optimizer states. It should be refactored by `dist_checkpointing` format.


<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## How to support new models
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. make sure the model is supported by vLLM
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. modelling the huggingface model with mcore `GPTModel` (The [Pai-Megatron-Path](https://github.com/alibaba/Pai-Megatron-Patch/tree/main) is a good reference)
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - a. convert the huggingface config to mcore `TransformerConfig`
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - b. init the mcore `GPTModel` with the converted config
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - c. load the huggingface model weights to the `GPTModel`
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - d. for VLM the interface might be different, it is ok to add a new model class with GPTModel as its module.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. offline weights conversion from huggingface to mcore `dist_checkpointing` format
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. support online weights conversion from mcore to huggingface
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - it is recommended to initilize a vLLM model with the converted mcore weights, and then test if the generating sequence is correct.


<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## How to scale up to larger models like deepseek-v3 or other 100B+ models
The greatest challenge for scaling up to larger models is the memory consumption.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
The necessary features under development for scaling up are
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Training engine part
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - expert parallel
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Rollout engine part
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - pipeline parallel
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - expert parallel
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - more efficient and general weight resharding and loading
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. Offline weights conversion
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - support weights larger then single GPU memory
