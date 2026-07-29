<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# Upgrading to vllm >= 0.7

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Note: verl+vllm 0.8.3 is now stable. Please see ``docs/README_vllm0.8.md`` for upgrade guide.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Installation

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Note: At time of writing, verl+vllm 0.7.x supports **FSDP** for training and **vLLM** for rollout.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```
# Create the conda environment
conda create -n verl python==3.10
conda activate verl

# Install verl
git clone https://github.com/volcengine/verl.git
cd verl
pip3 install -e .

# Install the latest stable version of vLLM
pip3 install vllm==0.7.3 

# Install flash-attn
pip3 install flash-attn --no-build-isolation

```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Note that if you are installing lower versions of vLLM (0.7.0, 0.7.1, 0.7.2), you need to make some tiny patches manually on vllm (/path/to/site-packages/vllm after installation) after the above steps:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- vllm/distributed/parallel_state.py: Remove the assertion below:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```
if (world_size
        != tensor_model_parallel_size * pipeline_model_parallel_size):
    raise RuntimeError(
        f"world_size ({world_size}) is not equal to "
        f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
        f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")

```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- vllm/executor/uniproc_executor.py: change `local_rank = rank` to `local_rank = int(os.environ["LOCAL_RANK"])`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- vllm/model_executor/model_loader/weight_utils.py: remove the `torch.cuda.empty_cache()` in `pt_weights_iterator`

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Features

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Use cuda graph

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
After installation, examples using FSDP as training backends can be used. By default, the `enforce_eager` is set to True, which disables the cuda graph. To enjoy cuda graphs and the sleep mode of vLLM>=0.7, add the following lines to the bash script:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.free_cache_engine=False \

```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
For a typical job like examples/ppo_trainer/run_qwen2-7b_seq_balance.sh, the rollout generation time is 115 seconds with vLLM0.6.3, while it is 85 seconds with vLLM0.7.0. By enabling the cudagraph, the generation duration is further reduced to 62 seconds.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Note:** Currently, if the `n` is greater than 1 in `SamplingParams` in vLLM>=0.7, there is a potential performance issue on the stability of rollout generation time (Some iterations would see generation time bursts) using vLLM's V0 Engine.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Use vLLM V1 Engine

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Using the vLLM V1 engine can avoid instability issues and achieve additional performance improvements. To use the V1 engine, you can first uninstall the previously installed vLLM and then follow the steps below to install the newer version.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 2275784
sed -i "903a\    data_parallel_size = world_size // pipeline_model_parallel_size // tensor_model_parallel_size" ./vllm/distributed/parallel_state.py
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Then you can enable the V1 engine by setting `export VLLM_USE_V1=1`. In some benchmark tests, the V1 engine demonstrates a 1.5x speed improvement over the vLLM V0 engine.
The stable support of the vLLM V1 engine is available on verl main.
