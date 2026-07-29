<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# Models
Common modelzoo such as huggingface/transformers stuggles when using Pytorch native model parallelism. Following the design principle of vLLM, we keep a simple, parallelizable, highly-optimized with packed inputs in verl. 
<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Adding a New Huggingface Model
<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Step 1: Copy the model file from HF to verl
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Add a new file under verl/models/hf
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Copy ONLY the model file from huggingface/transformers/models to verl/models/hf

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Step 2: Modify the model file to use packed inputs
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Remove all the code related to inference (kv cache)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Modify the inputs to include only
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - input_ids (total_nnz,)
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - cu_seqlens (total_nnz + 1,)
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - max_seqlen_in_batch: int
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Note that this requires using flash attention with causal mask.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Step 2.5: Add tests
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Add a test to compare this version and the huggingface version
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Following the infrastructure and add tests to tests/models/hf

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Step 3: Add a function to apply tensor parallelism
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Please follow
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - https://pytorch.org/docs/stable/distributed.tensor.parallel.html
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - https://pytorch.org/tutorials/intermediate/TP_tutorial.html
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- General comments
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - Tensor Parallelism in native Pytorch is NOT auto-parallelism. The way it works is to specify how model parameters and input/output reshards using configs. These configs are then registered as hooks to perform input/output resharding before/after model forward.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Step 4: Add a function to apply data parallelism
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Please use FSDP2 APIs
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- See demo here https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L413

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Step 5: Add a function to apply pipeline parallelism
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Comes in Pytorch 2.4
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Currently only in alpha in nightly version
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Check torchtitan for more details

