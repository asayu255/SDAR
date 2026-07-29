<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Checklist Before Starting

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [ ] Search for similar PR(s).

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### What does this PR do?

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> Add one-line overview of what this PR aims to achieve or accomplish. 

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### High-Level Design

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> Demonstrate the high-level design if this PR is complex.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Specific Changes

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> List the specific changes.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### API

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> Demonstrate how the API changes if any.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Usage Example

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> Provide usage example(s) for easier usage.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```python
# Add code snippet or script demonstrating how to use this 
```

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Test

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> For changes that can not be tested by CI (e.g., algorithm implementation, new model support), validate by experiment(s) and show results like training curve plots, evaluatuion results, etc.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Additional Info.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **Issue Number**: Fixes issue # or discussion # if any.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **Training**: [Note which backend this PR will affect: FSDP, Megatron, both, or none]
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **Inference**: [Note which backend this PR will affect: vLLM, SGLang, both, or none]

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Checklist Before Submitting

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [ ] Read the [Contribute Guide](https://github.com/volcengine/verl?tab=readme-ov-file#contribution-guide).
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [ ] Apply [pre-commit checks](https://github.com/volcengine/verl?tab=readme-ov-file#code-linting-and-formatting).
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [ ] Add `[BREAKING]` to the PR title if it breaks any API.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [ ] Update the documentation about your changes in the [docs](https://github.com/volcengine/verl/tree/main/docs).
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [ ] Add CI test(s) if necessary.
