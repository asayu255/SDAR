<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# SPPO: Self-Play Preference Optimization for Language Model Alignment

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
This repository hosts the community implementation for the paper [Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/abs/2405.00675). SPPO can significantly enhance the performance of an LLM without strong external signals such as responses or preferences from GPT-4. It can outperform the model trained with iterative direct preference optimization (DPO), among other methods. SPPO is theoretically grounded, ensuring that the LLM can converge to the von Neumann winner (i.e., Nash equilibrium) under general, potentially intransitive preference, and empirically validated through extensive evaluations on multiple datasets.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Paper Authors: [Yue Wu](https://yuewu.us/)\*, [Zhiqing Sun](https://www.cs.cmu.edu/~zhiqings/)\*, [Huizhuo Yuan](https://scholar.google.com/citations?user=8foZzX4AAAAJ)\*, [Kaixuan Ji](https://scholar.google.com/citations?user=FOoKDukAAAAJ), [Yiming Yang](https://www.cs.cmu.edu/~yiming/), [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
verl Implementation Authors: [Yuhao Yang](https://github.com/yhyang201), [Chenyang Zhao](https://github.com/zhaochenyang20)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
[[Webpage](https://uclaml.github.io/SPPO/)] [[Huggingface](https://huggingface.co/papers/2405.00675)] [[Paper](https://arxiv.org/abs/2405.00675)][[Original Implementation](https://github.com/uclaml/SPPO)]

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Reproduce the Experiment

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
We evaluate the performance of SPPO on the MATH dataset. Starting from an initial score of 46.6 with Qwen2.5-7B-Instruct, we achieve a score of 65.6 after 20 epochs of training, placing our model approximately in the top 20 on the [MATH leaderboard](https://paperswithcode.com/sota/math-word-problem-solving-on-math). It's important to note that verl's internal evaluation metrics may not perfectly align with the official evaluation methodology for Qwen2.5-7B-Instruct. Therefore, for consistency and fair comparison, we report only the results based on verl's evaluation framework.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```
git clone git@github.com:volcengine/verl.git
cd verl
python3 -m uv pip install -e ".[sglang]"

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct

export CUDA_VISIBLE_DEVICES=0,1,2,3
bash recipe/sppo/run_qwen2.5-7b_rm.sh
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Note that the installation would occasionally fail to install flash-attn. If this happens, you can install it manually by running:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
```

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Acknowledgement

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
We sincerely thank the contribution and guidance from:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [Yue Wu](https://yuewu.us/)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [Chendong Wang](https://cdwang96.github.io/)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [Yifan Zhang](https://github.com/yifanzhang-pro)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [Yongan Xiang](https://github.com/BearBiscuit05)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [Junrong Lin](https://github.com/ocss884)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [Yuxuan Tong](https://github.com/tongyx361)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [Guangming Shen](https://github.com/PeterSH6)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [Biao He](https://www.linkedin.com/in/biao-he/)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [Qingquan Song](https://qingquansong.github.io/)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)
