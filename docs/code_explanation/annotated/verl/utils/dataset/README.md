<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# Dataset Format
<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## RLHF dataset
We combine all the data sources into a single parquet files. We directly organize the prompt into the chat format so that multi-turn chats can be easily incorporated. In the prompt, we may add instruction following texts to guide the model output the answers in a particular format so that we can extract the answers.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Math problems
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```json
{
    "data_source": "openai/gsm8k",
    "prompt": [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after \"####\""}],
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": ["72"]
    },
}
```
