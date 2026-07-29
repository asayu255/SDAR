<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Memory Manager

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
<p align="center">
    <img src="../../docs/gigpo/framework-comparison.png" alt="framework" width="100%">
</p>

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
`verl-agent` allows for flexibly choosing what history to include for each step, such as, recent steps, key events, summaries, or external knowledge.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
We provide a simplest memory implementation as a starting point. Developers are encouraged to extend this module with custom memory strategies, such as dynamic summarization, selective memory retention, or external knowledge integration, to improve the handling of long-horizon interaction histories.