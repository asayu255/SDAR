<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### TASK: multi_hop_reasoning ###

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
For multi-hop reasoning tasks, apply these specific strategies:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Decompose Question First: Split the main query into explicit sub-questions to identify each entity and the specific attribute or relationship needed before searching. Apply this to any multi-hop question that links two or more entities/facts (e.g., director's birthplace, album vs film comparisons).

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Targeted Sequential Searches: Issue separate, focused searches for each sub-question or entity instead of one broad query to gather precise intermediate facts. Apply this after decomposition, when distinct pieces of information must be collected individually.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. Collect-Then-Compare: Retrieve concrete values for all items involved (dates, places, relations) before performing any comparison or conclusion to avoid premature or unsupported answers. Apply this to comparative tasks (earlier/later, older/younger, bigger/smaller, etc.).

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. Contextual Disambiguation: Add clarifying descriptors (profession, year, medium, location) to queries to distinguish between entities with identical or similar names and prevent mixing facts. Apply this whenever an entity name is generic or shared by multiple subjects.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5. Iterative Query Refinement: If initial search fails or returns ambiguous results, promptly rephrase using synonyms, alternate titles, or additional context instead of repeating the same query. Apply this after a search yields no relevant results or conflicting information.
