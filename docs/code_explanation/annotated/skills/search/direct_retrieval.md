<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### TASK: direct_retrieval ###

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
For direct fact-retrieval tasks, apply these specific strategies:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Isolate Core Query: Strip the question to its key entity plus sought fact (who/what/when/where) and search exactly that pair first. Apply this at the start of any direct-retrieval task.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Refine When Empty: If the first search yields weak or no hits, instantly reformulate using synonyms, alternate names, dates, or quoted phrases instead of repeating the same query. Apply this after an initial search returns no clear answer or only tangential results.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. Anchor With Quotes: For song titles, quotes, episode names, etc., wrap the unique phrase in quotation marks to pull exact-match sources. Apply this when the query contains distinctive phrases, lyrics, book/film titles, or direct quotations.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. Check Temporal Context: Include recency cues (e.g., "current", year) in the search and verify publication date to avoid outdated or speculative info. Apply this for questions about "current", "latest", or future events/releases.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5. Evidence-Bound Answer: Only state an answer that is explicitly supported by the retrieved text; if unclear, continue searching rather than guess or hallucinate. Apply this before finalizing any factoid answer.
