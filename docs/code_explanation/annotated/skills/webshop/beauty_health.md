<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### TASK: beauty_health ###

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
For this beauty and health shopping task, apply these specific strategies:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Bundle Constraints Search: Combine product type with every user-stated filter (price cap, color, size, material, feature keywords) in one search query to surface highly relevant results. Apply this when the user lists two or more specific requirements (e.g., price plus color, size, feature).

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Feature-Led Click: Open the first result whose title explicitly mentions the key functional attribute (e.g., double sided, long handle, dry hair) to maximize match likelihood. Apply this after search results appear and at least one headline contains the core feature term.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. Variant Verification and Set: Within the product page, actively choose color, size, or pack options to ensure the selected variant meets all user constraints before purchase. Apply this when the product offers selectable options and the user has specified any of those attributes.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. Price Gatekeeping: Cross-check the displayed price (including chosen variant) against the user's maximum. If exceeded, abandon and return to results immediately. Apply this right after selecting the desired variant and before initiating checkout.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5. Minimal Path Purchase: Once a product fully satisfies all constraints, proceed directly to Buy Now without extra browsing to reduce error risk and time. Apply this when all user requirements are confirmed on the current product page.
