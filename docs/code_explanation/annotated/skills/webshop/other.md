<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### TASK: other ###

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
For this general product shopping task, apply these specific strategies:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Embed All Constraints: Build the initial search query to include every required attribute (product type, key features, color or size, and price cap) to surface only high-relevance results. Apply this right before issuing any product search.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Confirm Exact Category: Open a result only if its title clearly matches the requested product type, and abandon pages that reveal a different category despite matching other attributes. Apply this immediately upon landing on a product detail page.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. Select and Verify Variants: After choosing required color, size, or pack options, ensure the page's title and price update to the selected variant before proceeding to purchase. Apply this when a product offers dropdown or swatch selections.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. Check Fixed Specs: Manually read the description or features to confirm non-selectable requirements (materials, special functions, care instructions) are explicitly met. Apply this before clicking Buy Now on any configured product.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5. Refine If No Match: If initial results do not meet all constraints, adjust keywords or add filters (e.g., synonyms, price slider) instead of re-clicking unsuitable items. Apply this after reviewing a results page that lacks clear matches.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
6. Follow Purchase Flow: Maintain the disciplined sequence: Search, Open candidate, Apply constraints, Verify all criteria and price, Buy Now. Avoid unnecessary back-and-forth. Apply this for every shopping task from start to checkout.
