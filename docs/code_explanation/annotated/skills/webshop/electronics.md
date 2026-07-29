<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### TASK: electronics ###

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
For this electronics shopping task, apply these specific strategies:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Constraint-Rich Search: Pack product type plus every mandatory attribute (features, color, size, price cap) into the initial search string to surface only highly relevant electronics items. Apply this whenever starting a new product hunt or refining after poor results.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Configure In-Page Variants: Once on a promising product page, immediately set color, size or pack, and other variant selectors to test if the SKU can meet all requested specs. Apply this right after opening a candidate item from search results.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. Hard-Gate Price and Specs: Before proceeding to buy, cross-check the displayed title, specs, and price. Abandon any item that exceeds the budget or lacks the exact requested feature set. Apply this after configuring variants but before clicking Buy Now.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. Bail on Mismatch Fast: If variant clicks do not update the product details to the required spec, or if the page remains wrong, use Back to results and seek another item instead of retrying the same option. Apply this when repeated option clicks leave title or specs unchanged or incompatible.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5. Prioritize High-Match Results: Scan the result list for items whose titles show all key constraints. Open those first, rather than settling for the earliest but partial match. Apply this on any search results page with multiple potential electronics products.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
6. Avoid Click Loops: Track visited products and queries. If progress stalls, refine the search or change items instead of cycling between the same pages. Apply this after revisiting a page or running an identical search without new insights.
