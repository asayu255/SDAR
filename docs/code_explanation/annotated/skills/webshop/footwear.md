<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### TASK: footwear ###

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
For this footwear shopping task, apply these specific strategies:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Load Search Constraints: Include all hard filters (type, function, color, size, price cap) directly in the initial search query to surface only viable footwear candidates. Apply this before the first search or whenever starting a new search cycle.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Verify Key Features: Open the product description or specs to explicitly confirm required functional attributes (e.g., slip-resistant, rubber sole) before moving toward purchase. Apply this immediately after opening a product page and before selecting variants or buying.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. Select Variants First: Always choose the requested color and size options on the product page prior to any purchase action to ensure the exact variant meets constraints. Apply this once a candidate product page is open and verified for required features.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. Check Variant Price: Confirm the displayed price for the chosen size and color is strictly below the user's maximum before clicking Buy Now, especially when ranges are shown. Apply this after selecting variants and just before initiating checkout.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5. Exit Non-Matches Fast: If a product or result page clearly violates any hard constraint, back out immediately instead of toggling options or repeating identical searches. Apply this upon noticing missing attributes, wrong category, unavailable size or color, or over-budget price.
