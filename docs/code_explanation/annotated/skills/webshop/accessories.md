<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### TASK: accessories ###

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
For this accessories shopping task, apply these specific strategies:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Constraint-Packed Search: Compose the initial search query by combining the accessory type with every known constraint (color, size, price cap, material, washability) to surface only viable candidates. Apply this before the first search when the user provides multiple specific requirements.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Explicit Variant Selection: Always click and visibly confirm the exact color, size, or pattern variant on the product page instead of assuming the default matches the request. Apply this immediately after opening a product that offers multiple variant options.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. Function Fit Check: Read the title and description to verify the accessory's intended use (e.g., eye-shadow case vs. generic makeup bag) aligns with the user's need before proceeding. Apply this right after landing on a product page that seems category-relevant but may have ambiguous use.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. Post-Config Price Audit: After selecting all variants, re-confirm the displayed price is within the user's budget. If it exceeds, backtrack and search for alternatives. Apply this just before initiating checkout or adding to cart.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5. Fallback to Next Result: If any required variant is unavailable or constraints cannot be met on the current page, return to results and open the next candidate instead of forcing a mismatch. Apply this when the product lacks the needed option or violates any key constraint after inspection.
