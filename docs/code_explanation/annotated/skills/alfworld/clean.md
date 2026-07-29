<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### TASK: clean ###

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
For this clean-and-place task, apply these specific strategies:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Phase-Ordered Plan: Always execute clean tasks in the fixed sequence: (1) locate and acquire target object, (2) bring it to an available water source or sink to clean, (3) navigate to final location, (4) place object. Apply this as soon as the goal specifies the object must be clean before placement.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Pick Before You Wander: If the target object is visible or discovered, take it immediately. Never leave it behind to explore other places, as possession enables direct cleaning and prevents redundant searches. Apply this after visually confirming the presence of the target object on any surface or inside an opened container.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. Sink First for Cleaning: Upon holding the target object, go straight to the nearest sink, basin, or faucet and issue the clean or use command before any placement attempts. Apply this once the target object is in hand and its required state is clean.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. Systematic Container Sweep: When the object is not found on open surfaces, iterate through all unopened or unexamined containers in the room before revisiting already-checked spots to avoid search loops. Apply this after initial obvious surfaces are empty and the target object remains unfound.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5. State Verification Before Drop: Always inspect or infer the object state after cleaning. If still not confirmed clean, re-clean before final placement to satisfy goal conditions. Apply this immediately after a cleaning action and before placing the object at its target location.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
6. Use Location Priors: Begin search at the most probable category-specific surfaces (e.g., kitchenware on countertop or stove, food on dining table) to minimize exploration steps and avoid aimless roaming. Apply this at the very start of the task to choose the first search destination for the target object.
