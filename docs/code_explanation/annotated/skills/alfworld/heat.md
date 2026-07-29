<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### TASK: heat ###

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
For this heat-and-place task, apply these specific strategies:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Secure Exact Target First: Always identify and pick up the exact object type named in the goal before interacting with the microwave or destination. Ignore look-alikes (e.g., do not substitute a mug for a cup). Apply this after spotting any candidate object or during initial search phase, before opening or using appliances.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Systematic One-Pass Search: Search each plausible surface or closed container once by opening it and inspecting contents. Mark searched spots mentally to avoid redundant revisits. Apply this while locating the target object and you have not yet found it.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. Open Then Heat: Upon reaching the microwave with the target in hand, always open the door, place the object inside, and execute the heat action before leaving. Apply this immediately after navigating to the microwave with the target object held.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. No Appliance Before Object: Do not move to or interact with the microwave or final placement location until the target object is picked up, preventing wasted navigation steps. Apply this whenever you are tempted to head to microwave or destination without holding the required object.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5. Direct Post-Heat Placement: After heating, navigate straight to the specified destination and place the object once, avoiding extra exploration or detours. Apply this right after the heating action completes and the object is in hand.
