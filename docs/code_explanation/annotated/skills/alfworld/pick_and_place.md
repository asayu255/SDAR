<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### TASK: pick_and_place ###

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
For this pick-and-place task, apply these specific strategies:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Systematic First-Pass Search: Maintain a checklist of all visible and closed containers and surfaces. Open or inspect each unseen candidate exactly once before revisiting any location. Apply this after reading the goal and before acquiring every required object.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Grab When Seen: Whenever a needed object is visible and reachable, immediately take it before moving elsewhere. Do not leave targets uncollected. Apply this upon first sight of an unheld object that matches the goal specification.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. Transform Before Transport: If the goal specifies a state change (e.g., heated, cooled, cleaned), perform the transformation at the nearest appropriate appliance before heading to the final destination. Apply this right after acquiring an object that must change state.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. Place Before More Search: When holding any goal object and the target location is known and reachable, navigate there and place it immediately, then resume searching if more items are needed. Apply this while carrying a required object and the destination has been identified.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5. Track Counts for Multiples: Maintain a tally of how many target objects are required, held, and already placed. Stop searching only after the placed count meets the goal. Apply this throughout tasks demanding two or more instances of the same object.
