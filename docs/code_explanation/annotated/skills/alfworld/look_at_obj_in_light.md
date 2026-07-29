<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### TASK: look_at_obj_in_light ###

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
For this examine-object-under-light task, apply these specific strategies:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Seek Lamp Surfaces First: Head straight to furniture that commonly hosts a desklamp (desk, sidetable, nightstand) because the target must end up under that light. Apply this right after parsing the goal.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Switch Lamp On: Issue the use desklamp command as soon as you reach it so the light condition is satisfied before or immediately after handling the target object. Apply this upon arriving at a desklamp that is currently off.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. Acquire Or Fetch Target: If the target is on the lit surface, take it. If found elsewhere, pick it up and carry it back to the lit lamp area. Apply this whenever the target object becomes visible, whether under the lamp or not.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. Pair Objects Early: When both the target and desklamp are seen on the same surface, immediately pick up the target there and use the desklamp without moving elsewhere. Apply this upon first observation that the target object and desklamp share the current location.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5. Grab Target First: If you see the target but not the desklamp yet, take the target immediately so you can carry it to wherever the desklamp is found. Apply this when the target object is visible and not yet held, while the desklamp location is unknown.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
6. Shortcut To Tool: Once holding the target, navigate straight to the nearest known desklamp and issue a single use command with no intervening searches or detours. Apply this when the target object is in inventory and at least one desklamp location is known.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
7. Single Toggle Rule: Use the desklamp only after the target is in hand or co-located. Avoid repeated or premature toggles that waste steps. Apply this when about to interact with a desklamp.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
8. Systematic Surface Search: On each candidate lamp surface, look, open nearby containers and drawers once, then move on, ensuring all plausible lamp locations are covered. Apply this after the lamp at the current surface is on and the target has not been found.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
9. Explore New Surfaces: If either the target or the desklamp has not been found, systematically check unvisited surfaces and containers instead of re-inspecting places already confirmed empty. Apply this when the location of a required object remains unknown after current area is exhausted.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
10. Lock In Observed Items: After spotting either required object, do not leave without acting. Pick it up if it is the target, or note its location if it is the desklamp, to prevent costly backtracking loops. Apply this just after observing a required object during scanning.
