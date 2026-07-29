# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import random
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import numpy as np
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import marshal
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import copy
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import deque

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import matplotlib.pyplot as plt
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import matplotlib.animation as animation

# [EXPLAIN] `get_shortest_action_path` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_shortest_action_path(room_fixed, room_state, MAX_DEPTH=100):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Get the shortest action path to push all boxes to the target spots.
        Use BFS to find the shortest path.
        NOTE currently only support one player, only one shortest solution
        =========================================================
        Parameters:
            room_state (np.ndarray): the state of the room
                - 0: wall
                - 1: empty space
                - 2: box target
                - 3: box on target
                - 4: box not on target
                - 5: player
            room_fixed (np.ndarray): the fixed part of the room
                - 0: wall
                - 1: empty space
                - 2: box target
            MAX_DEPTH (int): the maximum depth of the search
        =========================================================
        Returns:
            action_sequence (list): the action sequence to push all boxes to the target spots
        """
        
        # BFS queue stores (room_state, path)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        queue = deque([(copy.deepcopy(room_state), [])])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        explored_states = set()
        
        # Possible moves: up, down, left, right
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        actions = [1, 2, 3, 4] # Corresponding action numbers
        
        # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
        while queue:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            room_state, path = queue.popleft()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if len(path) > MAX_DEPTH:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return [] # No solution found

            # reduce the search space by checking if the state has been explored
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_tohash = marshal.dumps(room_state)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if state_tohash in explored_states:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            explored_states.add(state_tohash)
            

            # get information of the room
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            player_pos = tuple(np.argwhere(room_state == 5)[0])
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            boxes_on_target = set(map(tuple, np.argwhere((room_state == 3))))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            boxes_not_on_target = set(map(tuple, np.argwhere((room_state == 4))))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            boxes = boxes_on_target | boxes_not_on_target


            # Check if all boxes are on targets
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if not boxes_not_on_target:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return path
                
            # Try each direction
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for move, action in zip(moves, actions):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                new_room_state = copy.deepcopy(room_state)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                new_player_pos = (player_pos[0] + move[0], player_pos[1] + move[1])
                
                # Check is new player position is wall or out of bound
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if new_player_pos[0] < 0 or new_player_pos[0] >= room_fixed.shape[0] \
                    or new_player_pos[1] < 0 or new_player_pos[1] >= room_fixed.shape[1] \
                    or room_fixed[new_player_pos] == 0:
                    # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                    continue
                    
                # If there's a box, check if we can push it
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if new_player_pos in boxes:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    box_pos = new_player_pos # the original box position
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    new_box_pos = (new_player_pos[0] + move[0], new_player_pos[1] + move[1])
                    
                    # Can't push if hitting wall or another box or out of bound
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if room_fixed[new_box_pos] == 0 or new_box_pos in boxes \
                        or new_box_pos[0] < 0 or new_box_pos[0] >= room_fixed.shape[0] \
                        or new_box_pos[1] < 0 or new_box_pos[1] >= room_fixed.shape[1]:
                        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                        continue
                        
                    # move the box
                    
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    new_room_state[box_pos] = room_fixed[box_pos]
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if room_fixed[new_box_pos] == 2:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        new_room_state[new_box_pos] = 3
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        new_room_state[new_box_pos] = 4
                
                # player moves
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                new_room_state[player_pos] = room_fixed[player_pos]
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                new_room_state[new_player_pos] = 5
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                queue.append((new_room_state, path + [action]))
                        
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [] # No solution found

# def plot_animation(imgs):
#     fig, ax = plt.subplots()
#     im = ax.imshow(imgs[0])
#     def init():
#         im.set_data(imgs[0])
#         return [im]
#     def update(i):
#         im.set_data(imgs[i])
#         return [im]
#     ani = animation.FuncAnimation(fig, update, frames=len(imgs), init_func=init, blit=True)
#     return ani

# [EXPLAIN] `plot_animation` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def plot_animation(imgs):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    height, width = imgs[0].shape[:2]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    fig = plt.figure(figsize=(width/100, height/100), dpi=500)
    
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ax = fig.add_axes([0, 0, 1, 1])
    
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ax.set_xticks([])
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ax.set_yticks([])
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ax.set_frame_on(False)
    
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    im = ax.imshow(imgs[0])
    # [EXPLAIN] `init` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        im.set_data(imgs[0])
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [im]
    # [EXPLAIN] `update` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def update(i):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        im.set_data(imgs[i])
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return [im]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ani = animation.FuncAnimation(fig, update, frames=len(imgs), init_func=init, blit=True)
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return ani

# [EXPLAIN] `solve_sokoban` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def solve_sokoban(env, saved_animation_path):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Solve the given sokoban environment and save the animation
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    actions = get_shortest_action_path(env.room_fixed, env.room_state)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    print(f"Found {len(actions)} actions: {actions}")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    imgs = []
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    img_before_action = env.render('rgb_array')
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    imgs.append(img_before_action)
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for action in actions:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        env.step(action)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        img_after_action = env.render('rgb_array')
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        imgs.append(img_after_action)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ani = plot_animation(imgs)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    ani.save(saved_animation_path)



        





# [EXPLAIN] `add_random_player_movement` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def add_random_player_movement(room_state, room_structure, move_probability=0.5, continue_probability=0.5, max_steps=3):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Randomly move the player after reverse_playing to make the level more challenging, also fix the problem that in generated map, the player is always adjacent to the box
    
    Parameters:
        room_state (np.ndarray): Current state of the room
        room_structure (np.ndarray): Fixed structure of the room
        move_probability (float): Probability of moving the player at all (0.0-1.0)
        continue_probability (float): Probability of continuing to move after each step (0.0-1.0)
        max_steps (int): Maximum number of steps the player can move (1-3)
    
    Returns:
        np.ndarray: Updated room state with randomly moved player
    """
    # Check if we should move the player at all
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if random.random() > move_probability:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return room_state
    
    # Find player position
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    player_pos = np.where(room_state == 5)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    player_pos = np.array([player_pos[0][0], player_pos[1][0]])
    
    # Keep track of previous positions to avoid moving back
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    previous_positions = [tuple(player_pos)]
    
    # Make 1-3 random moves
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    steps_taken = 0
    # [EXPLAIN] 終了条件を満たすまで rollout または状態更新を反復する。
    while steps_taken < max_steps:
        # Get all valid moves (can't move into walls or boxes)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        valid_moves = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for action in range(4):  # 0: up, 1: down, 2: left, 3: right
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            change = CHANGE_COORDINATES[action]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            next_pos = player_pos + change
            
            # Check if next position is valid (empty space or target) and not a previous position
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if (room_state[next_pos[0], next_pos[1]] in [1, 2] and 
                tuple(next_pos) not in previous_positions):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                valid_moves.append((action, next_pos))
        
        # If no valid moves, break
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not valid_moves:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break
        
        # Choose a random valid move
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        chosen_action, next_pos = random.choice(valid_moves)
        # print(f"player_pos: {player_pos}, next_pos: {next_pos}")
        
        # Move player
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room_state[player_pos[0], player_pos[1]] = room_structure[player_pos[0], player_pos[1]]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room_state[next_pos[0], next_pos[1]] = 5
        
        # Update player position and track previous position
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        player_pos = next_pos
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        previous_positions.append(tuple(player_pos))
        
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        steps_taken += 1
        
        # Decide whether to continue moving
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if steps_taken >= max_steps or random.random() > continue_probability:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break
    
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return room_state



# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
Following code is adapted from the nicely written gym_sokoban repo
"""

# [EXPLAIN] `generate_room` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def generate_room(dim=(13, 13), p_change_directions=0.35, num_steps=25, num_boxes=3, tries=4, second_player=False, search_depth=100):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Generates a Sokoban room, represented by an integer matrix. The elements are encoded as follows:
    wall = 0
    empty space = 1
    box target = 2
    box not on target = 3
    box on target = 4
    player = 5

    :param dim:
    :param p_change_directions:
    :param num_steps:
    :param num_boxes:
    :param tries:
    :param second_player:
    :return: Numpy 2d Array, box mapping, action sequence
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    room_state = np.zeros(shape=dim)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    room_structure = np.zeros(shape=dim)

    # Some times rooms with a score == 0 are the only possibility.
    # In these case, we try another model.
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for t in range(tries):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room = room_topology_generation(dim, p_change_directions, num_steps)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room = place_boxes_and_player(room, num_boxes=num_boxes, second_player=second_player)

        # Room fixed represents all not movable parts of the room
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room_structure = np.copy(room)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room_structure[room_structure == 5] = 1

        # Room structure represents the current state of the room including movable parts
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room_state = room.copy()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room_state[room_state == 2] = 4

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room_state, box_mapping, action_sequence = reverse_playing(room_state, room_structure, search_depth)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room_state[room_state == 3] = 4

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if box_displacement_score(box_mapping) > 0:
            # [EXPLAIN] 現在の分岐または反復の制御を明示する。
            break

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if box_displacement_score(box_mapping) == 0:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise RuntimeWarning('Generated Model with score == 0')

    # Add random player movement after reverse_playing
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if box_displacement_score(box_mapping) == 1:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        move_probability = 0.8
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        move_probability = 0.5
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    room_state = add_random_player_movement(
        room_state, 
        room_structure,
        move_probability=move_probability,       # 50% chance the player will move
        continue_probability=0.5,   # 50% chance to continue moving after each step
        max_steps=3                 # Maximum of 3 steps
    )

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return room_structure, room_state, box_mapping, action_sequence


# [EXPLAIN] `room_topology_generation` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def room_topology_generation(dim=(10, 10), p_change_directions=0.35, num_steps=15):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Generate a room topology, which consits of empty floors and walls.

    :param dim:
    :param p_change_directions:
    :param num_steps:
    :return:
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dim_x, dim_y = dim

    # The ones in the mask represent all fields which will be set to floors
    # during the random walk. The centered one will be placed over the current
    # position of the walk.
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    masks = [
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0]
        ],
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0]
        ]
    ]

    # Possible directions during the walk
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    direction = random.sample(directions, 1)[0]

    # Starting position of random walk
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    position = np.array([
        random.randint(1, dim_x - 1),
        random.randint(1, dim_y - 1)]
    )

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    level = np.zeros(dim, dtype=int)

    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for s in range(num_steps):

        # Change direction randomly
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if random.random() < p_change_directions:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            direction = random.sample(directions, 1)[0]

        # Update position
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position = position + direction
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position[0] = max(min(position[0], dim_x - 2), 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        position[1] = max(min(position[1], dim_y - 2), 1)

        # Apply mask
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask = random.sample(masks, 1)[0]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask_start = position - 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        level[mask_start[0]:mask_start[0] + 3, mask_start[1]:mask_start[1] + 3] += mask

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    level[level > 0] = 1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    level[:, [0, dim_y - 1]] = 0
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    level[[0, dim_x - 1], :] = 0

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return level


# [EXPLAIN] `place_boxes_and_player` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def place_boxes_and_player(room, num_boxes, second_player):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Places the player and the boxes into the floors in a room.

    :param room:
    :param num_boxes:
    :return:
    """
    # Get all available positions
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    possible_positions = np.where(room == 1)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_possible_positions = possible_positions[0].shape[0]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_players = 2 if second_player else 1

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if num_possible_positions <= num_boxes + num_players:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise RuntimeError('Not enough free spots (#{}) to place {} player and {} boxes.'.format(
            num_possible_positions,
            num_players,
            num_boxes)
        )

    # Place player(s)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ind = np.random.randint(num_possible_positions)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    player_position = possible_positions[0][ind], possible_positions[1][ind]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    room[player_position] = 5

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if second_player:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ind = np.random.randint(num_possible_positions)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        player_position = possible_positions[0][ind], possible_positions[1][ind]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room[player_position] = 5

    # Place boxes
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for n in range(num_boxes):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        possible_positions = np.where(room == 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        num_possible_positions = possible_positions[0].shape[0]

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ind = np.random.randint(num_possible_positions)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        box_position = possible_positions[0][ind], possible_positions[1][ind]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room[box_position] = 2

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return room


# Global variables used for reverse playing.
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
explored_states = set()
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
num_boxes = 0
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
best_room_score = -1
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
best_room = None
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
best_box_mapping = None


# [EXPLAIN] `reverse_playing` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def reverse_playing(room_state, room_structure, search_depth=100):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    This function plays Sokoban reverse in a way, such that the player can
    move and pull boxes.
    It ensures a solvable level with all boxes not being placed on a box target.
    :param room_state:
    :param room_structure:
    :param search_depth:
    :return: 2d array, box mapping, action sequence
    """
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping, best_action_sequence

    # Box_Mapping is used to calculate the box displacement for every box
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    box_mapping = {}
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    box_locations = np.where(room_structure == 2)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    num_boxes = len(box_locations[0])
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for l in range(num_boxes):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        box = (box_locations[0][l], box_locations[1][l])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        box_mapping[box] = box

    # explored_states globally stores the best room state and score found during search
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    explored_states = set()
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    best_room_score = -1
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    best_room = None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    best_box_mapping = box_mapping
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    best_action_sequence = []

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=search_depth, action_sequence=[])

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return best_room, best_box_mapping, best_action_sequence


# [EXPLAIN] `depth_first_search` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=300, action_sequence=[]):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Searches through all possible states of the room.
    This is a recursive function, which stops if the ttl is reduced to 0 or
    over 1.000.000 states have been explored.
    :param room_state:
    :param room_structure:
    :param box_mapping:
    :param box_swaps:
    :param last_pull:
    :param ttl:
    :param action_sequence:
    :return:
    """
    # [EXPLAIN] 後続処理で参照する状態の寿命またはスコープを明示的に調整する。
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping, best_action_sequence

    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    ttl -= 1
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if ttl <= 0 or len(explored_states) >= 300000:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    state_tohash = marshal.dumps(room_state)

    # Only search this state, if it not yet has been explored
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if not (state_tohash in explored_states):

        # Add current state and its score to explored states
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room_score = box_swaps * box_displacement_score(box_mapping)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if np.where(room_state == 2)[0].shape[0] != num_boxes:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            room_score = 0

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if room_score > best_room_score:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            best_room = room_state.copy()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            best_room_score = room_score
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            best_box_mapping = box_mapping.copy()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            best_action_sequence = action_sequence.copy()

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        explored_states.add(state_tohash)

        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for action in ACTION_LOOKUP.keys():
            # The state and box mapping need to be copied to ensure
            # every action starts from a similar state.

            # TODO: A tentitive try here to make less moves
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if action >= 4:
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                continue

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            room_state_next = room_state.copy()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            box_mapping_next = box_mapping.copy()

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            room_state_next, box_mapping_next, last_pull_next = \
                reverse_move(room_state_next, room_structure, box_mapping_next, last_pull, action)

            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            box_swaps_next = box_swaps
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if last_pull_next != last_pull:
                # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
                box_swaps_next += 1
            
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            action_sequence_next = action_sequence + [action]
            # action_sequence_next = action_sequence + [(action, box_mapping_next != box_mapping)] # add whether a box is moved
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            depth_first_search(room_state_next, room_structure, box_mapping_next, box_swaps_next, last_pull_next, ttl, action_sequence_next)
            

# [EXPLAIN] `reverse_move` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def reverse_move(room_state, room_structure, box_mapping, last_pull, action):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Perform reverse action. Where all actions in the range [0, 3] correspond to
    push actions and the ones greater 3 are simmple move actions.
    :param room_state:
    :param room_structure:
    :param box_mapping:
    :param last_pull:
    :param action:
    :return:
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    player_position = np.where(room_state == 5)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    player_position = np.array([player_position[0][0], player_position[1][0]])

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    change = CHANGE_COORDINATES[action % 4]
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    next_position = player_position + change

    # Check if next position is an empty floor or an empty box target
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if room_state[next_position[0], next_position[1]] in [1, 2]:

        # Move player, independent of pull or move action.
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room_state[player_position[0], player_position[1]] = room_structure[player_position[0], player_position[1]]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        room_state[next_position[0], next_position[1]] = 5

        # In addition try to pull a box if the action is a pull action
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if action < 4:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            possible_box_location = change[0] * -1, change[1] * -1
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            possible_box_location += player_position

            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if room_state[possible_box_location[0], possible_box_location[1]] in [3, 4]:
                # Perform pull of the adjacent box
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                room_state[player_position[0], player_position[1]] = 3
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                room_state[possible_box_location[0], possible_box_location[1]] = room_structure[
                    possible_box_location[0], possible_box_location[1]]

                # Update the box mapping
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for k in box_mapping.keys():
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if box_mapping[k] == (possible_box_location[0], possible_box_location[1]):
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        box_mapping[k] = (player_position[0], player_position[1])
                        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                        last_pull = k

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return room_state, box_mapping, last_pull


# [EXPLAIN] `box_displacement_score` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def box_displacement_score(box_mapping):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Calculates the sum of all Manhattan distances, between the boxes
    and their origin box targets.
    :param box_mapping:
    :return:
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    score = 0
    
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for box_target in box_mapping.keys():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        box_location = np.array(box_mapping[box_target])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        box_target = np.array(box_target)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dist = np.sum(np.abs(box_location - box_target))
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        score += dist

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return score


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
TYPE_LOOKUP = {
    0: 'wall',
    1: 'empty space',
    2: 'box target',
    3: 'box on target',
    4: 'box not on target',
    5: 'player'
}

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'move up',
    5: 'move down',
    6: 'move left',
    7: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}
