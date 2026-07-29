<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# Visual Card Games at Gym

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
<p align="center">
  <img src="../imgs/nl.png" alt="nl" width="150"/>
  <img src="../imgs/ezp.png" alt="ezp" width="150"/>
  <img src="../imgs/p24.png" alt="p24" width="150"/>
  <img src="../imgs/bj.png" alt="bj" width="150"/>
</p>

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
A Custom Gym Environment containing these four games: Numberline, EZPoints, Points24, and Blackjack.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Given pixel-based observation of game state, the agent need to write the solution by typing unit by unit.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Install

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
This env is based on **gymnasium**.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Use

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```
cd gym-cards
pip install -e .
```

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Usage

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
We provide a simple usage example below, also see a more detailed usage [here](./play_gymcards.ipynb).
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```
import gym_cards
import gymnasium as gym
# numberline
env = gym.make('gym_cards/NumberLine-v0')
# EZPoints
env = gym.make('gym_cards/EZPoints-v0')
# 24game
env = gym.make('gym_cards/Points24-v0')
# blackjack game
env = gym.make('gym_cards/Blackjack-v0')
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Blackjack also has a symbolic version, which is the same as the original gymnasium version.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```
env = gym.make('gym_cards/Blackjack-v0', is_pixel=False)
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Then you can use the env as a normal gym env.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```
obs, info = env.reset()
op = 1
obs, reward, terminated, truncated, info = env.step(op)
```

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Doc

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### NumberLine-v0

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
The Number Line Environment is a custom Gym environment that simulates a simple number line. It is designed for easy debugging.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Actions**
The environment accepts two discrete actions:
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `0`: Move left (decrease the current position by 1, if greater than 0).
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `1`: Move right (increase the current position by 1, if less than `max_position`).

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Observation Space**
The observation is an RGB image with the following characteristics:
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Shape: (200, 200, 3)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Pixel values range from 0 to 255.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- The image displays the current and goal positions on the number line.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Termination**
An episode ends when the current position reaches the goal position.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Reward**
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `1`: If the current position is the same as the goal position.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `-1`: is given if the action does not move the current position closer to the goal position.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `0`: Otherwise.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Initialization Options**
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `max_position` (default: 5): Sets the maximum value on the number line. Both the start and goal positions are randomly initialized between 0 and `max_position`.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Notes**
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- The start and goal positions are randomized at the beginning of each episode.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- If after 2 * max_position steps the target still is not meet, the episode is terminated and the environment is reset.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- The environment ensures that the start and goal positions are not initially the same.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Observations are returned as RGB images with the current and goal positions clearly labeled.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### EZPoints-v0


<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Actions**

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **0-9**: Represent numbers 1-10
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **10-12**: Operators and control actions: '+', '\*', '='

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Observations**

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
A numpy uint8 array of shape (300,300,3).

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Termination**

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- If an invalid action is taken.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- If the formula length exceeds 5.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- If '=' action is taken, the formula is evaluated.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Reward**

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **10**: If the formula evaluates to the target_points.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **-1**: if an invalid action is taken or the result isn't correct.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **0**: Otherwise.


<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Points24-v0


<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Initialization Options**

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `treat_face_cards_as_10`: Treats face cards J, Q, K as 10 (default is True).
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `target_points`: The target sum to reach (default is 24).

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Actions**

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- When `treat_face_cards_as_10=True`:

  <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
  - **0-9**: Represent numbers 1-10
  <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
  - **10-16**: Operators and control actions: '+', '-', '\*', '/', '(', ')', '='

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- When `treat_face_cards_as_10=False`:
  <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
  - **0-12**: Represent numbers 1-13
  <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
  - **13-19**: Operators and control actions: '+', '-', '\*', '/', '(', ')', '='

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Observations**

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
A numpy uint8 array of shape (300,300,3).

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Termination**

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- If an invalid action is taken.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- If the formula length exceeds 20.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- If '=' action is taken, the formula is evaluated.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Reward**

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **10**: If the formula evaluates to the target_points.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **-1**: if an invalid action is taken or the result isn't correct.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- **0**: Otherwise.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### Blackjack-v0

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Please refer to gymnasium's Blackjack implementation [here](https://gymnasium.farama.org/environments/toy_text/blackjack/). Everything is the same except for the observation space being purely pixel-based.

