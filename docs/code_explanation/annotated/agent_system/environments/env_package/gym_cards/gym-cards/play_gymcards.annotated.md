# play_gymcards.ipynb セル別解説

<!-- [EXPLAIN] Notebook JSON は変更せず、cell 順序を保った解説を作成する。 -->

## Cell 1: code

<!-- [EXPLAIN] この cell の原文と実行・説明上の役割を対応付ける。 -->

```python
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_cards
```

## Cell 2: code

<!-- [EXPLAIN] この cell の原文と実行・説明上の役割を対応付ける。 -->

```python
env = gym.make('gym_cards/NumberLine-v0')
obs, info = env.reset()
plt.imshow(obs)
```

## Cell 3: code

<!-- [EXPLAIN] この cell の原文と実行・説明上の役割を対応付ける。 -->

```python
obs, reward, terminated, truncated, info = env.step(1)
plt.imshow(obs)
print(reward, terminated, truncated, info)
```

## Cell 4: code

<!-- [EXPLAIN] この cell の原文と実行・説明上の役割を対応付ける。 -->

```python
env = gym.make('gym_cards/Blackjack-v0')
obs, info = env.reset()
plt.imshow(obs)
```

## Cell 5: code

<!-- [EXPLAIN] この cell の原文と実行・説明上の役割を対応付ける。 -->

```python
obs, reward, terminated, truncated, info = env.step(1)
plt.imshow(obs)
print(reward, terminated, truncated, info)
```

## Cell 6: code

<!-- [EXPLAIN] この cell の原文と実行・説明上の役割を対応付ける。 -->

```python
env = gym.make('gym_cards/EZPoints-v0')
obs, info = env.reset()
plt.imshow(obs)
```

## Cell 7: code

<!-- [EXPLAIN] この cell の原文と実行・説明上の役割を対応付ける。 -->

```python
obs, reward, terminated, truncated, info = env.step(1)
plt.imshow(obs)
print(reward, terminated, truncated, info)
```

## Cell 8: code

<!-- [EXPLAIN] この cell の原文と実行・説明上の役割を対応付ける。 -->

```python
env = gym.make('gym_cards/Points24-v0')
obs, info = env.reset()
plt.imshow(obs)
```

## Cell 9: code

<!-- [EXPLAIN] この cell の原文と実行・説明上の役割を対応付ける。 -->

```python
obs, reward, terminated, truncated, info = env.step(1)
plt.imshow(obs)
print(reward, terminated, truncated, info)
```

## Cell 10: code

<!-- [EXPLAIN] この cell の原文と実行・説明上の役割を対応付ける。 -->

```python

```
