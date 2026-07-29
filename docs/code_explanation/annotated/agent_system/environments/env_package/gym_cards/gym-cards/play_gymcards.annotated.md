# play_gymcards.ipynb セル別参照

## Cell 1: code

```python
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_cards
```

## Cell 2: code

```python
env = gym.make('gym_cards/NumberLine-v0')
obs, info = env.reset()
plt.imshow(obs)
```

## Cell 3: code

```python
obs, reward, terminated, truncated, info = env.step(1)
plt.imshow(obs)
print(reward, terminated, truncated, info)
```

## Cell 4: code

```python
env = gym.make('gym_cards/Blackjack-v0')
obs, info = env.reset()
plt.imshow(obs)
```

## Cell 5: code

```python
obs, reward, terminated, truncated, info = env.step(1)
plt.imshow(obs)
print(reward, terminated, truncated, info)
```

## Cell 6: code

```python
env = gym.make('gym_cards/EZPoints-v0')
obs, info = env.reset()
plt.imshow(obs)
```

## Cell 7: code

```python
obs, reward, terminated, truncated, info = env.step(1)
plt.imshow(obs)
print(reward, terminated, truncated, info)
```

## Cell 8: code

```python
env = gym.make('gym_cards/Points24-v0')
obs, info = env.reset()
plt.imshow(obs)
```

## Cell 9: code

```python
obs, reward, terminated, truncated, info = env.step(1)
plt.imshow(obs)
print(reward, terminated, truncated, info)
```

## Cell 10: code

```python

```
