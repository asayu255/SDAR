# play_gymcards_test_text_wrap.ipynb セル別参照

## Cell 1: code

```python
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_cards
from text_wrapper import info_to_text_obs, text_projection
```

## Cell 2: code

```python
env_name = 'gym_cards/NumberLine-v0'
env = gym.make(env_name)
obs, info = env.reset()
plt.imshow(obs)
text_obs = info_to_text_obs(env_name, info)
print(info)
print(text_obs)
```

## Cell 3: code

```python
"""
Please check the relationship between the symbolic operators
and the numbers here:
https://github.com/RL4VLM/RL4VLM/blob/main/gym-cards/gym_cards/envs/numberline.py#L17

"""
obs, reward, terminated, truncated, info = env.step(1)
plt.imshow(obs)
print(reward, terminated, truncated, info)
```

## Cell 4: code

```python
## Please make sure the string contain the format "action": {act}"
act = text_projection(['"action": "+"'], env_name)
obs, reward, terminated, truncated, info = env.step(act)
plt.imshow(obs)
print(reward, terminated, truncated, info)
```

## Cell 5: code

```python
text_projection(["action: +"], env_name)
```

## Cell 6: code

```python
env_name = 'gym_cards/Blackjack-v0'
env = gym.make(env_name)
obs, info = env.reset()
plt.imshow(obs)
text_obs = info_to_text_obs(env_name, info)
print(info)
print(text_obs)
```

## Cell 7: code

```python
"""
Please check the relationship between the symbolic operators
and the numbers here:
https://github.com/RL4VLM/RL4VLM/blob/main/gym-cards/gym_cards/envs/blackjack.py#L106

"""
obs, reward, terminated, truncated, info = env.step(1)
plt.imshow(obs)
text_obs = info_to_text_obs(env_name, info)
print(reward, terminated, truncated, info)
print(text_obs)
```

## Cell 8: code

```python
## Please make sure the string contain the format "actopm: {act}"
act = text_projection(['"action": "hit"'], env_name)
obs, reward, terminated, truncated, info = env.step(act)
plt.imshow(obs)
text_obs = info_to_text_obs(env_name, info)
print(reward, terminated, truncated, info)
print(text_obs)
```

## Cell 9: code

```python
env_name = 'gym_cards/EZPoints-v0'
env = gym.make(env_name)
obs, info = env.reset()
plt.imshow(obs)
text_obs = info_to_text_obs(env_name, info)
print(info)
print(text_obs)
```

## Cell 10: code

```python
"""
Please check the relationship between the symbolic operators
and the numbers here:
https://github.com/RL4VLM/RL4VLM/blob/main/gym-cards/gym_cards/envs/ezpoints.py#L26

"""
obs, reward, terminated, truncated, info = env.step(2)
plt.imshow(obs)
text_obs = info_to_text_obs(env_name, info)
print(reward, terminated, truncated, info)
print(text_obs)
```

## Cell 11: code

```python
## Please make sure the string contain the format "actopm: {act}"
act = text_projection(['"action": "+"'], env_name)
obs, reward, terminated, truncated, info = env.step(act)
plt.imshow(obs)
text_obs = info_to_text_obs(env_name, info)
print(reward, terminated, truncated, info)
print(text_obs)
```

## Cell 12: code

```python
env_name = 'gym_cards/Points24-v0'
env = gym.make(env_name)
obs, info = env.reset()
plt.imshow(obs)
text_obs = info_to_text_obs(env_name, info)
print(info)
print(text_obs)
```

## Cell 13: code

```python
"""
Please check the relationship between the symbolic operators
and the numbers here:
https://github.com/RL4VLM/RL4VLM/blob/main/gym-cards/gym_cards/envs/points.py#L27

"""
obs, reward, terminated, truncated, info = env.step(4)
plt.imshow(obs)
text_obs = info_to_text_obs(env_name, info)
print(reward, terminated, truncated, info)
print(text_obs)
```

## Cell 14: code

```python
## Please make sure the string contain the format "actopm: {act}"
act = text_projection(['"action": "*"'], env_name)
obs, reward, terminated, truncated, info = env.step(act)
plt.imshow(obs)
text_obs = info_to_text_obs(env_name, info)
print(reward, terminated, truncated, info)
print(text_obs)
```

## Cell 15: code

```python

```
