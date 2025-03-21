import gymnasium as gym
import random as rnd

N = 10

for _ in range(N):
    map=["G"]
    s = rnd.randint(1, 14)
    for i in range(1, 15):
        if i == s:
            map.append("S")
        else:
            map.append("F")
    map.append("G")

    desc = [map[i:i+4] for i in range(0, len(map), 4)]

    print(desc)

    env = gym.make('FrozenLake-v1', desc=desc, map_name=f"{4}x{4}", is_slippery=False, render_mode="human")
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

    env.close()