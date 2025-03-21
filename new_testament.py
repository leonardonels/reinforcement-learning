# https://gymnasium.farama.org/environments/toy_text/frozen_lake/

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import gymnasium as gym
import os

from typing import Optional
from collections import defaultdict 


## envs:
class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 4, terminal_state = np.array([3, 3], dtype=np.int32)):
        self.size = size

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = terminal_state

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

    #reset the environmnet at teh start of an episode
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        self._agent_location = np.random.randint(0, self.size, size=2)
        while np.array_equal(self._agent_location, self._target_location):
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        return self._agent_location

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 0 if terminated else -1

        return self._agent_location, reward, terminated

# windy_grid

# slippery_grid

# blackjack

# pacman


## utils:
def uniform_random_policy(env):
    """
    retunr a uniform random choice from the action space
    """

    return env.action_space.sample()

def generate_episode(env, policy):
    """
    Generates an episode following a given policy.
    """
    
    episode = []
    state = env.reset()
    done = False
    
    
    while not done:
        action = policy(env)
        
        next_state, reward, done = env.step(action)

        transition = (state, action, reward)
        episode.append(transition)

        state = next_state
    
    return episode

def manhattan_distance(state, terminal_state, alpha=-.1):
    i, j = state
    t_i, t_j = terminal_state
    distance = abs(i - t_i) + abs(j - t_j)  # Manhattan distance

    reward = - alpha * distance
    return reward

## plotting:
def plot_value_function(V, size=(4,4), episode=0):
    if not isinstance(size, tuple):
        size = (size, size)
        
    plt.figure(figsize=(6, 6))
    plt.imshow(V, cmap='coolwarm', interpolation='nearest')
    for i in range(size[0]):
        for j in range(size[1]):
            plt.text(j, i, f"{V[i, j]:.1f}", ha='center', va='center', color='black')
    plt.title(f"State Value Function (Episode {episode})")
    plt.colorbar()
    plt.show()

def print_grid(Q, size=(7, 10), episode=0, start_state=None, terminal_state=None):
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear terminal
    print(f"Total Episodes: {episode}\n")
    print("GridWorld State Values & Policy:")
    
    grid = np.zeros(size)
    policy = np.full(size, ' ')
    arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    for (i, j), actions in Q.items():
        grid[i, j] = max(actions)
        policy[i, j] = arrows[np.argmax(actions)]
    
    if start_state:
        policy[start_state] = 'S'
    if terminal_state:
        policy[terminal_state] = 'G'
    
    for i in range(grid.shape[0]):
        row_values = " ".join(f"{grid[i, j]:6.2f}" for j in range(grid.shape[1]))
        row_policy = " ".join(policy[i, j] for j in range(grid.shape[1]))
        print(f"{row_values}    {row_policy}")
    print("\n")

def print_value(V, actions=None, terminal_state=None, episode=0):
    V[tuple(terminal_state)] = 0.
    max_i = max(k[0] for k in V.keys()) + 1
    max_j = max(k[1] for k in V.keys()) + 1
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Terminal State: {terminal_state}\n")
    print("GridWorld State Values")
    
    for i in range(max_i):
        row_values = " ".join(f"{V[(i, j)]:6.2f}" if (i, j) in V else "{:6}".format('') for j in range(max_j))
        print(f"{row_values}")
    
    if not episode:
        print("\nUpdating...\n")
    else:
        print(f"\nUpdating...{episode}\n")


## algorithms:
"""first visit montecarlo prediction (planning), policy evaluation, entire episode must be generated"""
def first_visit_mc_prediction(env, policy, num_episodes=500, gamma=1.0, visualize=False):
    V = defaultdict(float)
    returns = defaultdict(list)
    value_snapshots = []

    for num_episode_ in range(num_episodes):

        episode_returns = {}
        episode = generate_episode(env, policy)
        
        G = 0
        
        for state, action, reward in reversed(episode):
            state = tuple(state)

            G = reward + gamma * G
            episode_returns[state] = G

        for state, discounted_return in episode_returns.items():
            returns[state].append(discounted_return)
            
        V[state] = np.mean(returns[state])

        # Save the value function snapshot at every episode
        snapshot = np.zeros((env.size, env.size))
        for (i, j), value in V.items():
            snapshot[i, j] = value
        value_snapshots.append(snapshot.copy())

        if visualize:
            print_value(V, terminal_state=env._target_location, episode=num_episode_+1)

    return V, V, value_snapshots

"""first visit montecarlo prediction (planning) with the manhattan distance as reward, policy evaluation, entire episode must be generated"""
def manhattan_mc_prediction(env, policy, num_episodes=500, gamma=1.0, alpha=1.0, visualize=False):
    V = defaultdict(float)
    returns = defaultdict(list)
    value_snapshots = []

    for num_episode_ in range(num_episodes):

        episode_returns = {}
        episode = generate_episode(env, policy)
        
        G = 0
        
        for state, action, reward in reversed(episode):
            state = tuple(state)
            reward = manhattan_distance(state, tuple(env._target_location), alpha=alpha)

            G = reward + gamma * G
            episode_returns[state] = G

        for state, discounted_return in episode_returns.items():
            returns[state].append(discounted_return)
            
        V[state] = np.mean(returns[state])

        # Save the value function snapshot at every episode
        snapshot = np.zeros((env.size, env.size))
        for (i, j), value in V.items():
            snapshot[i, j] = value
        value_snapshots.append(snapshot.copy())

        if visualize:
            print_value(V, terminal_state=env._target_location, episode=num_episode_+1)

    return V, V, value_snapshots

"""every visit montecarlo prediction (planning), policy evaluation, entire episode must be generated"""
def every_visit_mc_prediction(env, policy, num_episodes=500, gamma=1.0, visualize=False):
    V = defaultdict(float)
    returns = defaultdict(list)
    value_snapshots = []

    for num_episode_ in range(num_episodes):

        episode_returns = {}
        n_episode_returns = {}
        episode = generate_episode(env, policy)
        
        G = 0
        
        for state, _ , reward in reversed(episode):
            state = tuple(state)

            if state not in episode_returns:
                episode_returns[state] = 0
                n_episode_returns[state]= 0

            G = reward + gamma * G
            episode_returns[state] = ((episode_returns[state]*n_episode_returns[state])+G)/(n_episode_returns[state]+1)
            episode_returns[state] += 1


        for state, discounted_return in episode_returns.items():
            returns[state].append(discounted_return)
            
        V[state] = np.mean(returns[state])
        
        # Save the value function snapshot at every episode
        snapshot = np.zeros((env.size, env.size))
        for (i, j), value in V.items():
            snapshot[i, j] = value
        value_snapshots.append(snapshot.copy())

        if visualize:
            print_value(V, terminal_state=env._target_location, episode=num_episode_+1)


    return V, V, value_snapshots

# td_0

# td_lambda

# sarsa

# sarsa_lambda

# q_learning

# q_lambda


def main(N):
    env = GridWorldEnv(size=4)
    _ , _ , value_snapshots = manhattan_mc_prediction(env, uniform_random_policy, num_episodes=N, gamma=0)
    plot_value_function(value_snapshots[-1], env.size, N)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridWorld Monte Carlo Skeleton")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to run")
    args = parser.parse_args()
    
    main(args.episodes)