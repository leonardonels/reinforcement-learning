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
#
#    “S” for Start tile
#    “G” for Goal tile
#    “F” for frozen tile
#    “H” for a tile with a hole


# simple_grid
class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 4):
        self.size = size

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        #reset the environmnet at teh start of an episode

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
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

## algorithms:
# first_visit_montecarlo_manhattan

# first_visit_montecarlo
    # prediction (planning)
        # policy evaluation
            # episode must be generated
def first_visit_mc_prediction(env, policy, num_episodes=500, gamma=1.0):
    V = defaultdict(float)   # value function: S -> v(S)
    returns = defaultdict(list)  # return statistics

    for _ in range(num_episodes):

        episode_returns = {}
        episode = generate_episode(env, policy)
        
        G = 0
        
        for state, action, reward in reversed(episode):

            G = reward + gamma * G
            episode_returns[state] = G


        for state, discounted_return in episode_returns.items():
            returns[state].append(discounted_return)
            
        V[state] = np.mean(returns[state])

    return V

# every_visit_montecarlo

# td_0

# td_lambda

# sarsa

# sarsa_lambda

# q_learning

# q_lambda


def main(N):
    # Run first visit Monte Carlo Policy Evaluation
    env = GridWorldEnv(size=4)
    V = first_visit_mc_prediction(env, uniform_random_policy, num_episodes=N, gamma=0.9)
    plot_value_function(V, env.size, N)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridWorld Monte Carlo Skeleton")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to run")
    args = parser.parse_args()
    
    main(args.episodes)