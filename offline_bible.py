import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random
import argparse
import os

from collections import defaultdict 

class GridWorld:
    def __init__(
        self, 
        size=4,
        terminal_states=((0,0), (3,3)), 
        reward=-1
    ):
        self.size = size
        self.terminal_states = terminal_states
        self.reward = reward
        self.actions = ('U', 'D', 'L', 'R')
        self.reset()

    def is_terminal(self, state):
        return state in self.terminal_states
    
    def reset(self):
        self.state = (np.random.randint(self.size), np.random.randint(self.size))
        while self.state in self.terminal_states:
            self.state = (np.random.randint(self.size), np.random.randint(self.size))
        return self.state

    def step(self, action):
        if self.is_terminal(self.state):
            return self.state, 0, True

        x, y = self.state
        if action == 'U': x = max(x - 1, 0)
        if action == 'D': x = min(x + 1, self.size - 1)
        if action == 'L': y = max(y - 1, 0)
        if action == 'R': y = min(y + 1, self.size - 1)
        self.state = (x, y)
        
        return self.state, self.reward, self.is_terminal(self.state)

class WindyGridWorld:
    def __init__(
        self, 
        size=(7, 10),
        start=(3, 0),
        goal=(3, 7),
        wind=(0, 0, 0, 1, 1, 1, 2, 2, 1, 0),
        reward=-1
    ):
        self.size = size
        self.start_state = start
        self.terminal_state = goal
        self.wind = wind
        self.reward = reward
        # Up, Down, Left, Right
        self.actions = (0, 1, 2, 3)
        self.reset()

    def is_terminal(self, state):
        return state == self.terminal_state
    
    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action, state=None):
        if state is None:
            state = self.state

        if self.is_terminal(state):
            return state, 0, True
        
        x, y = state
        if action == 0: x = max(x - 1, 0)
        if action == 1: x = min(x + 1, self.size[0] - 1)
        if action == 2: y = max(y - 1, 0)
        if action == 3: y = min(y + 1, self.size[1] - 1)
        
        # Apply wind effect
        x = max(min(x - self.wind[y], self.size[0] - 1), 0)
        
        self.state = (x, y)
        
        return self.state, self.reward, self.is_terminal(self.state)

def uniform_random_policy(state):
    """
    Uniform random policy: returns a random action.
    """
    
    return np.random.choice(['U', 'D', 'L', 'R'])

def greedy_policy(env, state, Q):
    """
    Greedy policy: returns the action with the highest Q-value.
    """
    
    return np.argmax(Q[state])

def epsilon_greedy_policy(env, state, Q, epsilon=0.3):
    """
    Epsilon-greedy policy: returns the best action with probability 1 - epsilon
    and a random action with probability epsilon.
    """
    
    if np.random.random() < epsilon:
        return np.random.choice(range(len(env.actions)))
    else:
        return np.argmax(Q[state])

def generate_episode(env, policy):
    """
    Generates an episode following a given policy.
    """
    
    episode = []
    state = env.reset()
    done = False
    
    while not done:
        action = policy(state)
        
        next_state, reward, done = env.step(action)

        transition = (state, action, reward)
        episode.append(transition)

        state = next_state
    
    return episode


def first_visit_mc_prediction(env, policy, num_episodes=500, gamma=1.0):
    V = defaultdict(float)   # value function: S -> v(S)
    returns = defaultdict(list)  # return statistics
    value_snapshots = []  # Store value function snapshots for animation

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
        
        # Save the value function snapshot at every episode
        snapshot = np.zeros((env.size, env.size))
        for (i, j), value in V.items():
            snapshot[i, j] = value
        value_snapshots.append(snapshot.copy())

    return V, value_snapshots

def every_visit_mc_prediction(env, policy, num_episodes=500, gamma=1.0):
    V = defaultdict(float)   # value function: S -> v(S)
    returns = defaultdict(list)  # return statistics
    value_snapshots = []  # Store value function snapshots for animation

    for _ in range(num_episodes):

        episode_returns = {}
        n_episode_returns = {}
        episode = generate_episode(env, policy)
        
        G = 0
        
        for state, _ , reward in reversed(episode):
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

    return V, value_snapshots

def td_lambda(env, policy, num_episodes=500, gamma=1.0, lamnda=0.6, alpha=0.1):
    V = defaultdict(float)   # value function: S -> v(S)
    value_snapshots = []  # Store value function snapshots for animation

    for _ in range(num_episodes):

        episode = generate_episode(env, policy)

        E = defaultdict(float)  # Eligibility traces
        
        for t in range(len(episode)):
            state, _, reward = episode[t]

            if t == len(episode) - 1:
                next_state = None
            else:
                next_state, _, _ = episode[t + 1]

            # Update the temporal difference error
            if next_state is None:
                delta = reward - V[state]
            else:
                delta = reward + gamma * V[next_state] - V[state]

            # Update eligibility traces
            for s in E.keys():
                E[s] *= gamma * lamnda

            E[state] += 1.0
            
            # Update the value function based on eligibility traces
            for s in E.keys():
                V[s] += alpha * delta * E[s]

        # Save the value function snapshot at every episode
        snapshot = np.zeros((env.size, env.size))
        for (i, j), value in V.items():
            snapshot[i, j] = value
        value_snapshots.append(snapshot.copy())

    return V, value_snapshots

def sarsa_on_policy(env, policy, num_episodes=500, gamma=1.0, alpha=0.1):
    Q = defaultdict(lambda: np.zeros(len(env.actions)))  # Q-function: Q(S, A) -> q(S, A)
    value_snapshots = []  # Store value function snapshots for animation
    
    for e in range(num_episodes):
        # Initialize state and action
        state = env.reset()
        # Choose action from state using policy derived from Q (e.g., epsilon-greedy)
        action = policy(env, state, Q)
        # Repeat for each step of episode
        done = False
        while not done:
            # Take action A, observe R, S'
            next_state, reward, done = env.step(action, state)
            # Choose action A' from S' using policy derived from Q (e.g., epsilon-greedy)
            next_action = policy(env, next_state, Q)
            # Update Q(S, A)
            Q[state][env.actions.index(action)] += alpha * (reward + gamma * Q[next_state][env.actions.index(next_action)] - Q[state][env.actions.index(action)])
            # S <- S'; A <- A'
            state = next_state
            action = next_action

    # Save the value function snapshot at every episode
        snapshot = np.zeros(env.size)
        for (i, j), value in Q.items():
            snapshot[i, j] = max(value)
        value_snapshots.append(snapshot.copy())

    return Q, value_snapshots

def sarsa_lambda(env, policy, num_episodes=500, gamma=1.0, lamnda=0.6, alpha=0.1):
    Q = defaultdict(lambda: np.zeros(len(env.actions)))  # Q-function: Q(S, A) -> q(S, A)
    value_snapshots = []  # Store value function snapshots for animation
    
    for e in range(num_episodes):
        # Initialize state and action
        state = env.reset()
        action = policy(env, state, Q)
        E = defaultdict(float)  # Eligibility traces
        
        done = False
        while not done:
            # Take action A, observe R, S'
            next_state, reward, done = env.step(action, state)
            next_action = policy(env, next_state, Q)
            
            # Update the temporal difference error
            delta = reward + gamma * Q[next_state][env.actions.index(next_action)] - Q[state][env.actions.index(action)]
            
            # Update eligibility traces
            E[(state, action)] += 1.0
            
            # Update Q-values and eligibility traces
            for s, a in E.keys():
                Q[s][env.actions.index(a)] += alpha * delta * E[(s, a)]
                E[(s, a)] *= gamma * lamnda
            
            # S <- S'; A <- A'
            state = next_state
            action = next_action

        # Save the value function snapshot at every episode
        snapshot = np.zeros(env.size)
        for (i, j), value in Q.items():
            snapshot[i, j] = max(value)
        value_snapshots.append(snapshot.copy())

    return Q, value_snapshots


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

def plot_policy(Q, size=(7, 10), episode=0):
    policy_grid = np.zeros(size)
    for (i,j), actions in Q.items():
        policy_grid[i, j] = np.argmax(actions)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(policy_grid, cmap='coolwarm', interpolation='nearest')
    for i in range(size[0]):
        for j in range(size[1]):
            plt.text(j, i, f"{policy_grid[i, j]:.1f}", ha='center', va='center', color='black')
    plt.title(f"Optimal Policy (Episode {episode})")
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

def animate_value_function(value_snapshots):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(value_snapshots[0], cmap='coolwarm', interpolation='nearest')
    ax.set_title("State Value Function Evolution")

    def update(frame):
        im.set_array(value_snapshots[frame])
        ax.set_title(f"State Value Function (Episode {frame+1})")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(value_snapshots), interval=200)
    
    ani.save("monte_carlo_evolution.mp4", writer="ffmpeg", fps=10)


def main(N):

    # Run Monte Carlo Policy Evaluation
    # env = GridWorld()
    # _, value_snapshots = first_visit_mc_prediction(env, uniform_random_policy, num_episodes=N, gamma=0.9, lamnda=0.6, alpha=0.1)
    # plot_value_function(value_snapshots[-1], env.size, N)

    # Run Every Visit Monte Carlo Policy Evaluation
    # env = GridWorld()
    # _, value_snapshots = every_visit_mc_prediction(env, uniform_random_policy, num_episodes=N, gamma=0.9)
    # plot_value_function(value_snapshots[-1], env.size, N)

    # Run Temporal Difference Learning
    # env = GridWorld()
    # _, value_snapshots = td_lambda(env, uniform_random_policy, num_episodes=N, gamma=0.9, lamnda=0.6, alpha=0.1)
    # plot_value_function(value_snapshots[-1], env.size, N)

    # Run SARSA On-Policy Temporal Difference Learning
    # env = WindyGridWorld()
    # Q, _ = sarsa_on_policy(env, epsilon_greedy_policy, num_episodes=N, gamma=0.9, alpha=0.1)
    # print_grid(Q, env.size, N, env.start_state, env.terminal_state)
    # plot_policy(Q, env.size, N)

    # Run SARSA-lambda On-Policy Temporal Difference Learning - WIP
    env = WindyGridWorld()
    Q, _ = sarsa_lambda(env, epsilon_greedy_policy, num_episodes=N, gamma=0.9, lamnda=0.6, alpha=0.1)
    print_grid(Q, env.size, N, env.start_state, env.terminal_state)
    plot_policy(Q, env.size, N)

    # Animate the value function evolution
    # animate_value_function(value_snapshots)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridWorld Monte Carlo Skeleton")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to run")
    args = parser.parse_args()
    
    main(args.episodes)