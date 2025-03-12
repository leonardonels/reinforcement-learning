import numpy as np
import time
import random
import argparse
import os

def create_gridworld(size):
    return np.zeros(size)

def choose_random_state(size):
    w, h = size
    return random.randint(0, w - 1), random.randint(0, h - 1)

def get_best_action(grid, i, j):
    #Determines the best action based on the highest-value neighboring state.
    directions = {
        '↑': (i - 1, j),
        '↓': (i + 1, j),
        '←': (i, j - 1),
        '→': (i, j + 1)
    }
    best_action = '·'  # Default if no best move is found
    best_value = -float('inf')
    
    for action, (ni, nj) in directions.items():
        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
            if grid[ni, nj] > best_value:
                best_value = grid[ni, nj]
                best_action = action
    
    return best_action

def compute_policy(grid, terminal_state):
    #Computes a policy where each state points to the highest-value neighbor.
    policy = np.full(grid.shape, '·', dtype='<U1')
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if (i, j) == terminal_state:
                policy[i, j] = 'T'  # Mark terminal state
            else:
                policy[i, j] = get_best_action(grid, i, j)
    return policy

def print_grid(grid, policy=None, terminal_state=None, episode=0):
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear terminal
    print(f"Terminal State: {terminal_state}\n")
    print("GridWorld State Values & Policy:")
    for i in range(grid.shape[0]):
        row_values = " ".join(f"{grid[i, j]:6.2f}" for j in range(grid.shape[1]))
        row_policy = " ".join(policy[i]) if policy is not None else ""
        print(f"{row_values}    {row_policy}")
    if not episode:
        print("\nUpdating...\n")
    else:
        print(f"\nUpdating...{episode}\n")
    time.sleep(0.01)

def dummy_explore(grid, episodes=10, start_state=None, terminal_state=None):
    size = grid.shape
    for _ in range(episodes):
        state = choose_random_state(size)
        for _ in range(10):  # Simulate actions for 10 steps
            next_state = choose_random_state(size)
            grid[next_state] += random.uniform(-0.1, 0.1)  # Update value randomly
            policy = compute_policy(grid, terminal_state)
            print_grid(grid, policy, terminal_state)

def choose_next_state(grid, state):
    i, j = state  # row, column
    possible_moves = []

    # Up (row - 1)
    if i - 1 >= 0:
        possible_moves.append((i - 1, j))
    # Down (row + 1)
    if i + 1 < grid.shape[0]:
        possible_moves.append((i + 1, j))
    # Left (column - 1)
    if j - 1 >= 0:
        possible_moves.append((i, j - 1))
    # Right (column + 1)
    if j + 1 < grid.shape[1]:
        possible_moves.append((i, j + 1))

    return random.choice(possible_moves) if possible_moves else state

def distance_reward(state, terminal_state):
    #Calculate a reward based on the Manhattan distance to the terminal state.
    i, j = state
    t_i, t_j = terminal_state
    distance = abs(i - t_i) + abs(j - t_j)  # Manhattan distance

    reward = -distance * 0.1
    return reward

def first_visit_montecarlo(grid, episodes, terminal_state): # Gamma = 0
    returns = { (i, j): [] for i in range(grid.shape[0]) for j in range(grid.shape[1]) }

    for _ in range(episodes):
        last_policy = compute_policy(grid, terminal_state)
        state = choose_random_state(grid.shape)
        while state == terminal_state:
            state = choose_random_state(grid.shape)

        episode = []
        while state != terminal_state:
            reward = -0.1
            episode.append((state, reward))
            state = choose_next_state(grid, state)

        visited_states = set()
        for (s, reward) in episode.reverse():
            if s not in visited_states:
                returns[s].append(reward)
                grid[s] = np.mean(returns[s])
                visited_states.add(s)

        new_policy = compute_policy(grid, terminal_state)
        if np.array_equal(last_policy, new_policy):
            break

        print_grid(grid, new_policy, terminal_state)

def first_visit_montecarlo(grid, episodes, terminal_state, gamma=0.9):
    returns = {}
    visits = {}
    
    for episode in range(1, episodes + 1):
        last_policy = compute_policy(grid, terminal_state)
        state = choose_random_state(grid.shape)
        trajectory = []
        
        while state != terminal_state:
            reward = -1
            trajectory.append((state, reward))
            state = choose_next_state(grid, state) 
        
        G = 0
        visited_states = set()
        for t in range(len(trajectory) - 1, -1, -1):
            s, r = trajectory[t]
            G = r + gamma * G
            
            if s not in visited_states:
                visited_states.add(s)
                if s not in returns:
                    returns[s] = 0
                    visits[s] = 0
                
                returns[s] += G
                visits[s] += 1
                grid[s] = returns[s] / visits[s]
        
        if episode % 10 == 0:
            new_policy = compute_policy(grid, terminal_state)
            if np.array_equal(last_policy, new_policy):
                break
            print_grid(grid, policy=new_policy, terminal_state=terminal_state, episode=episode)



def first_visit_montecarlo_manhattan(grid, episodes, terminal_state):
    returns = { (i, j): [] for i in range(grid.shape[0]) for j in range(grid.shape[1]) }

    for _ in range(episodes):
        last_policy = compute_policy(grid, terminal_state)
        state = choose_random_state(grid.shape)
        while state == terminal_state:
            state = choose_random_state(grid.shape)

        episode = []
        while state != terminal_state:
            reward = distance_reward(state, terminal_state)
            episode.append((state, reward))
            state = choose_next_state(grid, state)

        visited_states = set()
        for (s, reward) in episode:
            if s not in visited_states:
                returns[s].append(reward)
                grid[s] = np.mean(returns[s])
                visited_states.add(s)

        if episode % 10 == 0:
            new_policy = compute_policy(grid, terminal_state)
            if np.array_equal(last_policy, new_policy):
                break
            print_grid(grid, policy=new_policy, terminal_state=terminal_state, episode=episode)

def every_visit_montecarlo(grid, episodes=100, terminal_state=None):
    for episode in range(episodes):
        last_policy = compute_policy(grid, terminal_state)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                state = (i,j)
                reward = 0
                if state == terminal_state:
                    continue
                while state != terminal_state:
                    reward -= .1
                    state = choose_next_state(grid, state)
                grid[(i,j)] *= episode
                grid[(i,j)] += reward
                grid[(i,j)] /= (episode+1)

            new_policy = compute_policy(grid, terminal_state)
            print_grid(grid, new_policy, terminal_state, episode)

        if episode % 10 == 0:
            new_policy = compute_policy(grid, terminal_state)
            if np.array_equal(last_policy, new_policy):
                break
            print_grid(grid, policy=new_policy, terminal_state=terminal_state, episode=episode)

def main(size, episodes):
    grid = create_gridworld(size)
    terminal_state = choose_random_state(size)
    print(f"Terminal State: {terminal_state}")
    time.sleep(1)
    
    first_visit_montecarlo(grid, episodes, terminal_state=terminal_state, gamma=0.9)
    
    policy = compute_policy(grid, terminal_state)
    print_grid(grid, policy, terminal_state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridWorld Monte Carlo Skeleton")
    parser.add_argument("--width", type=int, default=5, help="Width of the GridWorld")
    parser.add_argument("--height", type=int, default=5, help="Height of the GridWorld")
    parser.add_argument("--episodes", type=int, default=100, help="episodes for each state")
    args = parser.parse_args()
    
    main((args.width, args.height), args.episodes)