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
    """Determines the best action based on the highest-value neighboring state."""
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
    """Computes a policy where each state points to the highest-value neighbor."""
    policy = np.full(grid.shape, '·', dtype='<U1')
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if (i, j) == terminal_state:
                policy[i, j] = 'T'  # Mark terminal state
            else:
                policy[i, j] = get_best_action(grid, i, j)
    return policy


def print_grid(grid, policy=None, start_state=None, terminal_state=None):
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear terminal
    print(f"Start State: {start_state}, Terminal State: {terminal_state}\n")
    print("GridWorld State Values:              Policy:")
    for i in range(grid.shape[0]):
        row_values = " ".join(f"{grid[i, j]:6.2f}" for j in range(grid.shape[1]))
        row_policy = " ".join(policy[i]) if policy is not None else ""
        print(f"{row_values}    {row_policy}")
    print("\nUpdating...\n")
    time.sleep(0.01)

def dummy_explore(grid, episodes=10, start_state=None, terminal_state=None):
    size = grid.shape
    for _ in range(episodes):
        state = choose_random_state(size)
        for _ in range(10):  # Simulate actions for 10 steps
            next_state = choose_random_state(size)
            grid[next_state] += random.uniform(-0.1, 0.1)  # Update value randomly
            policy = compute_policy(grid, terminal_state)
            print_grid(grid, policy, start_state, terminal_state)

def main(size):
    grid = create_gridworld(size)
    start_state = choose_random_state(size)
    terminal_state = choose_random_state(size)
    print(f"Start State: {start_state}, Terminal State: {terminal_state}")
    time.sleep(1)
    
    dummy_explore(grid, start_state=start_state, terminal_state=terminal_state)
    
    policy = compute_policy(grid, terminal_state)
    print_grid(grid, policy, start_state, terminal_state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridWorld Monte Carlo Skeleton")
    parser.add_argument("--width", type=int, default=5, help="Width of the GridWorld")
    parser.add_argument("--height", type=int, default=5, help="Height of the GridWorld")
    args = parser.parse_args()
    
    main((args.width, args.height))