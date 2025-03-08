import numpy as np
import matplotlib.pyplot as plt
import time
import random

# missing policy visualisation, need to manually close the mathplotlib visualisation each time and i dont want it, missing argument parsing for custom dimenstions

def create_gridworld(size):
    return np.zeros(size)

def choose_random_state(size):
    w, h = size
    return random.randint(0, w - 1), random.randint(0, h - 1)

def visualize(grid, title="GridWorld", policy=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="Blues_r", origin="upper")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if policy is not None:
                plt.text(j, i, policy[i, j], ha='center', va='center', color='black')
            else:
                plt.text(j, i, f"{grid[i, j]:.2f}", ha='center', va='center', color='black')
    plt.title(title)
    plt.colorbar()
    plt.show()

def dummy_explore(grid, episodes=10):
    size = grid.shape
    for _ in range(episodes):
        state = choose_random_state(size)
        for _ in range(10):  # Simula un'azione per 10 passi
            next_state = choose_random_state(size)
            grid[next_state] += random.uniform(-0.1, 0.1)  # Aggiorna valore random
            visualize(grid, title="Exploration Values")
            time.sleep(0.5)  # Pausa per visualizzazione

def main(size=(5,5)):
    grid = create_gridworld(size)
    start_state = choose_random_state(size)
    terminal_state = choose_random_state(size)
    print(f"Start State: {start_state}, Terminal State: {terminal_state}")
    
    dummy_explore(grid)
    
    # Creazione di una policy fittizia per la visualizzazione
    policy = np.full((size, size), "→")  # Esempio di policy con frecce
    visualize(grid, title="Final State Values", policy=policy)

if __name__ == "__main__":
    main()
