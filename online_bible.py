# https://gymnasium.farama.org/environments/toy_text/frozen_lake/

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os

## envs:

# simple_grid

# windy_grid

# slippery_grid

# blackjack

# pacman


## algorithms:

# first_visit_montecarlo_manhattan

# first_visit_montecarlo

# every_visit_montecarlo

# td_0

# td_lambda

# sarsa

# sarsa_lambda

# q_learning

# q_lambda


def main(N):

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridWorld Monte Carlo Skeleton")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to run")
    args = parser.parse_args()
    
    main(args.episodes)