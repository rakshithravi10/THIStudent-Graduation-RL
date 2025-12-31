import numpy as np
from thistud import create_env
import pygame
import time
import sys
import os

# Load the trained Q-table
q_table_path = "q_table.npy"
if not os.path.exists(q_table_path):
    print(f"Q-table file '{q_table_path}' not found. Run training first.")
    sys.exit(1)

q_table = np.load(q_table_path)

# Setup environment
goal_coordinates = (6, 6)
obstacle_coordinates = [(0, 3), (2, 1), (3, 7), (3, 4), (6, 3), (5, 0)]
env = create_env(goal_state=goal_coordinates,
                 blockers=obstacle_coordinates,
                 random_initialization=False)

state, _ = env.reset()
state = tuple(int(s) for s in state)

for _ in range(100):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()

    action = int(np.argmax(q_table[state]))
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    print(f"State: {next_state}, Action: {action}, Reward: {reward}, Life: {info['life']}")
    state = tuple(int(s) for s in next_state)

    if done:
        if reward > 0:
            print("Yehhh! Student Graduated from THI! ")
        else:
            print("oh no student failed in an exam")
        break

env.close()
