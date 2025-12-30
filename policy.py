import numpy as np
from thistud import create_env
import pygame
import time

# Load the trained Q-table
q_table = np.load("q_table.npy")

# Setup environment
goal_coordinates = (6, 6)
obstacle_coordinates = [(1, 3), (2, 3), (3, 3), (4, 5), (6, 2), (7, 1)] # coordinates are different keeping as is but renaming var
env = create_env(goal_state=goal_coordinates,
                 blockers=obstacle_coordinates,
                 random_initialization=False)

# Run the agent using learned policy
state = env.reset()
state = tuple(state)

for _ in range(100):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()

    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action) # fixed variable state assignment (original code had next_state then state=next_state)
    env.render()
    print(f"State: {state}, Action: {action}, Reward: {reward}, Life: {info['life']}")
    state = tuple(state) # Ensure tuple for q_table lookup if needed next loop (though next loop effectively starts with argmax(q_table[state]))

    if done:
        if reward > 0:
            print("Yehhh! Student Graduated from THI! ")
        else:
            print("oh no student failed in an exam")
        break

env.close()
