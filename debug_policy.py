import numpy as np
from thistud import create_env

def debug_policy():
    try:
        q_table = np.load("q_table.npy")
        print("Q-table loaded successfully.")
    except FileNotFoundError:
        print("Q-table not found!")
        return

    goal_coordinates = (6, 6)
    obstacle_coordinates = [(0, 3), (2, 1), (3, 7), (3, 4), (6, 3), (5, 0)]
    env = create_env(goal_state=goal_coordinates, blockers=obstacle_coordinates, random_initialization=False)

    state, _ = env.reset()
    state = tuple(state)
    path = [state]
    total_reward = 0
    max_steps = 50

    print(f"Start State: {state}")
    print(f"Goal State: {goal_coordinates}")
    print(f"Obstacles: {obstacle_coordinates}")

    for step in range(max_steps):
        action = np.argmax(q_table[state])
        next_state_arr, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = tuple(next_state_arr)
        
        print(f"Step {step+1}: State {state} -> Action {action} -> Next {next_state} | Reward: {reward} | Done: {done}")
        
        state = next_state
        path.append(state)
        total_reward += reward

        if done:
            print("Reached Goal!")
            break
    
    if not done:
        print("Failed to reach goal within max steps.")

    print(f"Total Reward: {total_reward}")
    print(f"Path taken: {path}")

    # Check if Q-table has potential
    print("\nQ-table sample (start state values):")
    print(q_table[0,0])

if __name__ == "__main__":
    debug_policy()
