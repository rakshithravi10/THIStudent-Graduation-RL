import numpy as np
from thistud import create_env
import gymnasium as gym

def verify_env_logic():
    print("--- Verifying Environment Logic ---")
    goal = (6, 6)
    obstacles = [(0, 3), (2, 1), (3, 7), (3, 4), (6, 3), (5, 0)]
    env = create_env(goal, obstacles, random_initialization=False)

    # 1. Check reset
    state, info = env.reset()
    print(f"Reset State: {state} (Expected [0 0])")
    if not np.array_equal(state, [0, 0]):
        print("FAIL: Reset did not return [0, 0]")
    
    # 2. Force state to neighbor of goal (6, 5)
    env.agent_state = np.array([6, 5], dtype=np.int32)
    print(f"Forced State: {env.agent_state}")

    # 3. Take Action Up (0) to allow moving to (6, 6)
    print("Taking Action 0 (Up)...")
    next_state, reward, terminated, truncated, _ = env.step(0)
    
    print(f"Next State: {next_state}")
    print(f"Reward: {reward} (Expected 10)")
    print(f"Terminated: {terminated} (Expected True)")
    
    if not np.array_equal(next_state, [6, 6]):
        print("FAIL: Did not reach goal state (6, 6)")
    if reward != 10:
        print("FAIL: Reward is not 10")
    if not terminated:
        print("FAIL: Episode did not terminate at goal!")
    else:
        print("SUCCESS: Episode terminated at goal.")

    # 4. Check what happens if we step FROM goal (should theoretically not happen in loop, but checking behavior)
    env.agent_state = np.array([6, 6], dtype=np.int32)
    next_state_2, reward_2, term_2, trunc_2, _ = env.step(0) # Move Up from Goal
    print(f"Step from Goal (Up): Next={next_state_2}, Reward={reward_2}, Term={term_2}")
    
    # Check Q-table
    print("\n--- Inspecting Q-Table ---")
    try:
        q_table = np.load("q_table.npy")
        print(f"Q-table shape: {q_table.shape}")
        
        with open("verification_results.txt", "w") as f:
            f.write(f"Q-values at Goal (6, 6): {q_table[6, 6]}\n")
            f.write(f"Q-values at Neighbor (6, 5): {q_table[6, 5]}\n")
            
            goal_values = q_table[6, 6]
            if np.any(goal_values != 0):
                 f.write("WARNING: Non-zero Q-values at goal state!\n")
            else:
                 f.write("SUCCESS: Q-values at goal are all zero.\n")
                 
            # Check Step from Goal behavior
            env.agent_state = np.array([6, 6], dtype=np.int32)
            next_s, r, term, trunc, _ = env.step(0)
            f.write(f"Step from Goal (Up): Terminated={term}, Reward={r}\n")

    except FileNotFoundError:
        print("Q-table.npy not found.")

if __name__ == "__main__":
    verify_env_logic()
