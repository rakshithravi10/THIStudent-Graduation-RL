import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def train_q_learning(env,
                     no_episodes=1000,
                     epsilon=1.0,
                     epsilon_min=0.015,
                     epsilon_decay=0.995,
                     alpha=0.1,
                     gamma=0.95,
                     max_steps_per_episode=100,
                     q_table_save_path="q_table.npy"):

    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    success_count = 0

    for episode in range(no_episodes):
        state = env.reset()
        state = tuple(state)
        total_reward = 0

        for step in range(max_steps_per_episode):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            
           # env.render()

            next_state = tuple(next_state)
            total_reward += reward

            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            state = next_state

            if done:
                if reward == 10:
                    success_count += 1
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        # Only print every 100 episodes to reduce noise
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()
    np.save(q_table_save_path, q_table)
    print("Training finished and Q-table saved.")
    print(f"Success rate: {success_count}/{no_episodes}")


def visualize_q_table(blockers=[(1, 3), (2, 3), (3, 3), (4, 5), (6, 2), (7, 1)],
                      goal_state=(6, 6),
                      actions=["Up", "Down", "Left", "Right"],
                      q_values_path="q_table.npy",
                      save_path=None,
                      show_plot=True):

    try:
        q_table = np.load(q_values_path)
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            mask = np.zeros_like(heatmap_data, dtype=bool)

            def grid_coord(coord):
                x, y = coord
                return (y, x)

            mask[grid_coord(goal_state)] = True
            for h in blockers:
                mask[grid_coord(h)] = True

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

            ax.invert_yaxis()
            ax.set_yticks(np.arange(q_table.shape[0]) + 0.5)
            ax.set_yticklabels([str(i) for i in range(q_table.shape[0])])

            gy, gx = grid_coord(goal_state)
            ax.text(gx + 0.5, gy + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)

            for h in blockers:
                hy, hx = grid_coord(h)
                ax.text(hx + 0.5, hy + 0.5, 'O', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f"Action: {action}")

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Q-table visualization saved to {save_path}")

        if show_plot:
            plt.show(block=True)

    except FileNotFoundError:
        print("Q-table file not found. Train the agent first.")
