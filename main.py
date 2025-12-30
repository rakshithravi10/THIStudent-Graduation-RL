from thistud import create_env
from qlearning import train_q_learning, visualize_q_table

if __name__ == "__main__":
    goal_coordinates = (6, 6)
    obstacle_coordinates = [(0, 3), (2, 1), (3, 7), (3, 4), (6, 3), (5, 0)]
    random_start = False

    env = create_env(goal_state=goal_coordinates,
                     blockers=obstacle_coordinates,
                     random_initialization=random_start)

    train_q_learning(env,
                     no_episodes=1000,
                     epsilon=1.0,
                     epsilon_min=0.015,
                     epsilon_decay=0.995,
                     alpha=0.1,
                     gamma=0.95,
                     q_table_save_path="q_table.npy")

    visualize_q_table(
        blockers=obstacle_coordinates,
        goal_state=goal_coordinates,
        q_values_path="q_table.npy",
        save_path="q_table_heatmap.png",
        show_plot=False
    )