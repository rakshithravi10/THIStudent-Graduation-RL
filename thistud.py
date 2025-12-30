import numpy as np 
import pygame
import gymnasium as gym
import time
import random

class THIStudentEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, grid_size=8, cell_size=80):
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size
        self.agent_state = np.array([0, 0], dtype=np.int32)
        self.goal_state = np.array([6, 6], dtype=np.int32)
        # Default obstacles (tuples)
        self.obstacles = [(0, 3), (2, 1), (3, 7), (3, 4), (6, 3), (5, 0)]
        self.life = 100

        # Discrete actions
        self.action_space = gym.spaces.Discrete(4)
        # Multidiscrete for two discrete coordinates
        self.observation_space = gym.spaces.MultiDiscrete([self.grid_size, self.grid_size])

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("THI Student Graduation Journey")
        self.clock = pygame.time.Clock()

        # image
        try:
            self.student_img = pygame.transform.scale(pygame.image.load("student.png"), (cell_size, cell_size))
            self.graduation_img = pygame.transform.scale(pygame.image.load("graduation.png"), (cell_size, cell_size))
            self.obstacle_img = pygame.transform.scale(pygame.image.load("failedsubject.png"), (cell_size, cell_size))
            self._images_loaded = True
        except Exception:
            self._images_loaded = False

        # Allow optional random start 
        self.random_initialization = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Provide deterministic reset unless random_initialization True
        if self.random_initialization:
            # pick a random cell that is not an obstacle and not the goal
            candidates = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                          if (x, y) not in self.obstacles and (x, y) != tuple(self.goal_state)]
            sx, sy = random.choice(candidates)
            self.agent_state = np.array([sx, sy], dtype=np.int32)
        else:
            self.agent_state = np.array([0, 0], dtype=np.int32)

        self.life = 100
        return self.agent_state.copy(), {}

    def step(self, action):
        new_state = self.agent_state.copy()

        # Action mapping 0=Up, 1=Down, 2=Left, 3=Right
        if action == 0 and new_state[1] < self.grid_size - 1:
            new_state[1] += 1  # up
        elif action == 1 and new_state[1] > 0:
            new_state[1] -= 1  # down
        elif action == 2 and new_state[0] > 0:
            new_state[0] -= 1  # left
        elif action == 3 and new_state[0] < self.grid_size - 1:
            new_state[0] += 1  # right

        self.agent_state = new_state
        reward = -1
        terminated = False
        truncated = False

        if np.array_equal(self.agent_state, self.goal_state):
            reward = 10
            terminated = True
        elif tuple(self.agent_state) in self.obstacles:
            reward = -5
            self.life = 0
            terminated = True  # terminate episode on failure

        info = {"distance_to_goal": (self.goal_state - self.agent_state).tolist(), "life": int(self.life)}
        return self.agent_state.copy(), int(reward), bool(terminated), bool(truncated), info

    def render(self):
        self.screen.fill((255, 255, 255))

        # grid
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (self.window_size, y))

        # obstacles
        for bx, by in self.obstacles:
            px = bx * self.cell_size
            py = (self.grid_size - 1 - by) * self.cell_size
            if self._images_loaded:
                self.screen.blit(self.obstacle_img, (px, py))
            else:
                pygame.draw.rect(self.screen, (200, 50, 50), (px, py, self.cell_size, self.cell_size))

        # goal
        gx, gy = self.goal_state
        gx_px = gx * self.cell_size
        gy_px = (self.grid_size - 1 - gy) * self.cell_size
        if self._images_loaded:
            self.screen.blit(self.graduation_img, (gx_px, gy_px))
        else:
            pygame.draw.rect(self.screen, (50, 200, 50), (gx_px, gy_px, self.cell_size, self.cell_size))

        # agent
        ax, ay = self.agent_state
        ax_px = ax * self.cell_size
        ay_px = (self.grid_size - 1 - ay) * self.cell_size
        if self._images_loaded:
            self.screen.blit(self.student_img, (ax_px, ay_px))
        else:
            pygame.draw.circle(self.screen, (50, 50, 200), (ax_px + self.cell_size // 2, ay_px + self.cell_size // 2), self.cell_size // 3)

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        pygame.quit()


def create_env(goal_state,
               blockers,  
               random_initialization=False):
    env = THIStudentEnv()
    env.goal_state = np.array(goal_state, dtype=np.int32)
    # Ensure blockers are tuples
    env.obstacles = [tuple(b) for b in blockers]
    env.random_initialization = bool(random_initialization)
    return env


if __name__ == "__main__":
    env = THIStudentEnv()
    state, _ = env.reset()
    for _ in range(200):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"State: {state}, Action: {action}, Reward: {reward}, Life: {info['life']}")
        if terminated or truncated:
            if reward > 0:
                print("Yehhh! Student Graduated from THI!")
            else:
                print("oh no student failed in an exam")
            break

    env.close()