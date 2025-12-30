import numpy as np
import pygame
import gymnasium as gym
import time

class THIStudentEnv(gym.Env): 
    def __init__(self, grid_size=8, cell_size=80):
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size
        self.agent_state = np.array([0, 0])  
        self.goal_state = np.array([6, 6])
        self.obstacles = [(0, 3), (2, 1), (3, 7), (3, 4), (6, 3), (5, 0)] # Changed blockers to obstacles (failed subjects)
        self.life = 100

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=grid_size, shape=(2,), dtype=np.int32)

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("THI Student Graduation Journey") # Updated caption
        self.clock = pygame.time.Clock()

        # Updated image assets
        self.student_img = pygame.transform.scale(pygame.image.load("student.png"), (cell_size, cell_size))
        self.graduation_img = pygame.transform.scale(pygame.image.load("graduation.png"), (cell_size, cell_size))
        self.obstacle_img = pygame.transform.scale(pygame.image.load("failedsubject.png"), (cell_size, cell_size))

    def reset(self):
        self.agent_state = np.array([0, 0])
        self.life = 100
        return self.agent_state

    def step(self, action):
        new_state = self.agent_state.copy()

        if action == 0 and new_state[1] < self.grid_size - 1:
            new_state[1] += 1  # up
        elif action == 1 and new_state[1] > 0:
            new_state[1] -= 1  # down
        elif action == 2 and new_state[0] > 0:
            new_state[0] -= 1  # left
        elif action == 3 and new_state[0] < self.grid_size - 1:
            new_state[0] += 1  # right

        self.agent_state = new_state

        reward = -1  # default step penalty
        done = False 

        if np.array_equal(self.agent_state, self.goal_state):
            reward = 10 # Graduation reward
            done = True 

        elif tuple(self.agent_state) in self.obstacles:
            reward = -5
            self.life = 0 # Failed subject
        
        info = {"distance to goal": self.goal_state - self.agent_state, "life": self.life}
        return self.agent_state, reward, done, info

    def render(self):
        self.screen.fill((255, 255, 255)) 

        for x in range(0, self.window_size, self.cell_size): 
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size): 
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (self.window_size, y))

        for bx, by in self.obstacles:
            self.screen.blit(self.obstacle_img, (bx * self.cell_size, (self.grid_size - 1 - by) * self.cell_size))

        gx, gy = self.goal_state
        self.screen.blit(self.graduation_img, (gx * self.cell_size, (self.grid_size - 1 - gy) * self.cell_size))

        ax, ay = self.agent_state
        self.screen.blit(self.student_img, (ax * self.cell_size, (self.grid_size - 1 - ay) * self.cell_size))

        pygame.display.flip() 
        self.clock.tick(10)
        time.sleep(0)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = THIStudentEnv() 
    state = env.reset() 
    for _ in range(200): 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        print(f"State: {state}, Action: {action}, Reward: {reward}, Life: {info['life']}") 

        if done:
            print("Yehhh! Student Graduated from THI!")
            break

    env.close()

def create_env(goal_state,
               blockers, # Keeping arg name blockers for compatibility
               random_initialization):
    # Ignoring random_initialization 
    env = THIStudentEnv()
   
    env.goal_state = np.array(goal_state)
    env.obstacles = blockers
    return env